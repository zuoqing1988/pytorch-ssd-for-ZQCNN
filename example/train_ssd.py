import argparse
import os,sys
import logging
import itertools
import configparser
import torch
import cv2
from torch.utils.tensorboard import SummaryWriter
import numpy as np
sys.path.append(os.getcwd())

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from core.utils.box_utils_zq import SSDSpec, SSDBoxSizes, generate_ssd_priors
from core.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from core.ssd import MatchPrior
from core.ssd_creator import create_ssd
from core.datasets.voc_dataset import VOCDataset
from core.datasets.open_images import OpenImagesDataset
from core.multibox_loss import MultiboxLoss
from core.data_preprocessing import TrainAugmentation, TestTransform


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--config_file", type=str, help='Specify config file')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")

parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")


# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,150", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=200, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--iou_thresh', default=0.5, type=float,
                    help='iou thresh')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=200, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=10, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--gpus_id', default='0', type=str,
                    help='GPUS ID')


parser.add_argument('--model_dir', default='models/',
                    help='Directory for saving checkpoint models')


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')




def train(writer, start_step, loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    cur_step = start_step
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels, _, _ = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            cur_step += debug_steps
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            writer.add_scalar('loss',avg_loss, cur_step)
            writer.add_scalar('reg_loss',avg_reg_loss, cur_step)
            writer.add_scalar('cls_loss',avg_clf_loss, cur_step)
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
    return cur_step

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels, _, _ = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':

    print('train_ssd')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENCV_OPENCV_RUNTIME'] = 'disabled'
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    args = parser.parse_args()
    logging.info(args)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    # load config file and setup
    params = {}
    config = configparser.ConfigParser()
    config.read(args.config_file)

    #print(config)

    for _ in config.options("Train"):
        params[_] = eval(config.get("Train",_))



    # image_size_x, image_size_y, image_mean, image_std, use_gray
    image_size_x = int(params['image_size_x'])
    image_size_y = int(params['image_size_y'])
    image_std = float(params['image_std'])
    image_mean = params['image_mean']
    use_gray = bool(params['use_gray'])
    
    mean_values = list()
    for i in range(len(image_mean)):
        mean_values.append(float(image_mean[i]))
    image_mean = np.array(mean_values,dtype=np.float)

    # iou_thresh, center_varaiance, size_variance
    iou_thresh = float(params['iou_thresh'])
    center_variance = float(params['center_variance'])
    size_variance = float(params['size_variance'])
    
    # backbone type, header type
    backbone_type = params['backbone_type']
    header_type = params['header_type']
    
    # aspect_ratios
    aspect_ratios = params['aspect_ratios']
    if aspect_ratios is None:
        aspect_ratios = list()
    else:
        ratios = list()
        if type(aspect_ratios) == tuple:
            for j in range(len(aspect_ratios)):
                ratios.append(float(aspect_ratios[j]))
        else:
            ratios.append(float(aspect_ratios))
        aspect_ratios = ratios

    # specs
    specs = list()
    for i in range(1,100):
        name = 'spec%d'%i
        if not name in params:
            break
        line = params[name]
        feat_map_x = int(line[0])
        feat_map_y = int(line[1])
        shrinkage_x = int(line[2])
        shrinkage_y = int(line[3])
        bbox_min = float(line[4])
        bbox_max = float(line[5])
        specs.append(SSDSpec(feat_map_x, feat_map_y, shrinkage_x, shrinkage_y, SSDBoxSizes(bbox_min, bbox_max), aspect_ratios))
        
    # priors
    priors = generate_ssd_priors(specs, image_size_x, image_size_y)



    #
    train_transform = TrainAugmentation(image_size_x, image_size_y, image_mean, image_std)

    target_transform = MatchPrior(priors, center_variance, size_variance, args.iou_thresh)

    test_transform = TestTransform(image_size_x, image_size_y, image_mean, image_std)

    timer = Timer()

    # device
    if torch.cuda.is_available() and args.use_cuda:
        DEVICE = torch.device('cuda:'+args.gpus_id)
        print('use cuda')
    else:
        DEVICE = torch.device("cpu")
        print('use cpu')

    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("Use Cuda.")

    training_name = '%s-%s-%dx%d'%(backbone_type,header_type,image_size_x,image_size_y)
    training_dir = os.path.join(args.model_dir, training_name)
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)


    logging.info("Prepare training datasets.")
    datasets = []
    print(args.datasets)
    for dataset_path in args.datasets:
        print(dataset_path)
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, use_gray, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(args.model_dir, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path,
                 transform=train_transform, target_transform=target_transform,
                 dataset_type="train", balance_data=args.balance_data)
            label_file = os.path.join(args.model_dir, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            logging.info(dataset)
            num_classes = len(dataset.class_names)

        else:
            raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
        datasets.append(dataset)


    
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    logging.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, use_gray, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif args.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
        logging.info(val_dataset)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    logging.info("Build network.")

    # create ssd net
    net = create_ssd(backbone_type, header_type, use_gray, num_classes, aspect_ratios, priors, center_variance, size_variance, is_test=False, device=DEVICE)

    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(priors, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    writer = SummaryWriter()
    start_step = 0
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):

        start_step = train(writer, start_step, train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        scheduler.step()
        
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_reg_loss', val_regression_loss, epoch)
            writer.add_scalar('val_cls_loss', val_classification_loss, epoch)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(training_dir, f"Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")

    writer.close()


