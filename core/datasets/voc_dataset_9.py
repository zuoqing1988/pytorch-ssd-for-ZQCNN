import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os


class VOCDataset:

    def __init__(self, root, use_gray, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.use_gray = use_gray
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list
            
            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file: " + label_file_name)
            sys.exit()


        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        #print(boxes,labels,is_difficult)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id, self.use_gray)
        if self.transform:
            #print(image_id)
            image, boxes, labels = self.transform(image, boxes, labels)
        labels_mid = None
        labels_low = None
        if self.target_transform:
            #print(image_id)
            boxes, labels, labels_mid, labels_low, is_peer = self.target_transform(boxes, labels)
        return image, boxes, labels, labels_mid, labels_low, is_peer

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id, self.use_gray)
        if self.transform:
            #print(image_id)
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        #print(annotation_file)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_attr = object.find('difficult')
                if is_difficult_attr is None:
                    is_difficult.append(0)
                else:
                    is_difficult_str = is_difficult_attr.text
                    is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
        #if len(boxes) == 0:
        #    print(image_id + ' has no valid label')
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id, use_gray):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        if use_gray:
            image = cv2.imread(str(image_file),0)
        else:
            image = cv2.imread(str(image_file))
            if image.ndim == 2:
                image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        return image



