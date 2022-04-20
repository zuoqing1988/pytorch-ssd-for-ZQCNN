# pytorch-ssd-for-ZQCNN

Pytorch训练SSD

参考的代码是https://github.com/qfgaohao/pytorch-ssd 但是已经改动非常大了，两者不兼容

# 运行环境

pytorch1.6.0

其他缺啥装啥

# 数据集

VOC 格式

比如用VOC格式的wider

链接：https://pan.baidu.com/s/1vKPyPBVoCEDiKhUd_eakZg 
提取码：hw6m 

# 数据准备

进入VOC格式数据集路径（此时应该能看到如下目录），以下称此目录为VOC_ROOT

	annotations
	ImageSets
	JPEGImages
	
	
运行如下代码

	python /path/to/pytorch-ssd-zq/core/datasets/generate_vocdata.py label_file
	
其中label_file是一个文件，内容是类别名（不包含__BACKGROUND__），以','隔开

我写的label_file是这样的, 比如有3类，dog, cat, person, 我写成3行

	dog,
	cat,
	person


训练时把label_file放在VOC数据里面，和JPEGImages同目录


# 训练


进入本项目路径

	python example/train_ssd9.py \
	            --config_file configs/model-face.cfg \
	            --datasets VOC_ROOT \
	            --validation_dataset VOC_ROOT \
	            --batch_size 128 \
	            --num_epochs 200 \
	            --lr 0.01 \
	            --gpus_id 0
				
如果用fp16 需要加上参数

	--fp16 True

如果用带fpn的模型,用下面的命令

	python example_fpn/train_ssd_fpn9.py \
	            --config_file configs_fpn/ssd_fpn_zq14.cfg \
	            --datasets VOC_ROOT \
	            --validation_dataset VOC_ROOT \
	            --batch_size 128 \
	            --num_epochs 200 \
	            --lr 0.01 \
	            --gpus_id 0
				
**多个VOC数据一起训练**

--datasets后面接多个VOC数据集，用逗号隔开

	python example/train_ssd9.py \
	            --config_file configs/model-face.cfg \
	            --datasets VOC_ROOT1,VOC_ROOT2,VOC_ROOT3,VOC_ROOT4 \
	            --validation_dataset VOC_ROOT1 \
	            --batch_size 128 \
	            --num_epochs 200 \
	            --lr 0.01 \
	            --gpus_id 0

# 测试模型精度

进入本项目路径

	python example/eval_ssd.py \
	            --config_file configs/zq3.cfg \
	            --trained_model YOUR_MODEL \
	            --dataset VOC_ROOT \
	            --use_cuda True \
	            --gpus_id 0 \
	            --label_file YOUR_LABEL_FILE
				
# 测试单张图

进入本项目路径

	python example/run_ssd_example.py config_file model_file label_file image_file
	

# 导出onnx模型

进入本项目路径

	python example/pth2onnx.py config_file in_file out_file num_valid_classes withsoftmax

# onnx模型简化

用https://github.com/daquexian/onnx-simplifier

python -m onnxsim in_file out_file

# 推理

**ZQCNN**

ZQCNN里有能加载的代码，转模型用https://github.com/zuoqing1988/ZQCNN/tree/master/onnx_to_ZQCNN 里的脚本

示例代码在https://github.com/zuoqing1988/ZQCNN/tree/master/SamplesZQCNN/SampleSSDDetectorPytorch


