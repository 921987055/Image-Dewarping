# Imgae Dewarping

​	这是我们小组的大创项目, 目前我们的项目还没有做完

## Usage:

#### 训练前

​	在 dataset/train/ 和 dataset/val/ 目录下放好数据集文件(文件格式为.gw, 数据集的生成详见底下的DATASET说明), 并删除掉 README.md 文件

#### 训练

​	运行 train.py 进行训练:

```bash
python train.py
```

​	其保存的权重文件存储在 checkpoint 文件夹, 若要继续训练之前的权重, 请修改train.py中的continue_train变量为True

#### 预测

​	在 dataset/test/ 目录下存放需要预测的图片文件(最好是.png, 我不知道其他文件格式可不可以), 运行主目录下的 predict.py 进行预测

## Requirements:

<p>python >=3.7</p>
<p>pytorch</p>
<p>opencv-python</p>
<p>scipy</p>

## DATASET

The training dataset can be synthesised using the [scripts](https://github.com/gwxie/Synthesize-Distorted-Image-and-Its-Control-Points).



文件夹:

- checkpoint : 保存权重的文件夹
- dataset: 存储数据集
  - train: 训练集
  - val: 验证集
  - test: 预测的.png图片请放在这里
- image: 存储在github上的图片
- Logs: tensorboard的输出目录
- mark_: 预测出的带有预测点的图片
- predict: 预测的结果图片

代码:

- dataloader.py: dataloader的代码
- debug_predict.py: 输入是.gw的预测代码(这里预测的输入的.gw文件其实就是 dataset/train/*.gw)
- loss.py: loss函数的代码
- network: 网络的代码
- predict.py: 输入是图片格式的代码(如果报错了尝试将图片转为.png)
- tool.py tpsV2.py utilsV4.py三个额外的需要import的脚本
- train.py 训练的代码

同时tran.py中训练的代码的Loss是直接使用变量来记录的, 这样做不好管理, 日后会使用List链表来管理Loss(或创建一个类)

​	
