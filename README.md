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



## 这次的更新

我又又又又又修改了代码, 现在的代码可以实现训练和预测了, 预测的比较准确, 但生产图像的程序有错误, 详见下面的"**目前的问题**"

2023.3.2 -> 添加了predict.py程序, 是预测结果的程序, 该程序像train.py ( 和test2.py ) 程序一样, 输入的图片以**数据集的形式**来预测, 同时输入的图片为**.gw格式**. 

​	**不**使用**单张图片**和**普通图片**预测的原因是因为会报错, 我还没有修复这个错误. 在程序中数据集的位置如下:

<img src=".\image\6.png" alt="071924d6bfff74e1e9b9523c8be7119" style="zoom:33%;" />



​	程序首先会加载模型的权重, 我把权重文件放在了 ***./checkpoint*** 里 ( 注意这个权重是测试用的, 不是最终的权重 ), 使用时**请修改权重名称**(如果直接用的我的权重那么不用改名), 其在程序中的位置如下:

<img src=".\image\5.png" alt="9148e77a841982f8930fd70b306dd92" style="zoom:25%;" />

​	预测完毕后, 预测好的图像在 ***./predict*** 文件夹内, reference points 和 原图叠加的图片在 ***./mark_*** 内

2023.3.16 -> 一次大更新:

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
