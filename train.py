import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import time
import random

from network import FiducialPoints, DilatedResnetForFlatByFiducialPointsS2
from dataloader import DewarpingDataset
from loss import Losses
from utilsV4 import SaveFlatImage

if __name__ == '__main__':
    # 定义训练设备
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(device)

    batch_size = 4
    postprocess_list = ['tps', 'interpolation']

    # 加载数据集
    train_data = DewarpingDataset(root=r"./dataset", split='train')
    test_data = DewarpingDataset(root=r"./dataset", split='test')
    val_data = DewarpingDataset(root=r'./dataset', split='val')
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=min(batch_size, 8), drop_last=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=min(batch_size, 8), drop_last=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=min(batch_size, 8), drop_last=True, pin_memory=True)

    # 数据集大小
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    val_data_size = len(val_data)
    print(f"训练集长度: {train_data_size}")
    print(f"验证集长度: {val_data_size}")
    print(f"测试集长度: {test_data_size}")

    # 创建网络模型
    n_classes = 2
    model = FiducialPoints(n_classes=n_classes, num_filter=32, architecture=DilatedResnetForFlatByFiducialPointsS2, BatchNorm='BN', in_channels=3)
    model.to(device)
    save_flat_mage = SaveFlatImage(r'./', None, None, None, r'./dataset/val', r'./dataset/test', batch_size, preproccess=False, postprocess=postprocess_list[0], device=torch.device(device))

    # 损失函数
    loss_fun_classes = Losses(classify_size_average=True, args_gpu=device)
    loss_fun = loss_fun_classes.loss_fn4_v5_r_4   # *
    loss_fun2 = loss_fun_classes.loss_fn_l1_loss

    # 迭代器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, weight_decay=1e-12)

    # 学习率优化器
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 90, 150, 200], gamma=0.5)

    # 设置网络的一些参数
    # 记录训练次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0

    # 训练轮次
    epoch = 10

    # 添加tensorboard
    writer = SummaryWriter("logs")
    start_time = time.time()

    for i in range(epoch):
        print(f"----------第{i+1}轮训练开始----------")

        # 开始训练
        model.train()   # 当网络中有dropout层、batchnorm层时，这些层能起作用
        for images, labels, segment in train_loader:
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            segment = Variable(segment.to(device))
            optimizer.zero_grad()   # 梯度清零

            outputs, outputs_segment = model(images)

            loss_l1, loss_local, loss_edge, loss_rectangles = loss_fun(outputs, labels, size_average=True)
            loss_segment = loss_fun2(outputs_segment, segment)
            loss = loss_l1 + loss_local + loss_edge + loss_rectangles + loss_segment

            loss.backward()         # 反向传播, 计算损失函数的梯度
            optimizer.step()        # 根据梯度, 对网络的参数进行调优

            total_train_step =  total_train_step + 1
            if total_train_step % 4 ==0:
                end_time = time.time()
                print(f"训练时间: {end_time - start_time}")    # 运行训练一百次后的时间间隔
                print(f"训练次数: {total_train_step}, Loss: {loss}")
                writer.add_scalar("train_loss", loss, total_train_step)
                print()
            
            try:
                scheduler.step()
            except:
                pass


        # 测试步骤开始(每一轮训练后都查看在测试数据集上的Loss情况)
        model.eval()    # 当网络中有dropout层, batchnorm层时, 这些层不能起作用
        total_test_step = 0
        total_test_loss = 0
        with torch.no_grad():   # 不更新梯度
            for images, labels, segment in val_loader:    # 测试集提取数据
                try:
                    save_img_ = random.choices([True, False], weights=[0.05, 0.95])[0]  # 不知道什么意思

                    images = Variable(images.cuda(device))
                    labels = Variable(labels.cuda(device))
                    segment = Variable(segment.cuda(device))

                    outputs, outputs_segment = model(images)

                    loss_overall, loss_local, loss_edge, loss_rectangles = loss_fun(outputs, labels, size_average=True)
                    loss_segment = loss_fun2(outputs_segment, segment)

                    loss = loss_overall + loss_local + loss_edge + loss_rectangles + loss_segment

                    pred_regress = outputs.data.cpu().numpy().transpose(0, 2, 3, 1)         # (4, 1280, 1024, 2)
                    pred_segment = outputs_segment.data.round().int().cpu().numpy()  # (4, 1280, 1024)  ==outputs.data.argmax(dim=0).cpu().numpy()

                    if save_img_:
                        save_flat_mage.flatByRegressWithClassiy_multiProcessV2(pred_regress,
                                                                               pred_segment, f"image{total_test_step}",
                                                                               epoch + 1,
                                                                               perturbed_img=images.numpy(), scheme='validate', is_scaling=False)

                except:
                    print('* save image validated error')
                
                total_test_loss = total_test_loss + loss # 所有loss


        print(f"整体测试集上的Loss: {total_test_loss}")
        print(f"测试集的平均Loss: {total_test_loss/test_data_size}")
        writer.add_scalar("total_test_loss", total_test_loss, total_test_step)
        total_test_step = total_test_step + 1

        torch.save(model.state_dict(), f"./checkpoint/Dewarp_epoch{i}.pth")    # 保存每一轮训练后的结果
        print("模型已保存")

    writer.close()
