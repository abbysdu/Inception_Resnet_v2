import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datasets import ImageNet
from inception_resnet_v2 import Inception_ResNet_v2
from tensorboardX import SummaryWriter

#---------------------#
#    settings         #
#---------------------#
import argparse

# 获取超参数
parser = argparse.ArgumentParser(description='traing paremeters')
parser.add_argument('--start_epoch',default=0, type=int, help='start')
parser.add_argument('--path', help='resume path')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--epoch', default=100,type=int, help='total epoches')
parser.add_argument('--batch_size', default=32, help='batch size')
args = parser.parse_args()
print(args)

#---------------------#
#      ImageNet       #
#---------------------#
dataset = ImageNet('/home/datacenter/ssd2/ImageNet_val', '/home/zhaiyize/models/resnet_inception/data/imagenet_classes.txt', '/home/zhaiyize/models/resnet_inception/data/imagenet_2012_validation_synset_labels.txt')
traindata = Subset(dataset, range(0,int(0.8*len(dataset))))
valdata = Subset(dataset, range(int(0.8*len(dataset)),len(dataset)))

train_data = DataLoader(dataset=traindata,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=4)
val_data = DataLoader(dataset=valdata,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=4)

for img, label in val_data:
    print('Image batch dimensions:', img.shape)
    print('Image label dimensions:', label.shape)
    break

writer = SummaryWriter()
images, labels = next(iter(val_data))
writer.add_graph(Inception_ResNet_v2(), images)

from tqdm import tqdm
import os
if "CUDA_VISIBLE_DEVICE" not in os.environ.keys():
    os.environ["CUDA_VISIBLE_DEVICE"]="0"

# 判断是否重新训练
if args.start_epoch > 0:
    print('-----resuming from logs------')
    state = torch.load(args.path)
    net = state['net']
    best_acc = state['acc']
else:
    print('-----build new model-----')
    net = Inception_ResNet_v2()
    best_acc = 0
    
# 设置训练模式
use_cuda = torch.cuda.is_available()
ids=[int(i) for i in range(len(os.environ["CUDA_VISIBLE_DEVICE"].split(",")))]
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=ids)
    torch.backends.cudnn.benchmark = True
    
# 定义度量和优化
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# 训练
def train(epoch):
    train_loss = 0
    correct = 0
    total = 0
    # 训练阶段
    print(net)
    net.train()
    loop_train = tqdm(enumerate(train_data), total =len(train_data))
    for batch_idx,(inputs, targets)in loop_train:
        # 将数据转移到gpu上
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        # 模型输出，更新梯度
        optimizer.zero_grad()
        pred = net(inputs)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
        
        # 数据统计
        train_loss += loss.item()
        _, predict = torch.max(pred.data, 1)  # 找出行最大值
        total += targets.size(0)
        correct += predict.eq(targets.data).cpu().sum()
        loss = train_loss/(batch_idx+1)
        acc = 100.*correct/total
        writer.add_scalar('Loss/train', loss, batch_idx)
        writer.add_scalar('Loss/val', loss, batch_idx)
        writer.add_scalar('Accuracy/train', acc, batch_idx)
        writer.add_scalar('Accuracy/val', acc, batch_idx)
        # loss = compute_epoch_loss(net, train_loader)
        # acc = compute_accuracy(net, train_loader)
        loop_train.set_description(f'Epoch_train [{epoch}/{args.start_epoch+args.epoch}]')
        loop_train.set_postfix(loss = loss,acc = acc)
        # print('\n%d epoch done!'%epoch+1, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# 测试阶段
def test(epoch):
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    # 训练阶段
    net.eval()
    loop_test = tqdm(enumerate(val_data), total =len(val_data))
    for batch_idx,(inputs, targets)in loop_test:
        # 将数据转移到gpu上
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # 数据统计
        test_loss += loss.item()
        _, predict = torch.max(outputs.data, 1)  # 找出行最大值
        total += targets.size(0)
        correct += predict.eq(targets.data).cpu().sum()
        loss = test_loss/(batch_idx+1)
        acc = 100.*correct/total
        loop_test.set_description(f'Epoch_test [{epoch}/{args.start_epoch+args.epoch}]')
        loop_test.set_postfix(loss = loss,acc = acc)

    # 保存模型
    acc = 100.*correct/total
    if acc > best_acc:
        print('==============>Saving...')
        cur_net = {
                'net': net.module if use_cuda else net,
                'acc': acc
            }
        if not os.path.exists('log'):
            os.mkdir('log')
        torch.save(cur_net, './log/epoch%d_acc%4f_loss%4f.pth'%(epoch+1,acc,loss))
        best_acc = acc

if __name__ =="__main__":
    for epoch in range(args.start_epoch, args.start_epoch+args.epoch):
        train(epoch)
        test(epoch)
        # 清除部分无用变量 
        torch.cuda.empty_cache()