from model_zoo.vgg import VGG

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import os
from tqdm import tqdm

personalize = True
if personalize:
    from enrico_utils.get_data import get_dataloader
    (trainloader, val_loader, test_loader), weights = get_dataloader("enrico_corpus")
else:
    transform = transforms.Compose(
    [transforms.ToTensor(),     # 将图片转成tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   # 将图片有[0,1]转成[-1,1]

    trainset = torchvision.datasets.CIFAR10(root='./cv/cifar10',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=32,
                                            shuffle=True,
                                            num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./cv/cifar10',
                                            train=False,
                                            download=True,
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=32,
                                            shuffle=False,
                                            num_workers=0)

#print(train_loader)




# 在GPU上训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = VGG('VGG16').to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


if __name__ == '__main__':
    # 训练网络，并将每个epoch的结果保存下来
    for epoch in range(20):
        train_loss = 0.0
        total = 0.0
        for i, data in tqdm(enumerate(trainloader, 0), desc="iters"):
            if i == 0 and personalize:
                print(data[0].shape)
                print(data[2].shape)
            elif i == 0 and not personalize:
                print(data[0].shape)
                print(data[1].shape)
            inputs, labels = data[0], data[2]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += labels.size(0)

        print('epoch:{}, loss:{}'.format(epoch+1, train_loss/total))
        print('Saving epoch {} model...' .format(epoch+1))
        state = {
            'net': net.state_dict(),
            'epoch': epoch+1
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/enrico_epoch_{}.ckpt' .format(epoch+1))
    print('Finished Training!')
