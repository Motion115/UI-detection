from model_zoo.vgg import VGG

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import os
from tqdm import tqdm

from enrico_utils.get_data import get_dataloader
(train_loader, val_loader, test_loader), weights = get_dataloader("enrico_corpus")

configurations = {
    'net': 'VGG16',
    'train_batch_size': 8,
    'val_batch_size': 8,
    'test_batch_size': 8,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'weight_decay': 0.0005,
    'is_continue': False,
    'best_model': './checkpoint/enrico_epoch_20.ckpt'
}

# load on gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = VGG(configurations['net']).to(device)
criterion = nn.CrossEntropyLoss()
# use adam optimizer
optimizer = optim.Adam(net.parameters(), lr=configurations['learning_rate'], weight_decay=configurations['weight_decay'])

def configure_trian(is_continue, net, current_best_model):
    if is_continue:
        # load current best model
        checkpoint = torch.load(current_best_model)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        val_acc = checkpoint['val_acc']
        print('Start training from epoch {}...' .format(start_epoch+1))
    else:
        start_epoch = 0
        loss = 10000
        val_acc = 0
        print('Start training from scratch...')
    return start_epoch, loss, val_acc

def early_stopping():
    pass

def train(train_loader):
    train_loss = 0.0
    total_sample = 0.0
    for i, data in tqdm(enumerate(train_loader, 0), desc="iters"):
        inputs, labels = data[0], data[2]
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_sample += labels.size(0)
    loss = train_loss / total_sample
    return loss

def validation(val_loader, current_net):
    val_total, val_correct = 0, 0
    for i, data in tqdm(enumerate(val_loader, 0), desc="validation"):   
        val_image, val_label = data[0], data[2]
        val_image, val_label = val_image.to(device), val_label.to(device)
        output = net(val_image)
        _, predicted = torch.max(output, 1)
        val_total += val_label.size(0)
        val_correct += (predicted == val_label).sum().item()   
    acc = 100 * val_correct / val_total 
    print('Current Acc:', acc, '%')
    return acc

if __name__ == '__main__':
    start_epoch, bench_loss, bench_val_acc = configure_trian(configurations['is_continue'], net, configurations['best_model'])

    for epoch in range(start_epoch, start_epoch + configurations['num_epochs']):
        loss = train(train_loader)
        acc = validation(val_loader = val_loader, current_net = net.state_dict())

        print('epoch:{}, loss:{}'.format(epoch + 1, loss))
        # only store the models that imporve on validation and drop in loss
        if acc > bench_val_acc and loss < bench_loss:
            bench_loss = loss
            bench_val_acc = acc

            print('Saving best model...')
            state = {
                'net': net.state_dict(),
                'epoch': epoch+1,
                'loss': loss,
                'val_acc': acc
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/enrico_epoch_{}.ckpt'.format(epoch+1))

    print('Finished Training!')
