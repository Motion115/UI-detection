from models.cv.vgg import VGG
from models.cv.vit import ViT

import torch
import torch.optim as optim
import torch.nn as nn
import os

from utils import *

import torch
from torch.utils.tensorboard import SummaryWriter
from enrico_utils.get_data import get_dataloader

def operations(config: dict, net, is_test: bool = False):
    writer = SummaryWriter()
    # load on gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    (train_loader, val_loader, test_loader), weights = get_dataloader("enrico_corpus", 
        img_dim_x=config['img_dim'], img_dim_y=config['img_dim'], 
        batch_size=config['batch_size'])
    # initialize net
    net.to(device)
    
    if is_test:
        print("Testing " + config["net"] + "...")
        test(config['continue_on'], test_loader, net, device)
        return
    
    print("Training " + config["net"] + "...")

    # use adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    start_epoch, bench_loss, bench_val_acc = configure_trian(config['is_continue'], net, config['continue_on'])
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        loss = train(device, train_loader, net, optimizer, criterion)
        #loss = train(device, val_loader, net, optimizer, criterion)
        acc = validation(device, val_loader, net)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Acc/validation", acc, epoch)

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

            if not os.path.isdir(config['weights']):
                os.mkdir(config['weights'])
            torch.save(state, config['weights'] + 'enrico_epoch_{}.ckpt'.format(epoch+1))

    print('Finished Training!')
    writer.flush()

def vgg_operations(is_test: bool = False):
    vgg_config = {
        'net': 'VGG16',
        'batch_size': 8,
        'num_epochs': 2,
        'img_dim': 256,
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'is_continue': True,
        'weights': './weights/vgg/',
        'continue_on': './weights/vgg/enrico_epoch_1.ckpt'
    }
    net = VGG(num_classes=20)
    operations(vgg_config, net, is_test=is_test)

def vit_operations(is_test: bool = False):
    print("Training ViT...")
    vit_config = {
        'net': 'ViT',
        'batch_size': 4,
        'num_epochs': 2,
        'img_dim': 224,
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'is_continue': False,
        'weights': './weights/vit/',
        'continue_on': './weights/vit/enrico_epoch_1.ckpt'
    }
    net = ViT(num_classes=20)
    operations(vit_config, net, is_test=is_test)

if __name__ == '__main__':
    vgg_operations(is_test = True)
    
    
    