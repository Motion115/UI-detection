from models.cv.vgg import VGG, VGGSmall
from models.cv.vit import ViT

import torch
import torch.optim as optim
import torch.nn as nn
import os

from utils import *

import torch
from torch.utils.tensorboard import SummaryWriter
from enrico_utils.get_data import get_dataloader


def operations(config: dict, net):
    writer = SummaryWriter()
    # load on gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    (train_loader, val_loader, test_loader) = get_dataloader("enrico_corpus", 
        img_dim_x=config['img_dim'], img_dim_y=config['img_dim'], 
        batch_size=config['batch_size'])
    # initialize net
    net.to(device)
    
    if config['is_test'] == True:
        print("Testing " + config["net"] + "...")
        test(config['continue_on'], test_loader, net, device)
        return
    
    print("Training " + config["net"] + "...")

    # use adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=config['weight_decay'])
    start_epoch, bench_loss, bench_val_acc = configure_trian(config['is_continue'], net, config['continue_on'])
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        loss = train(device, train_loader, net, optimizer, criterion)
        #loss = train(device, val_loader, net, optimizer, criterion)
        acc = validation(device, val_loader, net)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Acc/validation", acc, epoch)
        writer.flush()

        print('epoch:{}, loss:{}'.format(epoch + 1, loss * 100))
        print("------------------------------------")
        # only store the models that imporve on validation and drop in loss
        if (acc > bench_val_acc and loss < bench_loss) or epoch % 10 == 0 :
            bench_loss = loss
            bench_val_acc = acc

            print('Saving model...')
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

def operations_nosplit(config: dict, net):
    writer = SummaryWriter()
    # load on gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    (train_loader, val_loader, test_loader, all_loader) = get_dataloader("enrico_corpus", 
        img_dim_x=config['img_dim'], img_dim_y=config['img_dim'], 
        batch_size=config['batch_size'])
    # initialize net
    net.to(device)
    
    if config['is_test'] == True:
        print("Testing " + config["net"] + "...")
        test(config['continue_on'], test_loader, net, device)
        return
    
    print("Training " + config["net"] + "...")

    # use adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=config['weight_decay'])
    start_epoch, bench_loss, bench_val_acc = configure_trian(config['is_continue'], net, config['continue_on'])
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        loss = train(device, all_loader, net, optimizer, criterion)
        acc = validation(device, val_loader, net)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Acc/validation", acc, epoch)
        writer.flush()

        print('epoch:{}, loss:{}'.format(epoch + 1, loss * 100))
        print("------------------------------------")
        # only store the models that imporve on validation and drop in loss
        if (acc > bench_val_acc and loss < bench_loss) or epoch % 10 == 0 :
            bench_loss = loss
            bench_val_acc = acc

            print('Saving model...')
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


def vgg_operations(is_test=False, is_nosplit=True):
    vgg_config = {
        'net': 'VGG16',
        'batch_size': 64,
        'num_epochs': 300,
        'img_dim': 256,
        'learning_rate': 0.001,
        'weight_decay': 1e-08,
        'is_continue': False,
        'weights': './weights/vgg/',
        'continue_on': './weights/vgg/enrico_epoch_101.ckpt',
        'is_test': False
    }
    if is_test:
        vgg_config['is_test'] = True
    net = VGG(num_classes=20)
    if is_nosplit:
        operations_nosplit(vgg_config, net)
    else:
        operations(vgg_config, net)

def vgg_small_operations(is_test=False):
    vgg_config1 = {
        'net': 'VGG8',
        'batch_size': 64,
        'num_epochs': 300,
        'img_dim': 256,
        'learning_rate': 0.001,
        'weight_decay': 1e-08,
        'is_continue': False,
        'weights': './weights/vggsmall/',
        'continue_on': './weights/vggsmall/enrico_epoch_118.ckpt',
        'is_test': False
    }
    if is_test:
        vgg_config1['is_test'] = True
    net = VGGSmall(num_classes=20)
    operations(vgg_config1, net)

def vit_operations(is_test=False):
    vit_config = {
        'net': 'ViT',
        'batch_size': 64,
        'num_epochs': 150,
        'img_dim': 224,
        'learning_rate': 0.001,
        'weight_decay': 1e-08,
        'is_continue': False,
        'weights': './weights/vit/',
        'continue_on': './weights/vit/enrico_epoch_123.ckpt',
        'is_test': False
    }
    if is_test:
        vit_config['is_test'] = True
    net = ViT(num_classes=20)
    operations(vit_config, net)

if __name__ == '__main__':
    i = input("net, train/test > ")
    i = i.split(' ')
    if i[0] == 'vgg':
        if i[1] == "train":
            vgg_operations()
        elif i[1] == "test":
            vgg_operations(is_test=True)
    elif i[0] == 'vggsmall':
        if i[1] == "train":
            vgg_small_operations()
        elif i[1] == "test":
            vgg_small_operations(is_test=True)
    else:
        if i[1] == "train":
            vit_operations()
        elif i[1] == "test":
            vit_operations(is_test=True)
    
    
    