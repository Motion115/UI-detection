from models.classification.mlp import MLP
from enrico_utils.get_embedding_data import get_learned_embedding_dataset

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import os

import torch
from torch.utils.tensorboard import SummaryWriter

def train(device, train_loader, net, optimizer, criterion):
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

def validation(device, val_loader, net):
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

def test(best_model, test_loader, net, device):
    checkpoint = torch.load(best_model)
    net.load_state_dict(checkpoint['net'])
    
    # testbench = test_loader['image']    
    correct_t1 = 0
    correct_t3 = 0
    correct_t5 = 0
    total = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader, 0), desc='iters'): 
            test_image, test_label = data[0], data[2]
            test_image, test_label = test_image.to(device), test_label.to(device)
            output = net(test_image)
            # output is a 20-dim vector, extract the index of the top 5 values
            _, predicted = torch.topk(output, 5, dim=1)
            # calculate top-1, top-3, top-5 accuracy
            total += test_label.size(0)
            for j in range(test_label.size(0)):
                if test_label[j] in predicted[j]:
                    correct_t5 += 1
                    if test_label[j] in predicted[j][:3]:
                        correct_t3 += 1
                        if test_label[j] == predicted[j][0]:
                            correct_t1 += 1
    
    print('top1-acc: %.3f%%' % (100 * correct_t1 / total))
    print('top3-acc: %.3f%%' % (100 * correct_t3 / total))
    print('top5-acc: %.3f%%' % (100 * correct_t5 / total))

def configure_trian(is_continue, net, current_best_model):
    if is_continue:
        # load current best model
        checkpoint = torch.load(current_best_model)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('Start training from epoch {}...' .format(start_epoch+1))
    else:
        start_epoch = 0
        loss = 10000
        print('Start training from scratch...')
    return start_epoch, loss

def operations(config: dict, net, is_test):
    # load data
    train_loader, val_loader, test_loader = get_learned_embedding_dataset("enrico_corpus", batch_size=config['batch_size'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    # initialize net
    net.to(device)
    
    if config['is_test'] == True:
        print("Testing " + config["net"] + "...")
        test(config['continue_on'], test_loader, net, device)
        return
    
    print("Training " + config["net"] + "...")

    writer = SummaryWriter()

    # use adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=config['weight_decay'])

    start_epoch, bench_loss = configure_trian(config['is_continue'], net, config['continue_on'])
    
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        loss = train(device, train_loader, net, optimizer, criterion)
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

def classify_on_embedding(is_test = False):
    mlp_config = {
        'net': 'classify_on_embedding',
        'batch_size': 64,
        'num_epochs': 300,
        'learning_rate': 0.001,
        'weight_decay': 1e-08,
        'is_continue': False,
        'weights': './weights/classify_on_embedding/',
        'continue_on': './weights/classify_on_embedding/fuse_epoch_1.ckpt',
        'is_test': False
    }
    net = MLP(150, 20)
    if is_test:
        mlp_config['is_test'] = True
    operations(mlp_config, net, is_test)

if __name__ == '__main__':
    classify_on_embedding(is_test = False)
    
    