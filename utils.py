import torch
from torchvision.models._utils import IntermediateLayerGetter
from tqdm import tqdm

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

'''
        out = net.get_embedding(val_image)
        print(out.shape)
'''

'''
# aquire the embedding of the image by concating a submodule      
t = torch.nn.Sequential(*(list(net.features)+list(net.embedding)))
output = t(val_image)
print(output)
print(output.shape)
'''

# use get_embedding function from the network instance to get the desired embedding
# all the embeddings are with the dimension of 768


def test(best_model, test_loader, net, device):
    checkpoint = torch.load(best_model)
    net.load_state_dict(checkpoint['net'])
    
    # testbench = test_loader['image']    
    correct_t1 = 0
    correct_t3 = 0
    correct_t5 = 0
    total = 0

    '''
    with torch.no_grad():
        for i, testset in tqdm(enumerate(testbench, 0), desc='iters'): 
            for j, data in tqdm(enumerate(testset, 0), desc='batch testing'):
                test_image, test_label = data[0], data[2]
                test_image, test_label = test_image.to(device), test_label.to(device)
                output = net(test_image)
                _, predicted = torch.max(output, 1)
                total += test_label.size(0)
                correct += (predicted == test_label).sum().item()
            print('Current Acc:', 100 * correct / total, '%')
    '''

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