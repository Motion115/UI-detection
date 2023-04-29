# https://www.jianshu.com/p/bebfb6170e00

from main import *
#from enrico_utils.get_data import get_dataloader
#(trainloader, val_loader, test_loader), weights = get_dataloader("enrico_corpus")

if __name__ == '__main__':

    # 加载效果最好的网络模型
    checkpoint = torch.load('./checkpoint/enrico_epoch_20.ckpt')
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    
    # print(test_loader['image'])

    testbench = test_loader['image']    
    correct = 0
    total = 0

    with torch.no_grad():
        for i, testset in tqdm(enumerate(testbench, 0), desc='iters'): 
            for j, data in tqdm(enumerate(testset, 0), desc='batch testing'):
                test_image, test_label = data[0], data[2]
                test_image, test_label = test_image.to(device), test_label.to(device)
                output = net(test_image)
                _, predicted = torch.max(output, 1)
                #print(predicted)
                #print(test_label.data)
                #print('预测：' + ' '.join('%5s' % classes[predicted[i]] for i in range(1)))
                #print('实际：' + ' '.join('%5s' % classes[test_label[i]] for i in range(1)))

                total += test_label.size(0)
                correct += (predicted == test_label).sum().item()
            
            print('Current Acc:', 100 * correct / total, '%')
    
    print('Acc: %.3f%%' % (100 * correct / total))
    exit()