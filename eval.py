# https://www.jianshu.com/p/bebfb6170e00

from main import *

if __name__ == '__main__':
    # load current best model
    checkpoint = torch.load('./checkpoint/enrico_epoch_20.ckpt')
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    
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