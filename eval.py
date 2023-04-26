from main import *
#from enrico_utils.get_data import get_dataloader
#(trainloader, val_loader, test_loader), weights = get_dataloader("enrico_corpus")

classes = ["Text", "Text Button", "Icon", "Card", "Drawer", "Web View", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab",
                    "Background Image", "Image", "Video", "Input", "Number Stepper", "Checkbox", "Radio Button", "Pager Indicator", "On/Off Switch", "Modal", "Slider", "Advertisement", "Date Picker", "Map View"]

if __name__ == '__main__':

    # 加载效果最好的网络模型
    checkpoint = torch.load('./checkpoint/enrico_epoch_2.ckpt')
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    
    # print(test_loader['image'])
    
    for i, testset in enumerate(test_loader['image'], 0):
        '''
        for j, data in enumerate(testset, 0):
            #print(data)
            test_image = data[0]
            test_label = data[2]
            output = net(test_image)
            _, predicted = torch.max(output, 1)
            print('预测：' + ' '.join('%5s' % classes[predicted[i]] for i in range(1)))
            print('实际：' + ' '.join('%5s' % classes[test_label[i]] for i in range(1)))
        break
        '''

    exit()
    # 查看前十张图片的预测效果
    dataiter = iter(test_loader['image'])
    test_images, test_labels = dataiter.next()
    test_images, test_labels = test_images.to(device), test_labels.to(device)

    outputs = net(test_images[:10])
    _, predicted = torch.max(outputs, 1)
    print('预测：' + ' '.join('%5s' % classes[predicted[i]] for i in range(10)))
    print('实际：' + ' '.join('%5s' % classes[test_labels[i]] for i in range(10)))

    # 测试模型的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))