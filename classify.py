import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from alexnet import AlexNet


class Cifar_classify(object):
    def __init__(self):
        self.train_dataset = 0
        self.test_dataset = 0
        self.trainloader = 0
        self.testloader = 0
        self.criterion = 0
        self.optimizer = 0
        self.net = AlexNet().cuda()

    def get_and_load_data(self):
        root = 'data'
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        )
        self.train_dataset = torchvision.datasets.CIFAR100(root, train=True, download=False, transform=transform)
        self.test_dataset = torchvision.datasets.CIFAR100(root, train=False, download=False, transform=transform)
        batch_size = 32
        print("加载数据集")
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def loss_function(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    def train_data(self):
        self.get_and_load_data()
        self.loss_function()
        print("Starting training")
        for epoch in range(30):
            running_loss = 0.0
            running_acc = 0.0
            for data in self.trainloader:
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.net.forward(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.data.item() * labels.size(0)
                _, pred = torch.max(outputs, 1)
                num_correct = (pred == labels).sum()
                running_acc += num_correct.data.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('Train{} epoch,Loss:{:.6f},Acc:{:.6f}'.format
                      (epoch + 1, running_loss / len(self.train_dataset), running_acc / len(self.train_dataset)))
        torch.save(self.net, 'alexnet_model.pkl')
        print("Training finished")

    def test_data(self):
        self.get_and_load_data()
        self.loss_function()
        print("Start test")
        model_net = torch.load('alexnet_model.pkl')
        model_net = model_net.cuda()
        model_net.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        for data in self.testloader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model_net(inputs)
            loss = self.criterion(outputs, labels)
            eval_loss += loss.data.item() * labels.size(0)
            _, pred = torch.max(outputs, 1)
            num_correct = (pred == labels).sum()
            eval_acc += num_correct.data.item()
        print('Test Loss:{:.6f},Acc:{:.6f}'.format
              (eval_loss / len(self.test_dataset), eval_acc / len(self.test_dataset)))


if __name__ == '__main__':
    cifar_cl = Cifar_classify()
    cifar_cl.train_data()
    cifar_cl.test_data()
