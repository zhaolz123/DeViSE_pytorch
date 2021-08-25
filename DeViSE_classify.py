import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from DeViSE_alexnet import Devise_Alexnet
from devise_loss import myLoss
from read_label_embedding import get_labels_embedding
import numpy as np


class Devise_Cifar_classify(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = 0
        self.test_dataset = 0
        self.trainloader = 0
        self.testloader = 0
        self.model = 0
        self.criterion = 0
        self.optimizer = 0
        self.batch_size = 32
        self.embeddings = torch.from_numpy(get_labels_embedding('label_embedding.npy')).to(self.device)

    def load_pretrained_dict(self):
        # 加载预训练模型，获取1-7层训练好的参数
        print("加载预训练模型")
        pretrain_model = torch.load('alexnet_model.pkl')
        pretrained_dict = pretrain_model.state_dict()
        model = Devise_Alexnet()
        model_dict = model.state_dict()
        # 将预训练好的参数导入新模型，新模型和预训练模型第8层名字不同
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        for k, v in model.named_parameters():
            if k != 'embedding_layer.0.weight' and k != 'embedding_layer.0.bias':
                v.requires_grad = False
        self.model = model.to(self.device)
        # 创建损失函数类
        self.criterion = myLoss()
        # 将需要训练的第8层导入优化器
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=0.001, momentum=0.9, weight_decay=0.0005)

    def get_and_load_data(self):
        root = 'data'
        transform = transforms.Compose([
            # 需要对原始数据重新调整大小
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        )
        self.train_dataset = torchvision.datasets.CIFAR100(root, train=True, download=False, transform=transform)
        self.test_dataset = torchvision.datasets.CIFAR100(root, train=False, download=False, transform=transform)
        print("加载数据集")
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train_data(self):
        self.load_pretrained_dict()
        self.get_and_load_data()
        print("Starting training")
        for epoch in range(30):
            running_loss = 0.0
            for data in self.trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.data.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('Train{} epoch,Loss:{:.6f}'.format(epoch + 1, running_loss / len(self.train_dataset)))
        torch.save(self.model, 'devise_alexnet_model.pkl')
        print("Training finished")

    def cal_similarity(self, outputs, labels):
        # 计算每个batch中分类前n准确的个数
        count = 0
        # 计算前5
        n = 5
        for output, label in zip(outputs, labels):
            sim_results = []
            label = label.data.item()
            # 计算每条模型输出向量分别与100个文本向量的点乘求和
            for i in self.embeddings:
                sim_result = torch.sum(torch.mul(output, i)).to("cpu")
                sim_results.append(sim_result.data.item())
            # 求出相似度前n大的文本向量索引
            max_n_index = np.array(sim_results).argsort()[-n:][::-1]
            # 判断输入标签是否在索引中
            if label in max_n_index:
                count = count + 1
        return count

    def test_data(self):
        self.get_and_load_data()
        self.criterion = myLoss()
        print("Start test")
        model_net = torch.load('devise_alexnet_model.pkl')
        model_net = model_net.to(self.device)
        model_net.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        for data in self.testloader:
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model_net(inputs)
            loss = self.criterion(outputs, labels)
            eval_loss += loss.data.item()
            eval_acc += self.cal_similarity(outputs, labels)
        print('Test Loss:{:.6f},Max 5 Acc:{:.6f}'.format
              (eval_loss / len(self.test_dataset), eval_acc / len(self.test_dataset)))


if __name__ == '__main__':
    cifar_cl = Devise_Cifar_classify()
    cifar_cl.train_data()
    cifar_cl.test_data()
