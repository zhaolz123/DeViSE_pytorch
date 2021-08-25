import torch.nn as nn
import torch
from read_label_embedding import get_labels_embedding

class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings = torch.from_numpy(get_labels_embedding('label_embedding.npy')).to(self.device)
        self.margin = torch.tensor(0.1, dtype=float).to(self.device)

    def forward(self, outputs, labels):
        # 计算每个batch的大小，不能写死，因为最后一组batch的大小可能与设定不同
        size = outputs.shape[0]
        loss = torch.zeros(size, dtype=float).to(self.device)
        # 计算正确的预测相似度
        true_labels_embedding = torch.index_select(self.embeddings, 0, labels)
        x_mul = torch.mul(outputs, true_labels_embedding)
        predict_true_similarity = torch.sum(x_mul, dim=1)
        # 计算错误随机的预测相似度
        labels_random = torch.randperm(100)[:size].to(self.device)
        false_labels_embedding = torch.index_select(self.embeddings, 0, labels_random)
        y_mul = torch.mul(outputs, false_labels_embedding)
        predict_negative_similarity = torch.sum(y_mul, dim=1)
        # 按照损失函数计算损失
        loss += torch.max((self.margin - predict_true_similarity + predict_negative_similarity),
                          torch.tensor(0.0, dtype=float).to(self.device))
        loss = torch.sum(loss)
        return loss


