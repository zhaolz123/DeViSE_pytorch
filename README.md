# DeViSE_pytorch
此代码使用pytorch复现论文DeViSE: A Deep Visual-Semantic Embedding Model，其中数据集采用cifar-100，模型采用Alexnet
参考https://github.com/liangxiaotian/DeViSE_tensorflow
环境：pytorch1.5.0  numpy  gensim
1.将标签生成文本向量
下载预训练好的模型GoogleNews-vectors-negative300.bin
python get_word2vec.py
2.进行分类任务预训练
python classify.py
3.训练DeViSE模型
python DeViSE_alexnet.py
