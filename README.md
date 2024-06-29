## 中文实体命名识别

模型使用ALBERT   https://huggingface.co/ckiplab/albert-base-chinese-ner
在BERT基础上进行了优化和改进的轻量级的语言模型，ALBERT在拥有相近的性能的情况下，显著减少了参数量和训练时间

## 数据

```
北  B-LOC				# MSRA微软亚洲研究院开源数据
京  I-LOC				# BIO标注体系
是  O					# O, B-LOC, I-LOC, B-PER, I-PER,B-ORG, I-ORG
中  B-LOC				# B- 表示一个命名实体的开始部分（Beginning）
国  I-LOC				# I- 表示一个命名实体的中间部分（Inside）
的  O					# O 表示不属于任何命名实体（Outside）
首  O					# 句子之间间隔一行
都  O
```

数据划分在`partition_data.py`中设置划分训练集和测试集的比例

`recover_data.py`可以将数据去除标签恢复成原本的句子

## 训练

`train.py` 

```
# 可更改的参数
max_length 训练使用的最大句长，也是预测时的单次最大句长，参数大小和训练时长正相关
batch_size 一次性并行处理的样本数量，模型会同时计算这些样本的损失值，并更新模型参数
学习率      控制模型参数更新幅度
训练轮数    在整个训练数据集进行反复训练的次数，每轮对整个训练数据完整遍历
```

## 评估

训练完模型文件会保存到指定文件夹中，加载方式和加载预训练的ALBERT模型的方式一致

```
model_path = '文件路径\\save_model'
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = AlbertForTokenClassification.from_pretrained(model_path)
```

运行`predict.py`会对模型进行评估

|              | precision | recall | f1-score | support |
| :----------: | :-------: | :----: | :------: | :-----: |
|    B-LOC     |           |        |          |         |
|    I-LOC     |           |        |          |         |
|    B-PER     |           |        |          |         |
|    I-PER     |           |        |          |         |
|    B-ORG     |           |        |          |         |
|  micro avg   |           |        |          |         |
|  macro avg   |           |        |          |         |
| weighted avg |           |        |          |         |

```
precision（精确率）：	模型预测为正类别的样本中，真正为正类别的样本所占的比例
recall（召回率）：	真正为正类别的样本中，被模型预测为正类别的样本所占的比例
f1-score（F1 分数）： 精确率和召回率的调和平均值，用于综合评估模型的预测性能
support（支持数）：	每个标签在测试集中的真实样本数
```

## 预测

在`predict.py`中设置你要预测的文本

```
北京是中国的首都
```

```
LOC : 北京
LOC : 中国
```
