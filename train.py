import os
from transformers import BertTokenizerFast, AlbertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import time

# 读取数据
def read_data(file_path):
    # 初始化空列表用于存储句子和标签
    sentences = []
    labels = []
    # 初始化空列表用于存储当前正在构建的句子和标签
    sentence = []
    label = []
    # 打开文件进行读取，使用utf-8编码
    with open(file_path, 'r', encoding='utf-8') as file:
        # 逐行读取文件内容
        for line in file:
            if line.strip():  # 如果当前行不为空（去除首尾空格后）
                # 拆分当前行的单词和标签，假定每行以单词和标签的形式分隔
                word, tag = line.strip().split()
                # 将单词和标签分别添加到当前句子和标签列表中
                sentence.append(word)
                label.append(tag)
            else:  # 如果当前行为空行（表示句子结束）
                if sentence:  # 如果当前句子列表不为空
                    # 将当前句子和标签列表添加到总列表中
                    sentences.append(sentence)
                    labels.append(label)
                    # 重置当前句子和标签列表为新的空列表
                    sentence = []
                    label = []

    # 如果文件以非空行结尾，最后一个句子和标签仍需添加到总列表中
    if sentence:
        sentences.append(sentence)
        labels.append(label)
    # 返回解析后的句子和标签列表
    return sentences, labels

train_sentences, train_labels = read_data('文件路径\\train_data.txt')

# 定义NER数据集
class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, label_map, max_length=128):
        self.sentences = sentences  # 句子列表
        self.labels = labels  # 标签列表
        self.tokenizer = tokenizer  # 分词器
        self.label_map = label_map  # 标签映射，将标签转换为模型可识别的索引
        self.max_length = max_length  # 句子的最大长度

    def __len__(self):
        return len(self.sentences)  # 返回数据集中样本的数量

    def __getitem__(self, idx):
        sentence = self.sentences[idx]  # 获取索引为idx的句子
        labels = self.labels[idx]  # 获取索引为idx的标签

        # 使用分词器对句子进行编码
        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,  # 输入已经是单词列表形式
            return_offsets_mapping=True,  # 返回单词在原始句子中的偏移映射
            padding='max_length',  # 对输入进行填充到最大长度
            truncation=True,  # 截断输入句子以符合最大长度
            max_length=self.max_length,  # 句子的最大长度
            return_tensors="pt"  # 返回PyTorch张量
        )

        offset_mapping = encoding.pop("offset_mapping")  # 移除偏移映射，因为它不是模型输入的一部分

        # 将标签映射为模型可识别的索引，未知标签使用 "O" 对应的索引
        labels_enc = [self.label_map.get(label, self.label_map["O"]) for label in labels]

        # 填充标签列表到最大长度
        labels_enc += [self.label_map["O"]] * (self.max_length - len(labels_enc))

        # 将编码结果和标签转换为PyTorch张量，并使用长整型存储标签
        item = {key: val.squeeze() for key, val in encoding.items()}  # 压缩编码结果的维度
        item['labels'] = torch.tensor(labels_enc, dtype=torch.long)  # 转换标签为PyTorch张量

        return item

# 标签和标签映射
label_list = ["O", "B-LOC", "I-LOC", "B-PER", "I-PER","B-ORG", "I-ORG"]
label_map = {label: i for i, label in enumerate(label_list)}  # 创建标签到索引的映射字典

# 加载 tokenizer 和数据集
tokenizer = BertTokenizerFast.from_pretrained('下载的albert模型路径\\albert-base-chinese-ner')  # 加载预训练的分词器
train_dataset = NERDataset(train_sentences, train_labels, tokenizer, label_map)  # 创建训练数据集对象

# 输出数据集的样本数量
print(f"Total samples in dataset: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 创建训练数据加载器，每批次大小为32，打乱顺序

# 加载模型
model = AlbertForTokenClassification.from_pretrained(
    '下载的albert模型路径\\albert-base-chinese-ner',  # 指定预训练模型的路径
    num_labels=len(label_list),  # 类别数量，即标签的数量
    ignore_mismatched_sizes=True  # 忽略不匹配的大小，用于加载预训练模型时忽略与当前模型结构不匹配的权重
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU或CPU

model.to(device)  # 将模型移动到指定设备

optimizer = AdamW(model.parameters(), lr=2e-5)  # 使用AdamW优化器，学习率
total_steps = len(train_loader) * 1  # 总训练步数，假设进行3个epoch的训练

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)  # 设置学习率调度器

# 训练循环
model.train()  # 将模型设置为训练模式
for epoch in range(3):  # 进行3个epoch的训练
    epoch_start_time = time.time()  # 记录当前epoch开始时间
    tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}")  # 使用tqdm显示进度条，描述当前epoch数

    for batch in tqdm_loader:  # 迭代每个batch
        batch = {k: v.to(device) for k, v in batch.items()}  # 将batch数据移动到GPU或CPU上

        outputs = model(**batch)  # 前向传播计算输出
        loss = outputs.loss  # 获取损失值
        loss.backward()  # 反向传播计算梯度

        optimizer.step()  # 更新模型参数
        scheduler.step()  # 更新学习率
        optimizer.zero_grad()  # 梯度清零，准备处理下一个batch

        tqdm_loader.set_postfix({'loss': loss.item()})  # 在进度条上显示当前batch的损失值

    epoch_time_taken = time.time() - epoch_start_time  # 计算当前epoch训练时间
    tqdm.write(f"Epoch {epoch + 1}, Loss: {loss.item()}, Time: {epoch_time_taken:.2f} seconds")  # 打印当前epoch的损失值和训练时间

# 保存模型和tokenizer
model.save_pretrained('文件路径\\save_model')  # 保存模型到指定路径
tokenizer.save_pretrained('文件路径\\save_model')  # 保存tokenizer到指定路径
