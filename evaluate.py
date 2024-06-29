import os
from transformers import BertTokenizerFast, AlbertForTokenClassification
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import classification_report

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


# 定义函数读取测试集数据
def read_test_data(file_path):
    sentences = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        sentence = []
        label = []
        for line in file:
            if line.strip():
                word, tag = line.strip().split()
                sentence.append(word)
                label.append(tag)
            else:
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
        if sentence:
            sentences.append(sentence)
            labels.append(label)

    return sentences, labels

# 加载测试集数据
test_sentences, test_labels = read_test_data('文件路径\\test_data.txt')

# 定义评估函数
def evaluate_model(model, tokenizer, test_sentences, test_labels, label_map, device):
    model.eval()  # 设置模型为评估模式
    eval_dataset = NERDataset(test_sentences, test_labels, tokenizer, label_map)  # 创建评估数据集对象
    eval_loader = DataLoader(eval_dataset, batch_size=32)  # 创建评估数据加载器

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            true_labels.extend(batch['labels'].cpu().numpy().flatten())
            predicted_labels.extend(predictions.cpu().numpy().flatten())

    return true_labels, predicted_labels

# 定义标签和标签映射
label_list = ["O", "B-LOC", "I-LOC", "B-PER", "I-PER","B-ORG", "I-ORG"]
label_map = {label: i for i, label in enumerate(label_list)}  # 创建标签到索引的映射字典

# 加载保存的模型
saved_model_path = '文件路径\\save_model'
model = AlbertForTokenClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizerFast.from_pretrained(saved_model_path)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型移动到GPU或CPU上

# 执行评估
true_labels, predicted_labels = evaluate_model(model, tokenizer, test_sentences, test_labels, label_map, device)

# 输出评估结果（示例，根据需要进行具体指标的计算）
print(classification_report(true_labels, predicted_labels, target_names=label_list[1:], labels=range(1, len(label_list))))
