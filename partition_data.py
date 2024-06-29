import random

def read_data(file_path):
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

# 读取原始数据
file_path = '数据路径'
sentences, labels = read_data(file_path)

# 随机划分数据集
random.seed(42)  # 设置随机种子，以确保每次运行结果一致
data = list(zip(sentences, labels))
random.shuffle(data)

train_ratio = 0.8  # 训练集比例
train_size = int(len(data) * train_ratio)

train_data = data[:train_size]
test_data = data[train_size:]

# 写入训练集文件
train_file = '指定路径\\train_data.txt'
with open(train_file, 'w', encoding='utf-8') as train_f:
    for sentence, label in train_data:
        for word, tag in zip(sentence, label):
            train_f.write(f"{word} {tag}\n")
        train_f.write("\n")

# 写入测试集文件
test_file = '指定路径\\test_data.txt'
with open(test_file, 'w', encoding='utf-8') as test_f:
    for sentence, label in test_data:
        for word, tag in zip(sentence, label):
            test_f.write(f"{word} {tag}\n")
        test_f.write("\n")