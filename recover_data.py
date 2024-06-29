def remove_labels_and_concatenate(input_file, output_file):
    sentences = []
    with open(input_file, 'r', encoding='utf-8') as file:
        sentence = []
        for line in file:
            if line.strip():  # 如果当前行不为空
                word, tag = line.strip().split()  # 拆分单词和标签
                sentence.append(word)  # 只保留单词
            else:  # 如果当前行为空行，表示一个句子结束
                if sentence:  # 确保句子不为空
                    sentences.append(''.join(sentence))  # 将句子中的字连起来，并添加到句子列表中
                    sentence = []  # 重置当前句子列表

    if sentence:  # 如果文件以非空行结束，确保最后一个句子也被处理
        sentences.append(''.join(sentence))

    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(sentence + '\n\n')  # 写入每个句子，并在句子之间添加空行


# 输入和输出文件路径
input_file = '文件路径\\train_data.txt'
output_file = '文件路径\\train_data_origin.txt'

# 调用函数处理数据
remove_labels_and_concatenate(input_file, output_file)
