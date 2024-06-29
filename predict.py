import math
import torch
from transformers import BertTokenizerFast, AlbertForTokenClassification

# 加载模型和分词器
model_path = '文件路径\\save_model'
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = AlbertForTokenClassification.from_pretrained(model_path)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 标签映射
label_list = ["O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG"]
label_map = {i: label for i, label in enumerate(label_list)}

# 输入句子
sentence = "北京是中国的首都"

# 分段处理超出部分的句子
max_length = 128  # 模型最大处理长度
stride = 10  # 句子划分缩进的长度

segments = []
start_index = 0

while start_index < len(sentence):
    end_index = min(start_index + max_length, len(sentence))
    segment = sentence[start_index:end_index]
    segments.append(segment)
    start_index += max_length - stride

# 打印分段结果示例
# for i, segment in enumerate(segments):
#     print(f"Segment {i + 1}: {segment}")

# 初始化存储结果的列表
token_result = []
label_result = []


# 处理每个分段
for segment in segments:
    encoding = tokenizer(
        segment,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in encoding.items()}

    # 对每个字进行预测
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    predictions = predictions[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    print(tokens)
    print(predictions)

    # 删除[CLS]，查看打印的token列表发现首尾有两个特殊字符，去除首字符让tokens值和predictions值对应输出
    tokens = tokens[1:]

    # 解码预测结果，将数字的标签值值转化为对应的标签值
    for token, label_id in zip(tokens, predictions):
        if token in tokenizer.all_special_tokens:
            continue
        label = label_map.get(label_id, "O")
        if label != "O":
            token_result.append(f"{token}")
            label_result.append(f"{label}")

# 处理预测的字，组成词语，不保存重复出现的词
def output(token_result,label_result):
    # 初始化变量
    current_loc = None
    current_per = None
    current_org = None
    temp = ""
    list = []

    # 遍历结果列表
    for token, label in zip(token_result, label_result):
        if label == "B-LOC":
            if current_loc is not None:
                temp = "LOC : " + current_loc
                if temp not in list:
                    list.append(temp)
                current_loc = None
            if current_per is not None:
                temp = "PER : " + current_per
                if temp not in list:
                    list.append(temp)
                current_per = None
            if current_org is not None:
                temp = "ORG : " + current_org
                if temp not in list:
                    list.append(temp)
                current_org = None
            current_loc = token
        elif label == "I-LOC":
            if current_loc is not None:
                current_loc += token

        if label == "B-PER":
            if current_loc is not None:
                temp = "LOC : " + current_loc
                if temp not in list:
                    list.append(temp)
                current_loc = None
            if current_per is not None:
                temp = "PER : " + current_per
                if temp not in list:
                    list.append(temp)
                current_per = None
            if current_org is not None:
                temp = "ORG : " + current_org
                if temp not in list:
                    list.append(temp)
                current_org = None
            current_per = token
        elif label == "I-PER":
            if current_per is not None:
                current_per += token

        if label == "B-ORG":
            if current_loc is not None:
                temp = "LOC : " + current_loc
                if temp not in list:
                    list.append(temp)
                current_loc = None
            if current_per is not None:
                temp = "PER : " + current_per
                if temp not in list:
                    list.append(temp)
                current_per = None
            if current_org is not None:
                temp = "ORG : " + current_org
                if temp not in list:
                    list.append(temp)
                current_org = None
            current_org = token
        elif label == "I-ORG":
            if current_org is not None:
                current_org += token

    # 检查循环结束后是否还有未输出的地点信息
    if current_loc is not None:
        temp = "LOC : " + current_loc
        if temp not in list:
            list.append(temp)
        current_loc = None
    if current_per is not None:
        temp = "PER : " + current_per
        if temp not in list:
            list.append(temp)
        current_per = None
    if current_org is not None:
        temp = "ORG : " + current_org
        if temp not in list:
            list.append(temp)
        current_org = None

    for li in list:
        print(li)

output(token_result,label_result)
