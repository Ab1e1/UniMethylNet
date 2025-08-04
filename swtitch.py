import random

def generate_comment_line(chromosome, start, end, strand):
    return f">{chromosome}:{start}-{end};{strand}\n"

def process_txt_to_fasta(input_file):
    # 读取输入文件内容
    with open(input_file, 'r') as f:
        txt_content = f.read()

    # 分割文本内容为每行
    lines = txt_content.split('\n')

    # 初始化fasta字符串
    fasta_content = ""

    # 初始化注释行的基本信息
    chromosome = "AZIN2_1"
    start = 33545693
    end = 33546733

    # 遍历每一行
    for line in lines:
        # 如果行为空，则跳过
        if not line.strip():
            continue
        
        # 生成随机的strand信息
        strand = random.choice(["+", "-"])

        # 生成注释行
        comment_line = generate_comment_line(chromosome, start, end, strand)

        # 构造fasta行
        fasta_line = f"{comment_line}{line}\n"

        # 添加fasta行到fasta内容中
        fasta_content += fasta_line

    # 将fasta内容写入到文件中
    with open('datasets/4mC/4mC_Tolypocladium/aft_train_pos.txt', 'w') as f:
        f.write(fasta_content)

# 调用函数并传入输入txt文件路径
process_txt_to_fasta('datasets/4mC/4mC_Tolypocladium/train_pos.txt')
