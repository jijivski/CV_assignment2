import torch
import tqdm

# 检查CUDA（GPU支持）是否可用
if torch.cuda.is_available():
    print("CUDA is available! Training on GPU.")
    # 设置设备为第一个可见的CUDA设备
    device = torch.device("cuda:0")
else:
    print("CUDA is not available. Training on CPU.")
    device = torch.device("cpu")

# 定义矩阵大小
matrix_size = 1000

for i in tqdm.tqdm(range(10000)):
    # 在指定设备上创建随机矩阵
    matrix_a = torch.randn(matrix_size, matrix_size, device=device)
    matrix_b = torch.randn(matrix_size, 2*matrix_size, device=device)

    # 在GPU上执行矩阵乘法
    matrix_c = torch.mm(matrix_a, matrix_b)
# 输出结果的尺寸以确认运算完成
print("Resulting matrix size: ", matrix_c.size())

'''
python /root/autodl-tmp/MKSC-20-0237-codes-data/data/amazon/CV_assignment2/utils/gpu_avail.py
'''