import os
import os.path as osp
import tqdm
import numpy as np
import torch
from block_matching import add_padding, visualize_disparity
from dataset import KITTIDataset
from siamese_neural_network import StereoMatchingNetwork,calculate_similarity_score
import cv2
import scipy
import torch.nn.functional as F

def compute_disparity_CNN(infer_similarity_metric, img_l, img_r, max_disparity=50, window_size=9):
    """
    Computes the disparity of the stereo image pair.

    Args:
        infer_similarity_metric:  pytorch module object
        img_l: tensor holding the left image
        img_r: tensor holding the right image
        max_disparity (int): maximum disparity

    Returns:
        D: tensor holding the disparity
    """
    height, width, _ = img_l.shape
    padding = window_size // 2
    height -= 2*padding
    width -= 2*padding
    # D = np.zeros_like(img_l)
    D = np.zeros((height, width))
    
    # img_l = add_padding(img_l, padding)
    # img_r = add_padding(img_r, padding)
    
    for y in tqdm.tqdm(range(2*padding, height + padding, padding)):
        for x in tqdm.tqdm(range(2*padding, width + padding, padding)):
            max_similarity = -float('inf')
            best_d = 0
            
            # 左图窗口
            # should have [128, 1, 9, 9]
            # but have 9,9,1
            # breakpoint()
            window_left = img_l[y-padding:y+padding+1, x-padding:x+padding+1]
            # Step 1: Permute to change the shape to [1, 9, 9]
            window_left = window_left.permute(2, 0, 1)  # This changes [9, 9, 1] to [1, 9, 9]
            # Step 2: Unsqueeze to add batch dimension [1, 1, 9, 9]
            window_left = window_left.unsqueeze(0)  # This adds a batch dimension, resulting in [1, 1, 9, 9]

            # 在右图中向左搜索匹配窗口
            for d in range(max_disparity + 1):
                # 确保搜索区域在右图范围内
                if x - padding - d < 0:
                    break
                
                window_right = img_r[y-padding:y+padding+1, x-padding-d:x+padding+1-d]
                # Step 1: Permute to change the shape to [1, 9, 9]
                window_right = window_right.permute(2, 0, 1)  # This changes [9, 9, 1] to [1, 9, 9]
                # Step 2: Unsqueeze to add batch dimension [1, 1, 9, 9]
                window_right = window_right.unsqueeze(0)  # This adds a batch dimension, resulting in [1, 1, 9, 9]

                # sad_value = np.sum(np.abs(window_left - window_right))
                similarity_value = calculate_similarity_score(infer_similarity_metric,window_left,window_right)
                
                if similarity_value > max_similarity:
                    max_similarity = similarity_value
                    best_d = d
            
            for _x in range(padding):
                for _y in range(padding):
                    D[y-padding-_y, x-padding-_x] = best_d

    return D

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 8)
    # -------------------------------------
def median_filter(disparity_map, kernel_size=3):
    """
    对视差图进行中值滤波
    
    Args:
        disparity_map: 原始视差图
        kernel_size: 滤波窗口大小
    
    Returns:
        滤波后的视差图
    """
    # 如果是PyTorch张量
    if torch.is_tensor(disparity_map):
        # 使用torch的中值滤波
        padded = F.pad(disparity_map.unsqueeze(0).unsqueeze(0), 
                       pad=(kernel_size//2, kernel_size//2, 
                            kernel_size//2, kernel_size//2), 
                       mode='reflect')
        
        # 展开滑动窗口
        patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        median_vals = patches.median(dim=(-1, -2))
        
        return median_vals[0].squeeze()
    
    # 如果是NumPy数组
    else:
        return scipy.ndimage.median_filter(disparity_map, size=kernel_size)

def bilateral_filter(disparity_map, spatial_sigma=3, intensity_sigma=0.1):
    """
    对视差图进行双边滤波
    
    Args:
        disparity_map: 原始视差图
        spatial_sigma: 空间高斯权重标准差
        intensity_sigma: 灰度值相似性权重标准差
    
    Returns:
        滤波后的视差图
    """
    # 如果是PyTorch张量，转为NumPy处理
    if torch.is_tensor(disparity_map):
        disparity_map = disparity_map.numpy()
    
    # OpenCV实现双边滤波
    filtered = cv2.bilateralFilter(
        disparity_map.astype(np.float32), 
        d=spatial_sigma*2+1,  # 滤波直径
        sigmaColor=intensity_sigma*255,  # 颜色空间标准差 
        sigmaSpace=spatial_sigma  # 坐标空间标准差
    )
    
    return filtered


def main(filter='median_filter'):
    # Hyperparameters
    training_iterations = 750 
    batch_size = 128
    learning_rate = 3e-4
    patch_size = 9
    padding = patch_size // 2
    max_disparity = 50

    # Shortcuts for directories
    root_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(root_dir, "KITTI_2015_subset")
    out_dir = osp.join(root_dir, "output/siamese_network")
    # breakpoint()
    model_path = osp.join(out_dir, f"trained_model_{training_iterations}.pth")
    print(f'loading {model_path}')
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # Set network to eval mode
    infer_similarity_metric = StereoMatchingNetwork()
    infer_similarity_metric.load_state_dict(torch.load(model_path))
    infer_similarity_metric.eval()
    infer_similarity_metric.to("cpu")

    # Load KITTI test split
    dataset = KITTIDataset(osp.join(data_dir, "testing"))
    # Loop over test images
    for i in range(len(dataset)):
        print(f"Processing {i} image")
        # Load images and add padding
        img_left, img_right = dataset[i]
        img_left_padded, img_right_padded = add_padding(img_left, padding), add_padding(
            img_right, padding
        )
        img_left_padded, img_right_padded = torch.Tensor(img_left_padded), torch.Tensor(
            img_right_padded
        )

        disparity_map = compute_disparity_CNN(
            infer_similarity_metric,
            img_left_padded,
            img_right_padded,
            max_disparity=max_disparity,
        )
        # Visulization
        title = (
            f"Disparity map for image {i} with SNN (training iterations {training_iterations}, "
            f"batch size {batch_size}, patch_size {patch_size})"
        )
        file_name = f"{i}_training_iterations_{training_iterations}_block_{filter}.png"
        out_file_path = osp.join(out_dir, file_name)
        
        
        if filter=='median_filter':
            disparity_map = median_filter(disparity_map, kernel_size=3)
        if filter=='bilateral_filter':
            disparity_map = bilateral_filter(disparity_map, spatial_sigma=3, intensity_sigma=0.1)
        
        
        visualize_disparity(
            disparity_map.squeeze(),
            img_left.squeeze(),
            img_right.squeeze(),
            out_file_path,
            title,
            max_disparity=max_disparity,
        )
        break


if __name__ == "__main__":
    main(filter='median_filter')
    main(filter='bilateral_filter')
'''
cd /root/autodl-tmp/MKSC-20-0237-codes-data/data/amazon/CV_assignment2/task3/
cd /home/chenghao/workspace/CV_assignment2/task3/
CUDA_VISIBLE_DEVICES="" python test.py

python train.py
'''