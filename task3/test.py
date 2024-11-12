import os
import os.path as osp

import numpy as np
import torch
from block_matching import add_padding, visualize_disparity
from dataset import KITTIDataset
from siamese_neural_network import StereoMatchingNetwork


def compute_disparity_CNN(infer_similarity_metric, img_l, img_r, max_disparity=50):
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
    # Get image dimensions
    height, width = img_l.shape[0], img_l.shape[1]
    
    # Initialize disparity map
    D = torch.zeros((height, width))
    
    # Iterate through each pixel
    for y in range(height):
        for x in range(width):
            best_similarity = float('-inf')
            best_disparity = 0
            
            # Search for best matching disparity
            for d in range(max_disparity + 1):
                # Ensure we don't go out of image bounds
                if x - d < 0:
                    continue
                
                # Extract patches from left and right images
                left_patch = img_l[y:y+1, x:x+1]
                right_patch = img_r[y:y+1, x-d:x-d+1]
                
                # Compute similarity using the trained network
                with torch.no_grad():
                    similarity = calculate_similarity_score(infer_similarity_metric, left_patch, right_patch)
                
                # Update best disparity
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_disparity = d
            
            # Store best disparity
            D[y, x] = best_disparity
    
    return D

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 8)
    # -------------------------------------


def main():
    # Hyperparameters
    training_iterations = 250
    batch_size = 128
    learning_rate = 3e-4
    patch_size = 9
    padding = patch_size // 2
    max_disparity = 50

    # Shortcuts for directories
    root_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(root_dir, "KITTI_2015_subset")
    out_dir = osp.join(root_dir, "output/siamese_network")
    model_path = osp.join(out_dir, f"trained_model_{training_iterations}_final.pth")
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
        file_name = f"{i}_training_iterations_{training_iterations}.png"
        out_file_path = osp.join(out_dir, file_name)
        visualize_disparity(
            disparity_map.squeeze(),
            img_left.squeeze(),
            img_right.squeeze(),
            out_file_path,
            title,
            max_disparity=max_disparity,
        )


if __name__ == "__main__":
    main()
'''
cd /root/autodl-tmp/MKSC-20-0237-codes-data/data/amazon/CV_assignment2/task3/
CUDA_VISIBLE_DEVICES="" python test.py

python train.py
'''