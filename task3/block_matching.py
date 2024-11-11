import os
import os.path as osp

import numpy as np
from dataset import KITTIDataset
from matplotlib import pyplot as plt


def add_padding(I, padding):
    """
    Adds zero padding to an RGB or grayscale image.

    Args:
        I (np.ndarray): HxWx? numpy array containing RGB or grayscale image

    Returns:
        P (np.ndarray): (H+2*padding)x(W+2*padding)x? numpy array containing zero padded image
    """
    if len(I.shape) == 2:
        H, W = I.shape
        padded = np.zeros((H + 2 * padding, W + 2 * padding), dtype=np.float32)
        padded[padding:-padding, padding:-padding] = I
    else:
        H, W, C = I.shape
        padded = np.zeros((H + 2 * padding, W + 2 * padding, C), dtype=I.dtype)
        padded[padding:-padding, padding:-padding] = I

    return padded


def sad(image_left, image_right, window_size=3, max_disparity=50):
    """
    Compute the sum of absolute differences between image_left and image_right.

    Args:
        image_left (np.ndarray): HxW numpy array containing grayscale right image
        image_right (np.ndarray): HxW numpy array containing grayscale left image
        window_size: window size (default 3)
        max_disparity: maximal disparity to reduce search range (default 50)

    Returns:
        D (np.ndarray): HxW numpy array containing the disparity for each pixel
    """

    D = np.zeros_like(image_left)

    # add zero padding
    padding = window_size // 2
    image_left = add_padding(image_left, padding).astype(np.float32)
    image_right = add_padding(image_right, padding).astype(np.float32)

    height = image_left.shape[0]
    width = image_left.shape[1]

    for x in range(padding, padding + width - window_size + 1):
        for y in range(padding, padding + height - window_size + 1):
            min_sad = float('inf')
            best_d = 0
            window_left = image_left[y - padding:y + padding + window_size, x - padding:x + padding + window_size]
            for d in range(max_disparity + 1):
                window_right = image_right[y - padding:y + padding + window_size, x - padding + d:x + padding + window_size + d]
                sad_value = np.sum(np.abs(window_left - window_right))
                if sad_value < min_sad:
                    min_sad = sad_value
                    best_d = d
            D[y - padding:y + padding + window_size, x - padding:x + padding + window_size] = best_d

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 1)
    # -------------------------------------

    return D


def visualize_disparity(
    disparity, im_left, im_right, out_file_path, title="Disparity Map", max_disparity=50
):
    """
    Generates a visualization for the disparity map.

    Args:
        disparity (np.array): disparity map
        im_left (np.array): left image
        im_right (np.array): right image
        out_file_path: output file path
        title: plot title
        max_disparity: maximum disparity
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(im_left, cmap='gray')
    plt.title('Left Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(im_right, cmap='gray')
    plt.title('Right Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    # Normalize the disparity map to [0, 1] range and apply a colormap
    normalized_disparity = (disparity / max_disparity).astype(np.float32)
    plt.imshow(normalized_disparity, cmap='jet')
    plt.colorbar(label='Disparity')
    plt.title(title)
    plt.axis('off')

    plt.savefig(out_file_path)
    plt.close()

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 2)
    # -------------------------------------


def main():
    # Hyperparameters
    window_size = 3
    max_disparity = 50

    # Shortcuts
    root_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(root_dir, "KITTI_2015_subset")
    out_dir = osp.join(
        root_dir, "output/handcrafted_stereo", f"window_size_{window_size}"
    )
    if not osp.isdir(out_dir):
        os.makedirs(out_dir)

    # Load dataset
    dataset = KITTIDataset(osp.join(data_dir, "testing"))

    # Calculation and Visualization
    for i in range(len(dataset)):
        # Load left and right images
        im_left, im_right = dataset[i]
        im_left, im_right = im_left.squeeze(-1), im_right.squeeze(-1)

        # Calculate disparity
        D = sad(im_left, im_right, window_size=window_size, max_disparity=max_disparity)

        # Define title and output file name for the plot
        title = f"Disparity map for image {i} with block matching (window size {window_size})"
        out_file_path = osp.join(out_dir, f"{i}_w{window_size}.png")

        # Visualize the disparty and save it to a file
        visualize_disparity(
            D,
            im_left,
            im_right,
            out_file_path,
            title=title,
            max_disparity=max_disparity,
        )


if __name__ == "__main__":
    main()