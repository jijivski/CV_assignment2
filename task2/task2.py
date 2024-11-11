import os.path as osp

import cv2
import numpy as np


def histogram_equalization(img):
    """Returns the image after histogram equalization.
    Args:
        img: the input image to be executed for histogram equalization.
    Returns:
        res_img: the output image after histogram equalization.
    """
    # TODO: implement the histogram equalization function.
    # Placeholder that you can delete. An image with all zeros.
    # res_img = np.zeros_like(img)
    # return res_img
    
    if len(img.shape) == 2 or img.shape[2] == 1:
        gray_img = img
    else:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram
    # hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = cv2.calcHist([gray_img], [0], None, [1024], [0, 1024])

    
    # Normalize the histogram
    hist /= hist.sum()
    
    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Normalize the CDF to [0, 255]
    # cdf_normalized = (cdf * 255 / cdf.max()).astype(np.uint8)
    cdf_normalized = (cdf * 255 / cdf.max()).astype(np.float32)
    
    # Use the CDF as a mapping function to transform the original image
    equalized_img = cdf_normalized[gray_img]
    
    # breakpoint()
    
    return equalized_img



def local_histogram_equalization(img, window_size=60):
    """Returns the image after local histogram equalization.
    Args:
        img: the input image to be executed for local histogram equalization.
    Returns:
        res_img: the output image after local histogram equalization.
    """
    # TODO: implement the local histogram equalization function.
    # Placeholder that you can delete. An image with all zeros.
    # res_img = np.zeros_like(img)
    # return res_img

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # breakpoint()
    if len(gray_img.shape) == 2 or gray_img.shape[2] == 1:
        gray_img = gray_img
    else:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    assert len(gray_img.shape) == 2, "The input image must be a grayscale image."   
    # Create an empty image to store the result
    equalized_img = np.empty_like(gray_img)
    
    # Pad the image to handle the borders
    padded_img = np.pad(gray_img, window_size // 2, mode='edge')

    height, width = img.shape[0], img.shape[1]
    # equalized_img = np.zeros_like(img)
    # breakpoint()
    for i in range(0, height, window_size):
        for j in range(0, width, window_size):
            local_region = img[i:i+window_size, j:j+window_size]
            print(histogram_equalization(local_region).shape)
            print(equalized_img[i:i+window_size, j:j+window_size].shape)
            equalized_img[i:i+window_size, j:j+window_size] = histogram_equalization(local_region)
    # Apply histogram equalization to each local region
    # for i in range(gray_img.shape[0]):
    #     for j in range(gray_img.shape[1]):
    #         local_region = padded_img[i:i+kernel_size, j:j+kernel_size]
    #         equalized_img[i, j] = histogram_equalization(local_region)
    
    return equalized_img


if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "Original_HistEqualization.jpeg"))

    res_hist_equalization = histogram_equalization(img)

    cv2.imwrite(osp.join(root_dir, "HistEqualization.jpg"), res_hist_equalization)

    res_local_hist_equalization = local_histogram_equalization(img)
    
    cv2.imwrite(
        osp.join(root_dir, "LocalHistEqualization.jpg"), res_local_hist_equalization
    )
