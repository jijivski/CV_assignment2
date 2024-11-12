import os.path as osp
import skimage
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
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # hist = cv2.calcHist([gray_img], [0], None, [1024], [0, 1024])

    
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



def local_histogram_equalization(img, window_size=100):
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

    print('img.shape',img.shape)

    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Create an empty image to store the result
    equalized_img = np.zeros_like(gray_img)

    # Pad the image to handle the borders
    padded_img = np.pad(gray_img, window_size // 2, mode='reflect')

    cv2.imwrite('padded.png',padded_img)
    height, width = gray_img.shape

    for i in range(height):
        for j in range(width):
            # Define the local region
            local_region = padded_img[i:i+window_size, j:j+window_size]
            # Apply histogram equalization to the local region
            local_equalized = histogram_equalization(local_region)
            # Assign the equalized value to the central pixel
            equalized_img[i, j] = local_equalized[window_size//2, window_size//2]

    return equalized_img


if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    # img = cv2.imread(osp.join(root_dir, "Original_HistEqualization.jpeg"))
    img = skimage.data.moon()

    res_hist_equalization = histogram_equalization(img)

    cv2.imwrite(osp.join(root_dir, "HistEqualization.jpg"), res_hist_equalization)

    res_local_hist_equalization = local_histogram_equalization(img)
    
    cv2.imwrite(
        osp.join(root_dir, "LocalHistEqualization.jpg"), res_local_hist_equalization
    )

'''
cd /root/autodl-tmp/MKSC-20-0237-codes-data/data/amazon/CV_assignment2/task2/
python task2.py
'''