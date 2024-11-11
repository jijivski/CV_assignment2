import os.path as osp
import cv2
import numpy as np

def gaussian_filter(img, kernel_size, sigma):
    """Returns the image after Gaussian filter.
    Args:
        img: the input image to be Gaussian filtered.
        kernel_size: the kernel size in both the X and Y directions.
        sigma: the standard deviation in both the X and Y directions.
    Returns:
        res_img: the output image after Gaussian filter.
    """
    # Create a copy of the input image to store the filtered result
    res_img = img.copy()
    
    # Calculate the padding size
    pad = kernel_size // 2
    
    # Create padded image with zero padding
    padded_img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    
    # Generate Gaussian kernel
    x, y = np.meshgrid(np.linspace(-pad, pad, kernel_size), 
                       np.linspace(-pad, pad, kernel_size))
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()
    print('gaussian_kernel',gaussian_kernel)

    # Apply Gaussian filter
    for i in range(pad, img.shape[0] + pad):
        for j in range(pad, img.shape[1] + pad):
            for k in range(img.shape[2]):  # For each color channel
                # Extract local neighborhood
                local_region = padded_img[i-pad:i+pad+1, j-pad:j+pad+1, k]
                
                # Apply convolution
                filtered_value = np.sum(local_region * gaussian_kernel)
                
                # Store filtered value
                res_img[i-pad, j-pad, k] = filtered_value
    
    return res_img

if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "Lena-RGB.jpg"))
    kernel_size = 5
    sigma = 1
    res_img = gaussian_filter(img, kernel_size, sigma)
    cv2.imwrite(osp.join(root_dir, f"gaussian_result_k{kernel_size}_σ{sigma}.jpg"), res_img)

    kernel_size = 5
    sigma = 3
    res_img = gaussian_filter(img, kernel_size, sigma)
    cv2.imwrite(osp.join(root_dir, f"gaussian_result_k{kernel_size}_σ{sigma}.jpg"), res_img)

    kernel_size = 7
    sigma = 3
    res_img = gaussian_filter(img, kernel_size, sigma)
    cv2.imwrite(osp.join(root_dir, f"gaussian_result_k{kernel_size}_σ{sigma}.jpg"), res_img)


'''
gaussian_kernel 
[[0.00296902 0.01330621 0.02193823 0.01330621 0.00296902]
 [0.01330621 0.0596343  0.09832033 0.0596343  0.01330621]
 [0.02193823 0.09832033 0.16210282 0.09832033 0.02193823]
 [0.01330621 0.0596343  0.09832033 0.0596343  0.01330621]
 [0.00296902 0.01330621 0.02193823 0.01330621 0.00296902]]

gaussian_kernel 
[[0.0317564  0.03751576 0.03965895 0.03751576 0.0317564 ]
 [0.03751576 0.04431963 0.04685151 0.04431963 0.03751576]
 [0.03965895 0.04685151 0.04952803 0.04685151 0.03965895]
 [0.03751576 0.04431963 0.04685151 0.04431963 0.03751576]
 [0.0317564  0.03751576 0.03965895 0.03751576 0.0317564 ]]

gaussian_kernel 
[[0.01129725 0.01491455 0.01761946 0.01862602 0.01761946 0.01491455
  0.01129725]
 [0.01491455 0.01969008 0.02326108 0.02458993 0.02326108 0.01969008
  0.01491455]
 [0.01761946 0.02326108 0.02747972 0.02904957 0.02747972 0.02326108
  0.01761946]
 [0.01862602 0.02458993 0.02904957 0.03070911 0.02904957 0.02458993
  0.01862602]
 [0.01761946 0.02326108 0.02747972 0.02904957 0.02747972 0.02326108
  0.01761946]
 [0.01491455 0.01969008 0.02326108 0.02458993 0.02326108 0.01969008
  0.01491455]
 [0.01129725 0.01491455 0.01761946 0.01862602 0.01761946 0.01491455
  0.01129725]]
  
'''

'''
cd /root/autodl-tmp/MKSC-20-0237-codes-data/data/amazon/CV_assignment2/task1/
python task1.py
'''
