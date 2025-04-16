import numpy as np

def error_func(im):
    im_fft = np.fft.fft2(im)
    
    percentage = 0.35
    H, W = im.shape
    h_height, h_width = H/2, W/2
    threshold = (h_height**2 + h_width**2)*(percentage**2)
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    mask = ((X**2 + Y**2) <= threshold) | \
           (((W-1 - X)**2 + Y**2) <= threshold) | \
           ((X**2 + (H-1 - Y)**2) <= threshold) | \
           (((W-1 - X)**2 + (H-1 - Y)**2) <= threshold)
    # indices = np.argwhere(mask)

    im_fft_subset = im_fft[mask]

    return np.sum(im_fft_subset.flatten())
