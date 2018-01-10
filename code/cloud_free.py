import numpy as np
import matplotlib
# make figure windows active
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
import time
from PIL import Image
from skimage import filters


def cloud_free(file_glob):
    """
    produces a cloud free image using Otsu's method of thresholding on each
    through a stack of images
    """
    fnames = glob.glob(file_glob)
    # Figure out the image sizes. Assume they all have the same size as the
    # first.
    shape = Image.open(fnames[0]).size
    # number of images for histogram
    N = 75
    # Preallocate numpy arrays for cloud-free image and ground pixel counts.
    cfi = np.zeros(shape)
    tmp = np.zeros(shape)
    gpc = np.ones(shape)
    img = np.zeros((shape[0], shape[1],  N), dtype=int)
    thr = np.zeros(shape)

    # load N images to produce histograms for individual pixels
    for idx in range(0, N-1):
        print(idx, end="\r")
        tmp = np.asarray(Image.open(fnames[idx]), dtype=int)
        img[:, :, idx] = tmp

    # use otsu thresholding on each pixel slice
    for i in range(0, shape[0]):
        print(i, end="\r")
        for j in range(0, shape[0]):
            # otsu thresholding doesn't like single value pixels
            # account for space pixels
            if np.sum(img[i, j, :]) == 0:
                thr[i, j] = 0
            else:
                thr[i, j] = filters.threshold_otsu(img[i, j, :])

    # now define the cfi as in cloud_free.py
    for idx, fname in enumerate(fnames[0:30]):
        print(idx, end="\r")
        tmp = np.asarray(Image.open(fname), dtype=int)
        loc = (tmp <= thr) & (tmp > 0)
        cfi[loc] += tmp[loc]
        gpc[loc] += 1

    # calculate cloud free
    C = cfi//gpc
    # adjust contrast
    # C = C*(255/np.max(C))
    return C


file_glob = 'code/vis8/nine/*.jpg'
C = cloud_free(file_glob)
plt.figure()
plt.imshow(C, cmap='Greys')
plt.show()
