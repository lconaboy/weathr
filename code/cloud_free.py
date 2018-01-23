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
    # shape = (360, 870) # uncomment for test
    shape = Image.open(fnames[0]).size  # comment for test
    # number of images for histogram
    N = 100
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
    #    img[:, :, idx] = tmp[940:1300, 1970:2840] # uncomment for test
        img[:, :, idx] = tmp  # comment for test

    # use otsu thresholding on each pixel slice
    for i in range(0, shape[0]):
        print(i, end="\r")
        for j in range(0, shape[1]):
            # otsu thresholding doesn't like single value pixels
            # account for space pixels
            tmp = img[i, j, :]
            if tmp.min() == tmp.max():
                thr[i, j] = tmp.min()
            elif filters.threshold_otsu(tmp) != 0:
                thr[i, j] = filters.threshold_otsu(tmp)
            else:
                thr[i, j] = filters.threshold_mean(tmp)

    # now define the cfi as in cloud_free.py
    for idx, fname in enumerate(fnames[0:60]):
        print(idx, end="\r")
        tmp = img[:, :, idx]
        loc = (tmp <= thr) & (tmp > 0)
        cfi[loc] += tmp[loc]
        gpc[loc] += 1

    # calculate cloud free
    C = cfi//gpc
    # adjust contrast
    # C = C*(255/np.max(C))
    return C


def cloud_free_test(band, N, val):
    """
    produces a cloud free image using Otsu's method of thresholding on each
    through a stack of images
    
    this function is for testing on simulated images produced by test_band
    """
    # Figure out the image sizes. Assume they all have the same size as the
    # first.
    shape = band[:, :, 0].shape
    # Preallocate numpy arrays for cloud-free image and ground pixel counts.
    cfi = np.zeros(shape)
    tmp = np.zeros(shape)
    gpc = np.zeros(shape)
    thr = np.zeros(shape)

    # use otsu thresholding on each pixel slice
    for i in range(0, shape[0]):
        print(i, end="\r")
        for j in range(0, shape[1]):
            # account for no cloud
            if np.all(band[i, j, :] == val):
                thr[i, j] = val
            else:
                thr[i, j] = filters.threshold_otsu(band[i, j, :])

    # now define the cfi as in cloud_free.py
    for idx in range(0, val):
        print(idx, end="\r")
        tmp = band[:, :, idx]
        loc = (tmp <= thr) & (tmp > 0)
        cfi[loc] += tmp[loc]
        gpc[loc] += 1

    # calculate cloud free
    C = cfi//gpc
    # adjust contrast
    # C = C*(255/np.max(C))
    return C


def test_band(lx, ly, lz, n=15, val=25):
    """
    produces an array of size (lx, ly, lz), i.e. lz images, of size
    (lx,ly) stacked depthwise, with n amounts of noise (noise = 255)
    over a background of value = val
    """
    rand_x = np.random.randint(0, lx-1, size=(n, lz))
    rand_y = np.random.randint(0, ly-1, size=(n, lz))

    band = np.ones(shape=(lx, ly, lz))*val

    for z in range(0, lz):
        band[rand_x[:, z], rand_y[:, z], z] = 255

    return band


file_glob = 'code/vis8/*.jpg'
# band = test_band(50, 50, 75, 15, 25)
# C = cloud_free_test(band, 75, 25)
C = cloud_free(file_glob)

plt.figure()
plt.imshow(C, cmap='Greys')  # perhaps should use cmap='Greys_r'?
plt.show()
