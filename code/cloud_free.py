import numpy as np
import matplotlib
# make figure windows active
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
from PIL import Image
from skimage import filters
from scipy.signal import medfilt


def cloud_free(file_glob, land_mask, N=50, days=30, rgn=[0, 3712, 0, 3712]):
    """
    produces a cloud free image using Otsu's method of thresholding on each
    through a stack of images

    file_glob = path to images
    land_mask = path to land mask
    N         = number of images in histogram
    days      = number of days in image
    rgn       = region of image to focus on [y0, y1, x0, x1]
    """
    fnames = glob.glob(file_glob)
    # Figure out the image sizes. Assume they all have the same size as the
    # first.
#    shape = Image.open(fnames[0]).size
    shape = (rgn[1]-rgn[0], rgn[3]-rgn[2])

    # Preallocate numpy arrays for cloud-free image and ground pixel counts.
    cfi = np.zeros(shape)
    tmp = np.zeros(shape)
    gpc = np.zeros(shape)
    img = np.zeros((shape[0], shape[1],  N), dtype=int)
    thr = np.zeros(shape)

    # load the land mask
    lm = np.asarray(Image.open(land_mask), dtype=int)
    lm = lm[rgn[0]:rgn[1], rgn[2]:rgn[3]]  # reduce the size of the land mask
    lp = np.argwhere(lm != 1)  # find land pixels

    # load N images to produce histograms for individual pixels
    for idx in range(0, N-1):
        print(idx, end="\r")
        tmp = np.asarray(Image.open(fnames[idx]), dtype=int)
        tmp = tmp[rgn[0]:rgn[1], rgn[2]:rgn[3]]  # reduce size
        tmp[lm != 0] = 0  # use only land pixels
        img[:, :, idx] = tmp

    for i in range(0, shape[0]):
        print(i, end="\r")
        for j in range(0, shape[1]):
            # otsu thresholding doesn't like single value pixels
            # account for space pixels
            tmp = img[i, j, :]
            if tmp.min() == tmp.max():
                thr[i, j] = tmp.max()
            else:
                ots = filters.threshold_otsu(tmp)
                mu = np.mean(tmp)
                sigma = np.std(tmp)
                if ots != 0:
                    # the otsu thresholding alone doesn't work too well for nir
                    # so take average of otsu and mean
                    thr[i, j] = (ots + mu + 3*sigma)/2
                else:
                    thr[i, j] = mu + 3*sigma

    # now define the cfi as in cloud_free.py
    for idx in range(0, days):
        print(idx, end="\r")
        tmp = img[:, :, idx]
        loc = (tmp <= thr)
        sky = (tmp == 0)  # this method for the sky breaks it
        cfi[loc] += tmp[loc]
        gpc[loc] += 1
        # incrementing the sky pixels doesn't matter because their value
        # is zero anyway (i.e. 0/n = 0)
        gpc[sky] += 1

    # calculate cloud free
    C = cfi//gpc
    # adjust contrast
    # C = C*(255/np.max(C))
    # there are a few occasions where the algorithm has failed in small patches
    # to account for this we can smooth over those patches

    # find the nan
    nans = np.isnan(C)
    # if there are none skip the smoothing
    if np.any(~nans):
        C_final = C
    else:
        # zero the empty values
        C_smooth = C
        C_smooth[nans] = 0
        # smooth
        print('smoothing')
        C_smooth = medfilt(C_smooth, 5)
        # replace the empty values with smoothed values
        C_final = C
        print('replacing')
        C_final[nans] = C_smooth[nans]

    return (C_final, thr)


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


def false_colour(rfn, gfn, bfn):
    """
    rfn is the filename of the 'red' image
    gfn, bfn ...
    """
    r = np.asarray(Image.open(rfn), dtype=int)
    g = np.asarray(Image.open(gfn), dtype=int)
    b = np.asarray(Image.open(bfn), dtype=int)

    fcol = np.zeros(shape=(3712, 3712, 3))

    fcol[:, :, 0] = np.mean(r[:, :, 0:3], axis=2)/255
    fcol[:, :, 1] = np.mean(g[:, :, 0:3], axis=2)/255
    fcol[:, :, 2] = np.mean(b[:, :, 0:3], axis=2)/255

    return fcol


file_glob = 'code/bands13/vis6/*.jpg'
land_mask = 'code/landmask.gif'
# # data reduction sizes
# rgn = [2715, 3015, 2400, 2700]  # 300x300
rgn = [2615, 3015, 2350, 2750]  # 400x400
vals = cloud_free(file_glob, land_mask, 50, 30, rgn)

# # test data
# # band = test_band(100, 100, 75, 15, 25)
# # C = cloud_free_test(band, 75, 25)

# display the final figure
plt.figure()
plt.imshow(vals[0], cmap='Greys_r')
plt.show()

# # produce the false colour image
# rfn = 'code/bands13/falsecol/bands13-nir-fc.jpg'
# gfn = 'code/bands13/falsecol/bands13-vis8-fc.jpg'
# bfn = 'code/bands13/falsecol/bands13-vis6-fc.jpg'
# falsecol = false_colour(rfn, gfn, bfn)
# plt.figure()
# plt.imshow(falsecol)
# plt.show()
