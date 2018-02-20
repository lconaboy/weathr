import numpy as np
import matplotlib
# make figure windows active
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
from PIL import Image
from skimage import filters
from scipy.signal import medfilt


def threshold(images):
    """Uses Otsu's method to calculate a treshold that splits a bimodal
histogram. images is expected to be an 3D matrix, and the treshold
is calculated along the third axis.

Note: Applies some extra massaging of the input images when Otsu may
fail, i.e. when the input is not bimodal. In the event that the
histogram is not bimodal, the average is used instead. A reasonable
fallback.

    """
    thr = np.zeros((images.shape[0], images.shape[1]))

    # How can this be vectorized? Give me that speed. for loops bad.
    for i in np.arange(0, images.shape[0]):
        for j in np.arange(0, images.shape[1]):
            if np.any(images[i, j, :] != images[i, j, 0]):
                ots = filters.threshold_otsu(images[i, j, :])
                mu = np.mean(images[i, j, :])
                std = np.std(images[i, j, :])

                thr[i, j] = (ots + mu + 3*std)/2 if ots != 0 else mu + 3*std
            else:
                thr[i, j] = images[i, j, 0]

    return thr


def cloud_free(images, threshold):
    """Returns a cloud-free image using clear pixels from images as
determined by the given threshold. Output is an image of same shape as
input images.

    """
    M, N, P = images.shape
    cfi = np.zeros((M, N))
    gpc = np.zeros((M, N))

    for idx in np.arange(0, P):
        print('Searching image {} for clear pixel'.format(idx), end='\r')
        loc = (images[:, :, idx] <= threshold)
        sky = (images[:, :, idx] == 0)
        cfi[loc] += images[:, :, idx][loc]
        gpc[loc] += 1
        gpc[sky] += 1

    C = cfi//gpc

    nans = np.isnan(C)
    if np.any(~nans):
        C_final = C
    else:
        C_smooth = C
        C_smooth[nans] = 0
        C_smooth = medfilt(C_smooth, 5)
        C_final = C
        C_final[nans] = C_smooth[nans]

    return C_final


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

rgn = [slice(2615, 3015), slice(2350, 2750)]  # 400x400
files = glob.glob('bands13/vis6/*.jpg')
shape = Image.open(files[0]).size
images = np.zeros((400, 400, len(files)))
land_mask = np.asarray(Image.open('landmask.gif'), dtype=int)[rgn[0], rgn[1]]
for idx in np.arange(0, len(files)):
    print('Loading file idx = {}'.format(idx), end='\r')
    image = np.asarray(Image.open(files[idx]), dtype=int)
    images[:, :, idx] = image[rgn[0], rgn[1]] * (1 - land_mask)

thr = threshold(images)
vals = cloud_free(images, thr)

plt.figure()
plt.imshow(vals, cmap='Greys_r')
plt.show()




# vals = cloud_free(file_glob, land_mask, 50, 30, rgn)

# # # test data
# # # band = test_band(100, 100, 75, 15, 25)
# # # C = cloud_free_test(band, 75, 25)

# # display the final figure
# plt.figure()
# plt.imshow(vals[0], cmap='Greys_r')
# plt.show()

# # # produce the false colour image
# # rfn = 'code/bands13/falsecol/bands13-nir-fc.jpg'
# # gfn = 'code/bands13/falsecol/bands13-vis8-fc.jpg'
# # bfn = 'code/bands13/falsecol/bands13-vis6-fc.jpg'
# # falsecol = false_colour(rfn, gfn, bfn)
# # plt.figure()
# # plt.imshow(falsecol)
# # plt.show()
