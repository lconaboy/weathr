import numpy as np
import matplotlib
# make figure windows active
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
from PIL import Image
from skimage import filters
from scipy.signal import medfilt
from util import *
from cloud_free import threshold, cloud_free

def sep_months(fnames):
    months = np.zeros(len(fnames), dtype=int)
    path = path_to_weathr_data(year, band)
    x0 = len(path[:-5]) + 28
    x1 = x0 + 2
    for i in range(0, len(fnames)):
        months[i] = int(fnames[i][x0:x1])

    return months

def images_monthly_masked(fnames, rgn, m=1):
    fnames = np.array(fnames)  # have to convert to array for logic idxing
    months = sep_months(fnames)
    month_fn = fnames[months == m]
    images_masked = load_images_with_region(month_fn, rgn)

    return images_masked


year = '08'
band = 'vis8'
fnames = glob.glob(path_to_weathr_data(year, band))
months = sep_months(fnames)

thr = np.zeros(shape=(slice_d(weathr_regions['capetown'][0]),
                      slice_d(weathr_regions['capetown'][1]),
                      len(set(months))))
vals = np.zeros(shape=(slice_d(weathr_regions['capetown'][0]),
                       slice_d(weathr_regions['capetown'][1]),
                       len(set(months))))

for m in range(1, max(months)+1):
    images_masked = images_monthly_masked(fnames,
                                          weathr_regions['capetown'], m)
    thr[:, :, m-1] = threshold(images_masked)
    vals[:, :, m-1] = cloud_free(images_masked, thr[:, :, m-1])

savename = '_' + year + '_' + band

np.save('thr' + savename, thr)
np.save('vals' + savename, vals)


plt.figure()
plt.imshow(vals[:, :, 2], cmap='Greys_r')
plt.show()
