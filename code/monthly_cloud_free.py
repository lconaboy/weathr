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


def images_monthly_masked(fnames, dnames, y, m, rgn):
    fnames = np.array(fnames)  # array for logical idxing
    idxs = sep_months(dnames, y, m)
    images_masked = load_images_with_region(fnames[idxs], rgn)

    return images_masked


y = 2014  # year of data to look at
bands = ('vis6', 'vis8', 'nir')  # band to look at

for band in bands:
    paths = path_to_weathr_data(band)  # paths for this band
    mnths = np.arange(1,13)

    dnames = parse_data_datetime(paths[0], band)  # datetime parsed filenames
    fnames = glob.glob(paths[1])  # filenames

    thr = np.zeros(shape=(slice_d(weathr_regions['capetown'][0]),
                          slice_d(weathr_regions['capetown'][1]),
                          len(mnths)))
    vals = np.zeros(shape=(slice_d(weathr_regions['capetown'][0]),
                           slice_d(weathr_regions['capetown'][1]),
                           len(mnths)))

    for m in mnths:
        images_masked = images_monthly_masked(fnames, dnames, y, m,
                                              weathr_regions['capetown'])
        thr[:, :, m-1] = threshold(images_masked)
        vals[:, :, m-1] = cloud_free(images_masked, thr[:, :, m-1])

        fname = band + '_' + str(y) + '_' +'pingu'
        np.save(fname + 'cf', vals)
        np.save(fname + 'thr', thr)
