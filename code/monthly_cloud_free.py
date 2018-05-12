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
from cloud_free import threshold, cloud_free, variable_threshold

cf_dir = 'data/cf/'
thr_dir = 'data/thr/'

def images_monthly_masked(fnames, dnames, y, m, rgn):
    fnames = np.array(fnames)  # array for logical idxing
    idxs = sep_months(dnames, y, m)
    images_masked = load_images_with_region(fnames[idxs], rgn)

    return images_masked


def images_yearly_masked(fnames, dnames, y, rgn):
    fnames = np.array(fnames)  # array for logical idxing
    idxs = sep_years(dnames, y)
    images_masked = load_images_with_region(fnames[idxs], rgn)

    return images_masked

# year of data to look at
years = [2014]
bands = ('vis6', 'vis8', 'nir')  # band to look at
region = 'capetown'

for band in bands:
    paths = path_to_weathr_data(band)  # paths for this band
    months = np.arange(1,13)

    dnames = parse_data_datetime(paths[0], band)  # datetime parsed filenames
    fnames = glob.glob(paths[1])  # filenames

    thr = np.zeros(shape=(slice_d(weathr_regions[region][0]),
                          slice_d(weathr_regions[region][1])))

    vals = np.zeros(shape=(slice_d(weathr_regions[region][0]),
                           slice_d(weathr_regions[region][1])))

    for idx, year in enumerate(years):
        images_masked = images_yearly_masked(fnames, dnames, year,
                                             weathr_regions[region])
#        thr = threshold(images_masked)
        thr = variable_threshold(images_masked)

#        fname = thr_dir + '{}_{}_{}_thr'.format(year, band, region)
        fname = thr_dir + '{}_{}_{}_vthr'.format(year, band, region)
        np.save(fname, thr)

        for jdx, month in enumerate(months):
            images_masked = images_monthly_masked(fnames, dnames,
                                    year, month, weathr_regions[region])
            vals = cloud_free(images_masked, thr)

            fname = cf_dir + '{}_{}_{}_{}_cf'.format(year, month, band, region)
            np.save(fname, vals)
