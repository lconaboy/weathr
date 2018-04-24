import numpy as np
import cloud_free as cf
from util import *
from ndvi import *
import os
import glob

ndvi_dir = 'results/ndvi/'
# yearmonth_region.png
# e.g. 201708_capetown.png
ndvi_fmt = '{}{}_{}'

threshold_dir = 'data/thr/'
# region_year_band_thr.npy
# e.g. capetown_2017_vis8_thr.npy
threshold_fmt = '{}_{}_{}_thr.npy'

def ndvi_for_year_and_region(year, region):
    """Calculates NDVI for every month in given year and region. Output is
saved as numpy array into configured threshold_dir and threshold_fmt."""
    # Assume that thresholds exist.
    thresholds = {band: np.load(threshold_dir + threshold_fmt.format(region, year, band))
                  for band in ('vis6', 'vis8')}

    for month in np.arange(1, 13):
        m = '{:02d}'.format(month) # pad with zero

        nir = load_images_with_region(glob.glob("./data/eumetsat/{}{}*_vis8.png".format(year, m)),
                                           weathr_regions[region])
        vis = load_images_with_region(glob.glob("./data/eumetsat/{}{}*_vis6.png".format(year, m)),
                                           weathr_regions[region])

        nir_avg = average_with_thresholds(nir, thresholds['vis8'])
        vis_avg = average_with_thresholds(vis, thresholds['vis6'])

        nir_avg_cal = calibrate(nir_avg, 'vis8')
        # Make sure non-land pixels don't interfere with plotting.
        nir_avg_cal *= (1 - image_region(land_mask, weathr_regions[region]))
        vis_avg_cal = calibrate(vis_avg, 'vis6')
        vis_avg_cal *= (1 - image_region(land_mask, weathr_regions[region]))

        ndvi_month = ndvi(nir_avg_cal, vis_avg_cal)
        np.save(ndvi_dir + ndvi_fmt.format(year, m, region), ndvi_month)

    return None

for year in np.arange(2009, 2018):
    for region in ('capetown', 'eastafrica'):
        print('Producing NDVI data for {} {}'.format(region, year))
        ndvi_for_year_and_region(year, region)
