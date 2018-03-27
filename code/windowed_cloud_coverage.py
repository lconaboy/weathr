import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from util import *
import datetime

def windowed_threshold_cloud_coverage(date_start, x, delta, band, thr, region):
    # Size of window
    window_size = datetime.timedelta(days=delta)
    # Roll window by 1 day
    window_step = datetime.timedelta(days=1)

    # need to make sure there are enough days in the range for an integer
    # number of windows
    # +1 to round up and allow windows to cross years
    x1 = ((x+delta)//delta + 1) * delta
    date_range = datetime.timedelta(days=x1)

    # get list of datetimes and convert to array
    dnames = np.array(parse_data_datetime(path_to_weathr_data(band)[0], band))

    # datetimes for the closed interval (date_start, date_end)
    dates = dnames[dnames >= date_start - window_size/2]
    dates = sorted(dates)[0:x1]

    # get the filenames for those datetimes
    fnames = parse_data_string(dates, path_to_weathr_data(band)[0],
                           band)

    # load the given filenames
    images_masked = load_images_with_region(fnames, weathr_regions[region])

    # find number of land pixels
    n_land_pix = np.sum(image_region(land_mask, weathr_regions[region])==0)
    # find indices of land pixels
    i_land_pix = image_region(land_mask, weathr_regions[region]) == 0

    # this part is for 'windowing' the monthly thresholds
    print('Windowing thresholds')
    # first we preallocate an array the same size as images_masked
    threshold = np.zeros(shape=(thr.shape[0], thr.shape[1], x1))
    # then we create an array with the month for each image
    # e.g. if the 1st image is from August, then mnths[0] = 8
    mnths = np.array([d.month for d in dates])

    # now populate the threshold array with the correct threshold for each
    # month
    for i in range(0, x1):
        m = mnths[i] - 1
        threshold[:, :, i] = thr[:, :, m]

    # now we need to pad threshold on either end, to allow us window at
    # each end of the data set e.g. for the first threshold we will be
    # windowing by +/- delta/2 indices, so we need the thresholds to start
    # at (at least) delta/2
    pad_int = int(delta/2)  # define this useful quantity
    # pad the threshold by delta just to be sure
    threshold_padded = np.pad(threshold, (2*pad_int, 2*pad_int), 'wrap')
    threshold_padded = threshold_padded[2*pad_int:thr.shape[0]+2*pad_int,
                                        2*pad_int:thr.shape[1]+2*pad_int, :]

    # now for each threshold in threshold_padded, average over its
    # neighbours from -delta/2 to +delta/2, this should give a smoothly
    # transitioning window
    for i in range(2*pad_int, x1+2*pad_int):
        avg = threshold_padded[:, :, (i-pad_int):(i+pad_int)] 
        avg = np.mean(avg, axis=2)
        threshold[:, :, (i-2*pad_int)] = avg

    # now that we have the thresholds we move on to calculating cloud
    # coverage
 
    cloud_coverage = np.zeros(x1-delta)  # preallocate
    count = 0  # start counter
    
    for i in range(0, x1-delta):
        # Load all data between start and end dates accounting for new window
        start = date_start + i*window_step
        window_start = date_start - window_size/2
        window_end = start + window_size/2
        # preallocate array for cloud_fraction
        cloud_fraction = np.zeros(window_size.days)

        # pick out the threshold
        thresh = np.mean(threshold[:, :, i:i+delta], axis=2)

        # pick out the images
        images = images_masked[:, :, i:i+delta]

        # iterate over the days in the window
        for d in range(0, delta):
            # find cloud pixels
            cloud_pix = images[:, :, d] > thresh
            cloud_fraction[d] = np.sum(cloud_pix[i_land_pix])/n_land_pix

        cloud_coverage[count] = np.mean(cloud_fraction)  # cloud fraction

        count += 1  # step counter
        print('Windowing image {} [{}/{}]'.format(start, count, x1-delta), end='\r')

    return cloud_coverage


def windowed_cloud_coverage(date_start, x, delta, bands, thr, region):
    # Size of window
    window_size = datetime.timedelta(days=delta)
    # Roll window by 1 day
    window_step = datetime.timedelta(days=1)

    date_range = datetime.timedelta(days=x)

    # find number of land pixels
    n_land_pix = np.sum(image_region(land_mask, weathr_regions[region])==0)
    # find indices of land pixels
    i_land_pix = image_region(land_mask, weathr_regions[region]) == 0

    # now load all of the images for each band
    # first initialise the images array with size
    # (image[0], image[1], n_days, n_bands)
    images_masked = np.zeros(shape=(slice_d(weathr_regions[region][0]),
                                    slice_d(weathr_regions[region][1]),
                                    x, len(bands)))

    for idx, band in enumerate(bands):
        # get list of datetimes and convert to array
        dnames = np.array(parse_data_datetime(path_to_weathr_data(band)[0],
                                              band))

        # datetimes for the closed interval (date_start, date_end)
        dates = dnames[dnames >= date_start - window_size/2]
        dates = sorted(dates)[0:x]

        # get the filenames for those datetimes
        fnames = parse_data_string(dates, path_to_weathr_data(band)[0],
                                   band)

        # load the given filenames
        images_masked[:, :, :, idx] = load_images_with_region(fnames,
                                                              weathr_regions[region])

    # preallocate cloud coverage array
    cloud_coverage = np.zeros(x-delta)
    cloud_coverage_std = np.zeros(x-delta)

    count = 0  # start counter

    # iterate over all the time steps
    for i in range(0, x-delta):
        # Load all data between start and end dates accounting for new window
        start = date_start + i*window_step
        window_start = date_start - window_size/2
        window_end = start + window_size/2
        # preallocate array for cloud_fraction
        cloud_fraction = np.zeros(window_size.days)

        # iterate over the days in the window
        for d in range(0, delta):
            cloud_pix = np.zeros(shape=(slice_d(weathr_regions[region][0]),
                                        slice_d(weathr_regions[region][1])), dtype=bool)

            # pick out the images
            images = images_masked[:, :, i:i+delta, :]

            # now iterate over all of the bands for each time step
            for bdx in range(0, len(bands)):
                # find cloud pixels        
                cloud_pixels = images[:, :, d, bdx] > thr[:, :, bdx]
                # summing boolean arrays yields boolean values
                cloud_pix += cloud_pixels

            cloud_fraction[d] = np.sum(cloud_pix[i_land_pix])/n_land_pix

        cloud_coverage[count] = np.mean(cloud_fraction)
        cloud_coverage_std[count] = np.std(cloud_fraction)

        count += 1  # step counter
        print('Windowing image {} [{}/{}]'.format(start, count, x-delta), end='\r')
        
    return cloud_coverage, cloud_coverage_std


def actual_days(x, delta):
    # need to make sure there are enough days in the range for an integer
    # number of windows
    # +1 to round up and allow windows to cross years

    return ((x+delta)//delta + 1) * delta


years = (2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016)
bands = ('vis6', 'vis8')
region = 'capetown'

x = 365  # try to go for x days
delta = 30  # with a timedelta of delta

x = actual_days(x, delta)

for year in years:
    # load thresholds
    thr = np.zeros(shape=(slice_d(weathr_regions[region][0]),
                          slice_d(weathr_regions[region][1]),
                          len(bands)))
    for idx, band in enumerate(bands):
        thr[:, :, idx] = np.load('{}_{}_{}_thr.npy'.format(year, band, region))
        
    # Start at Jan 01, 20xx
    start_string = '{}01011212'.format(year)
    date_start = datetime.datetime.strptime(start_string, '%Y%m%d%H%M')

    cloud_values = windowed_cloud_coverage(date_start,
                                x, delta, bands, thr, region)

    np.save('{}_multiband_{}_cc'.format(year, region), cloud_values)



