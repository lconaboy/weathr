import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from util import *


def split_region(w_region, direction):
    x = w_region[0]
    y = w_region[1]

    north = [slice(x.start, x.start + slice_d(x)//2), y]
    south = [slice(x.start + slice_d(x)//2, x.stop), y]
    east = [x, slice(y.start + slice_d(y)//2, y.stop)]
    west = [x, slice(y.start, y.start + slice_d(y)//2)]

    # raw slices (i.e. slice(0, n)) for thresholds
    north_raw = [slice(0, slice_d(north[0]), None),
                 slice(0, slice_d(north[1]), None),]
    south_raw = [slice(slice_d(south[0]), 2*slice_d(south[0]), None),
                 slice(0, slice_d(south[1]), None),]
    
    east_raw = [slice(0, slice_d(east[0]), None),
                slice(slice_d(east[1]), 2*slice_d(east[1]), None)]
    west_raw = [slice(0, slice_d(west[0]), None),
                slice(0, slice_d(west[1]), None)]

    if direction == 'n': return [north, north_raw]
    if direction == 's': return [south, south_raw]
    if direction == 'e': return [east, east_raw]
    if direction == 'w': return [west, west_raw]


def cloud_coverage(region, y, m, direction=None):
    """Multiband cloud coverage algorithm. Now can split regions into north 'n', 
    south 's', east 'e' and west 'w'."""
    if not direction:
        # bands for multiband
        bands = ('vis6','vis8')
        # initialise land pixel constants
        n_land_pix = np.sum(image_region(land_mask,
                                         weathr_regions[region]) == 0)  # count
        i_land_pix = image_region(land_mask, weathr_regions[region]) == 0  # indices

        dnames = np.array(parse_data_datetime(path_to_weathr_data(bands[0])[0],
                                              bands[0]))
        # image shape
        xshape = slice_d(weathr_regions[region][0])  # image x length
        yshape = slice_d(weathr_regions[region][1])  # image y length
        idxs = sep_months(dnames, y, m)  # need indexes to work out
        # number of days in month
        zshape = sum(np.array(idxs)==True)  # number of days
        # preallocate
        img = np.zeros(shape=(xshape, yshape, zshape, len(bands)))  # images
        thr = np.zeros(shape=(xshape, yshape, len(bands)))  # thresholds
        cld = np.zeros(shape=(xshape, yshape, zshape))  # cloud masks
        cov = np.zeros(zshape)  # coverage

        # now load all of the images and thresholds that we will need
        for idx, band in enumerate(bands):
            # first, load thresholds
            thr[:, :, idx] = np.load('data/thr/{}_{}_{}_thr.npy'.format(y, band, region))

            # now for the images
            # get full list datetime names and convert to array
            dnames = np.array(parse_data_datetime(path_to_weathr_data(band)[0],
                                                  band))
            # pick out relevant months
            dnames = dnames[sep_months(dnames, y, m)]
            # convert back to filenames
            fnames = parse_data_string(dnames, path_to_weathr_data(band)[0],
                                       band)
            # load
            tmp = load_images_with_region(fnames, weathr_regions[region])  # images
            img[:, :, :, idx] = tmp

        # now for multiband cloud coverage -- would vectorising this make it
        # faster? is it worth my precious time?
        for jdx in range(0, zshape):
            # initialise for comparisons
            arr = np.zeros(shape=(xshape, yshape, len(bands)), dtype=bool)
            for kdx in range(0, len(bands)):
                # compare the image for that day with the threshold in the
                # relevant band
                arr[:, :, kdx] = img[:, :, jdx, kdx] > thr[:, :, kdx]

            # numpy sums return Boolean values
            # (i.e. np.sum([True, True]) = True)
            cld[:, :, jdx] = np.sum(arr, axis=2, dtype=bool)  # mask
            cov[jdx] = sum(cld[:,:,jdx][i_land_pix])/n_land_pix  # fraction

        # # lets av a look at the distribution then
        # plt.figure()
        # plt.hist(cov, histtype='step', color='k')
        # plt.title('Cloud fraction distribution {}/{}'.format(m, y))
        # plt.xlabel('CF')
        # plt.axvline(np.median(cov), color='r')
        # plt.axvline(np.median(cov)+abs(np.diff(np.percentile(cov, [75, 25])))/2,
        #             color='r', linestyle='dashed')
        # plt.axvline(np.mean(cov), color='b')
        # plt.axvline(np.mean(cov)+np.std(cov), color='b', linestyle='dashed')
        # plt.legend(['Median', 'Median + IQR/2', 'Mean', r'Mean + $\sigma$'])
        # plt.savefig('../diary/multiband_cf_dist_{}'.format(m))

        # # this line is for producing the distribution histograms,
        # # delete after they have been made
        # np.save('data/cloud/multiband_cf_dist_{}_{}_{}'.format(m, y, region),
        #         cov)

        # using builtin sum over the month returns a heatmap
        # of cloud coverage
        cld = cld.sum(2)

        # cov = np.array([np.mean(cov), np.std(cov)])
        # now using the median and IQR/2 to calculate errors on cloud fraction, after
        # looking at the distribution of cloud coverage this seems most appropriate
        cov = np.array([np.median(cov), abs(np.diff(np.percentile(cov, [75, 25])))/2])

        np.save('./data/cloud/{}_{}_multiband_{}_med_cloud'.format(m, y, region),
               [cld, cov])
        print('Calculated cloud coverage for {}/{}'.format(m, y), end='\r')

    else:
        # bands for multiband
        bands = ('vis6','vis8')
        # split region into direction
        w_region, w_region_thr = split_region(weathr_regions[region], direction)
        
        # initialise land pixel constants
        n_land_pix = np.sum(image_region(land_mask, w_region) == 0)  # count
        i_land_pix = image_region(land_mask, w_region) == 0  # indices

        dnames = np.array(parse_data_datetime(path_to_weathr_data(bands[0])[0],
                                              bands[0]))
        # image shape
        xshape = slice_d(w_region[0])  # image x length
        yshape = slice_d(w_region[1])  # image y length
        idxs = sep_months(dnames, y, m)  # need indexes to work out
        # number of days in month
        zshape = sum(np.array(idxs)==True)  # number of days
        # preallocate
        img = np.zeros(shape=(xshape, yshape, zshape, len(bands)))  # images
        thr = np.zeros(shape=(xshape, yshape, len(bands)))  # thresholds
        cld = np.zeros(shape=(xshape, yshape, zshape))  # cloud masks
        cov = np.zeros(zshape)  # coverage

        # now load all of the images and thresholds that we will need
        for idx, band in enumerate(bands):
            # first, load thresholds and slice to new region
            thr[:, :, idx] = np.load('data/thr/{}_{}_{}_thr.npy'.format(y, band,
                                                            region))[w_region_thr]

            # now for the images
            # get full list datetime names and convert to array
            dnames = np.array(parse_data_datetime(path_to_weathr_data(band)[0],
                                                  band))
            # pick out relevant months
            dnames = dnames[sep_months(dnames, y, m)]
            # convert back to filenames
            fnames = parse_data_string(dnames, path_to_weathr_data(band)[0],
                                       band)
            # load
            tmp = load_images_with_region(fnames, w_region)  # images
            img[:, :, :, idx] = tmp

        # now for multiband cloud coverage -- would vectorising this make it
        # faster? is it worth my precious time?
        for jdx in range(0, zshape):
            # initialise for comparisons
            arr = np.zeros(shape=(xshape, yshape, len(bands)), dtype=bool)
            for kdx in range(0, len(bands)):
                # compare the image for that day with the threshold in the
                # relevant band
                arr[:, :, kdx] = img[:, :, jdx, kdx] > thr[:, :, kdx]

            # numpy sums return Boolean values
            # (i.e. np.sum([True, True]) = True)
            cld[:, :, jdx] = np.sum(arr, axis=2, dtype=bool)  # mask
            cov[jdx] = sum(cld[:,:,jdx][i_land_pix])/n_land_pix  # fraction

        # using builtin sum over the month returns a heatmap
        # of cloud coverage
        cld = cld.sum(2)
        # cov = np.array([np.mean(cov), np.std(cov)])

        # now using the median and IQR/2 to calculate errors on cloud
        # fraction, after looking at the distribution of cloud
        # coverage this seems most appropriate
        cov = np.array([np.median(cov), abs(np.diff(np.percentile(cov, [75, 25])))/2])

        np.save('./data/cloud/{}_{}_multiband_{}_med_cloud'.format(m, y, w_region),
               [cld, cov])
        print('Calculated cloud coverage for {}/{}'.format(m, y), end='\r')


