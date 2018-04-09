import numpy as np
import matplotlib.pyplot as plt
from util import *
from PIL import Image
import datetime

region = 'eastafrica'  #current region
bands = ('vis6','vis8')  # bands for multiband
#years = np.arange(2013, 2018)
months = np.arange(1,13)
years = np.arange(2009, 2018)
count = 0  # counter for output

# GRIM LOOPS
for y in years:
    for m in months:
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
            thr[:, :, idx] = np.load('{}_{}_{}_thr.npy'.format(y, band, region))

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
            cld[:, :, jdx] = np.sum(arr, axis=2)  # mask
            cov[jdx] = sum(cld[:,:,jdx][i_land_pix])/n_land_pix  # fraction

        # using builtin sum over the month returns a heatmap
        # of cloud coverage
        cld = cld.sum(2)
        cov = np.array([np.mean(cov), np.std(cov)])

        np.save('{}_{}_multiband_{}_cloud'.format(m, y, region), [cld, cov])
        count += 1
        print('{}/{}'.format(count, len(years)*len(months)), end = '\r')

