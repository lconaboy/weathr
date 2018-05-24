import numpy as np
import matplotlib.pyplot as plt
from util import *
from PIL import Image
import datetime

vals = []

ndvi_dir = 'results/ndvi/'
# yearmonth_region.png
# e.g. 201708_capetown.png
ndvi_fmt = '{}{}_{}.npy'

region = 'eastafrica'
mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)


for m in ['01', '07']:
    for y in range(2008, 2018):
        ctmp = np.load('multiband_cf_dist_{}_{}.npy'.format(int(m), y))
        ntmp = np.load(ndvi_dir + ndvi_fmt.format(y, m, region))[mask]

    vals.append(ctmp.ravel())
#    vals.append(ntmp.ravel())

fig, axes = plt.subplots(2, 1, figsize=(4, 8))
axes_flat = axes.ravel()

for i, ax in enumerate(axes_flat):
    ax.hist(vals[i], histtype='step', color='k')
    mu = np.mean(vals[i])
    std = np.std(vals[i])
    med = np.median(vals[i])
    iqr = abs(np.diff(np.percentile(vals[i], [25, 75])))/2
    ax.axvline(med, color='b')
    ax.axvline(med+iqr, color='b', linestyle='dashed')
    ax.axvline(mu, color='r')
    ax.axvline(mu+std, color='r', linestyle='dashed')
    ax.set_ylabel('Days')

axes_flat[0].legend(['Median', 'Median + IQR/2', r'$\mu$', r'$\mu + \sigma$'])
axes_flat[0].set_title('January')
axes_flat[1].set_title('July')
axes_flat[1].set_xlabel('CF')
fig.suptitle('Monthly distribution of cloud fraction over 2008-2017 \n {}'.format(region_to_string(region)))
plt.show()
