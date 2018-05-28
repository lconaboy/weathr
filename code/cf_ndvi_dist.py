import numpy as np
import matplotlib.pyplot as plt
from util import *
from PIL import Image
import datetime

vals = [[], []]

ndvi_dir = 'results/ndvi/'
# yearmonth_region.png
# e.g. 201708_capetown.png
ndvi_fmt = '{}{}_{}.npy'

region = 'eastafrica'
mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)

for i, m in enumerate(['01', '07']):
    ctmp = []
    for y in range(2008, 2018):
        ctmp.append(np.load('data/cloud/multiband_cf_dist_{}_{}_{}.npy'.format(
            int(m), y, region)).ravel())

    vals[i] = np.concatenate(ctmp).ravel()
#    vals.append(ntmp.ravel())

fig, axes = plt.subplots(2, 1, figsize=(4, 8))
axes_flat = axes.ravel()

for i, ax in enumerate(axes_flat):
    ax.hist(vals[i], color='C0', alpha=0.5)
    mu = np.mean(vals[i])
    std = np.std(vals[i])
    med = np.median(vals[i])
    iqr = abs(np.diff(np.percentile(vals[i], [25, 75])))/2
    ax.axvline(med, color='C3')
    ax.axvline(med+iqr, color='C3', linestyle='dashed')
    ax.axvline(mu, color='darkgreen')
    ax.axvline(mu+std, color='darkgreen', linestyle='dashed')
    ax.set_ylabel('Days')

axes_flat[0].legend([r'$\tilde{x}$', r'$\tilde{x}$ + IQR/2', r'$\mu$',
                     r'$\mu + \sigma$'])
axes_flat[0].set_title('January')
axes_flat[1].set_title('July')
axes_flat[1].set_xlabel('CF')
fig.suptitle(region_to_string(region))
plt.subplots_adjust(left=0.2, top=0.9, bottom=0.05)
plt.savefig(figure_dir + 'cf_monthly_dist_{}'.format(region))
plt.show()
