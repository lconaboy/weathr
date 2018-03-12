import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from util import *

def cloud_coverage(fnames, dnames, y, thr, rgn):
    months = np.arange(1,13)
    fnames = np.array(fnames)  # for logical idxing
    # number of land pixels is the same each time
    n_land_pix = np.sum(land_mask == 0)

    monthly_fraction = np.zeros(len(set(months)))
    monthly_fraction_sig = np.zeros(len(set(months)))

    for m in months:
        images_masked = images_monthly_masked(fnames, dnames, y, m, rgn)
        cloud_fraction = np.zeros(images_masked.shape[2])

        for j in range(0, images_masked.shape[2]):
            cloud_pix = images_masked[:, :, j] > thr[:, :, m-1]
            n_cloud_pix = np.sum(cloud_pix)
            cloud_fraction[j] = n_cloud_pix/n_land_pix

        monthly_fraction[m-1] = np.mean(cloud_fraction)
        monthly_fraction_sig[m-1] = np.std(cloud_fraction)

    return (monthly_fraction, monthly_fraction_sig)


y = 2008
band = 'vis8'

# need to convert to array for logical indexing
fnames = np.array(glob.glob(path_to_weathr_data(band)[1]))
dnames = parse_data_datetime(path_to_weathr_data(band)[0], band)

thr = np.load('vis8_2017_thr.npy')

cloud = cloud_coverage(fnames, dnames, y, thr, weathr_regions['capetown'])

monthly = cloud[0]
monthly_sig = cloud[1]

monthly_fn = 'cc_monthly_' + str(y) + '_' + band + '_newdata'

np.save(monthly_fn, np.array([monthly, monthly_sig]))

# plt.figure(figsize=(6, 4))
# plt.plot(daily, color='#A93226')
# plt.title('20' + year + ' - ' + band)
# plt.legend(['Daily'])
# plt.xlabel('Day')
# plt.ylabel('Cloud Fraction')
# plt.savefig(daily_fn)
# plt.show()

# plt.figure()
# plt.plot(monthly, color='#229954')
# plt.title('20' + year + ' - ' + band)
# plt.xlabel('Month')
# plt.ylabel('Cloud Fraction')
# plt.legend(['Monthly'])
# plt.savefig(month_fn)
# plt.show()

# new = np.load('cc_monthly_2017_vis8_newdata.npy')
# old = np.load('cc_monthly_2008_vis8_newdata.npy')
# plt.figure()
# plt.plot(new[0], color='r')
# plt.plot(new[1], color='b')
# #plt.errorbar(np.arange(1,13), new[0], new[1], color='r')
# #plt.errorbar(np.arange(1,13), old[0], old[1], color='b')
# plt.xlabel('Month')
# plt.ylabel('Cloud Fraction')
# plt.ylim([0, 0.15])
# plt.xlim([0,11])
# plt.legend(['2017', '2008'])
# plt.title('VIS8 Monthly Cloud Fraction')
# plt.tight_layout()
# plt.savefig('cc_monthly_0817_vis8_newdata')
# plt.show()
