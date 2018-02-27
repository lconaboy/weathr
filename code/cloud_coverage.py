import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from util import *


def cloud_coverage(fnames, thr):
    months = sep_months(fnames, year, band)
    # number of land pixels is the same each time
    n_land_pix = np.sum(land_mask == 0)

    monthly_fraction = np.zeros(len(set(months)))
    monthly_fraction_sig = np.zeros(len(set(months)))
    
    for i in range(0, len(set(months))):
        month_fn = fnames[months == i+1]
        images_masked = load_images_with_region(month_fn,
                                                weathr_regions['capetown'])
        cloud_fraction = np.zeros(len(month_fn))

        for j in range(0, len(month_fn)):
            cloud_pix = images_masked[:, :, j] > thr[:, :, i]
            n_cloud_pix = np.sum(cloud_pix)
            cloud_fraction[j] = n_cloud_pix/n_land_pix

        monthly_fraction[i] = np.mean(cloud_fraction)
        monthly_fraction_sig[i] = np.std(cloud_fraction)
        
    return (monthly_fraction, monthly_fraction_sig)


year = '08'
band = 'vis8'

# need to convert to array for logical indexing
fnames = np.array(glob.glob(path_to_weathr_data(year, band))) 

thr_fname = 'thr_' + year + '_' + band + '.npy'
thr = np.load(thr_fname)

cloud = cloud_coverage(fnames, thr)

monthly = cloud[0]
monthly_sig = cloud[1]

monthly_fn = 'cc_monthly_' + year + '_' + band

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

# new = np.load('cc_monthly_17_vis8.npy')
# old = np.load('cc_monthly_08_vis8.npy')
# plt.figure()
# plt.plot(new[0], color='r')
# plt.plot(new[1], color='b')
# #plt.errorbar(np.arange(1,13), new[0], new[1], color='r')
# #plt.errorbar(np.arange(1,13), old[0], old[1], color='b')
# plt.xlabel('Month')
# plt.ylabel('Cloud Fraction')
# plt.legend(['2017', '2008'])
# plt.title('VIS8 Monthly Cloud Fraction')
# plt.savefig('cc_monthly_0817_vis8')
# plt.show()
