import numpy as np
import matplotlib.pyplot as plt
import datetime
from util import *
from coverage_analysis_functions import *

figure_dir = 'results/cloud/'


def all_monthly_cloud_mask_means(region):
    """Calculates all the monthly means of the cloud masks (i.e. the means of all masks for January, February, ...)"""
    start = datetime.datetime.strptime('20081', '%Y%m')  # earliest data
    end = datetime.datetime.strptime('201712', '%Y%m')  # latest data
    data = load_cloud_mask_period(start, end, region)
    months = np.array(data[1])
    masks = np.array(data[2])
    means = []
    
    for i in range(0, 12):
        # calculate the mean value of the cloud mask month wise,
        # i.e. the average value of each pixel across all of the
        # available months
        means.append(np.mean(masks[months==i+1], axis=0))

    return means


def cloud_mask_anomalies(data, means, region):
    """Calculates cloud mask anomalies as elsewhere, such that x_o = (x-u)/u"""
    months = np.array(data[1])
    masks = np.array(data[2])
    means = np.array(means)
    anoms = np.zeros(masks.shape)
    
    for idx, m in enumerate(masks):
        # catch pixels with no cloud
        anoms[idx, :, :] = np.divide((m-means[months[idx]-1]), means[months[idx]-1],
                                     out=np.zeros_like(m),
                                     where=(means[months[idx]-1]!=0))

    return anoms


vals = {'en': [datetime.datetime.strptime('20151', '%Y%m'),
               datetime.datetime.strptime('20161', '%Y%m'),
               '{} El Nino'.format(region_to_string(region)),
               'en_spatial_cloud_{}'.format(region)],
        'ln': [datetime.datetime.strptime('20111', '%Y%m'),
               datetime.datetime.strptime('20121', '%Y%m'),
               '{} La Nina'.format(region_to_string(region)),
               'ln_spatial_cloud_{}'.format(region)]}
region = 'capetown'


event = 'ln'                    # choose which event to look at

start = vals[event][0]
end = vals[event][1]

means = all_monthly_cloud_mask_means(region)
data = load_cloud_mask_period(start, end, region)
anoms = cloud_mask_anomalies(data, means, region)

date = start
titles = []
while date < end:
    titles.append('{}/{}'.format(date.month, date.year))
#    titles.append('{}'.format(date.month))
    date = add_month(date)
    
fig, axes = plt.subplots(3, 4, figsize=(8, 6))
for idx, ax in enumerate(axes.ravel()):
    im = ax.imshow(anoms[idx], cmap='seismic', vmin=-10, vmax=10)
    ax.set_title(titles[idx])
    ax.axis('off')

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
cbar.set_label(r'$x_{\sigma}$')
plt.suptitle('Spatial distribution of cloud anomalies \n' + vals[event][2])
plt.savefig(figure_dir + vals[event][3])
plt.show()
