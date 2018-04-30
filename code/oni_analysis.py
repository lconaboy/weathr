import datetime
import numpy as np
import matplotlib.pyplot as plt
from util import *
from coverage_analysis_functions import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd

region = 'capetown'
start = datetime.datetime.strptime('20081','%Y%m')
end = datetime.datetime.strptime('201512','%Y%m')
step = 2  # for smoothing

# first we look at cloud fraction
cf = load_cloud_fraction_period(start, end, region)
cf_means = yearly_mean(cf)
cf_anoms = cloud_fraction_anomalies(cf, cf_means, start, end)
cf_anoms_smoothed = smooth_data_with_window(cf_anoms[3], step)

# now rainfall
fname = 'sa_rainfall_1991_2015.txt'
raw_rf = np.loadtxt(fname, skiprows=1, usecols=(0, 1, 2))
# need to convert to list format to use with yearly_mean, what a bore
# yearly_mean does not play well with rainfall data
rf = [[x for x in raw_rf[:, y]] for y in [1, 2, 0]]
rf_means = yearly_mean(rf, err=False)
# now we need to narrow rf to the range of our data
rf_narr = narrow_to_range(rf, start, end)
rf_anoms = cloud_fraction_anomalies(rf_narr, rf_means, start, end, err=False)
rf_anoms_smoothed = smooth_data_with_window(rf_anoms[3], step)

# load the anomaly data and narrow to range
# match the central anomaly months to the smoothed data (i.e. if the
# data is smoothed with a window of 2, then start is actually 2 months
# later)
# step-1 accounts for the ONI being smoothed with a step of 1
new_start, new_end = adjust_dates_to_smoothed_range(start, end, step-1)
# shift the dates to check for correlation
shift = 0
shifted_start, shifted_end = shift_dates(new_start, new_end, shift)
nino = nino_range('detrend.nino34.ascii.txt', shifted_start, shifted_end)
# pick out consecutive anomalies
anoms = consecutive_anomalies(nino, shifted_start, shifted_end)
en = np.array(anoms[2])
ln = np.array(anoms[3])
tot = np.ones(shape=anoms[2].shape, dtype=bool)  # idxs for all data
idxs = [tot, en, ln]

titles = ['All', 'El Nino', 'La Nina']
corr_labels = ['CF', 'RF', 'ONI']
plotting_data = [cf_anoms_smoothed, rf_anoms_smoothed, np.array(anoms[-1])]
plot_with_inset_correlations(plotting_data, idxs, titles, corr_labels)
plt.suptitle('ONI shifted by {} months'.format(shift))
# plt.savefig('insetcorr_{}_onishift_{}'.format(region, shift))
# plt.show()
