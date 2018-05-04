import datetime
import numpy as np
import matplotlib.pyplot as plt
from util import *
from coverage_analysis_functions import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from netCDF4 import Dataset

regions = ['capetown', 'eastafrica']
east_or_south = 0  # 0 for south, 1 for east
region = regions[east_or_south]
start = datetime.datetime.strptime('20081','%Y%m')
end = datetime.datetime.strptime('201712','%Y%m')
step = 2  # for smoothing

# first we look at cloud fraction
cf = load_cloud_fraction_period(start, end, region)
cf_means = yearly_mean(cf)
cf_anoms = cloud_fraction_anomalies(cf, cf_means, start, end)
cf_anoms_smoothed = smooth_data_with_window(cf_anoms, step)

# load SWIO
swio_datetime, swio_anoms = load_swio('swio.nc', 'SWIO')
wtio_datetime, wtio_anoms = load_swio('wtio.nc', 'WTIO')
swio_tmm = swio_three_monthly_means(swio_datetime, swio_anoms)
wtio_tmm = swio_three_monthly_means(wtio_datetime, wtio_anoms)

# # now rainfall
# fname = 'eth_rainfall_1991_2015.txt'
# raw_rf = np.loadtxt(fname, skiprows=1, usecols=(0, 1, 2))
# # need to convert to list format to use with yearly_mean, what a bore
# # yearly_mean does not play well with rainfall data
# rf = [[x for x in raw_rf[:, y]] for y in [1, 2, 0]]
# rf_means = yearly_mean(rf, err=False)
# # now we need to narrow rf to the range of our data
# rf_narr = narrow_to_range(rf, start, end)
# rf_anoms = cloud_fraction_anomalies(rf_narr, rf_means, start, end, err=False)
# rf_anoms_smoothed = smooth_data_with_window(rf_anoms[3], step)

# load the anomaly data and narrow to range
# match the central anomaly months to the smoothed data (i.e. if the
# data is smoothed with a window of 2, then start is actually 2 months
# later)
# step-1 accounts for the ONI being smoothed with a step of 1
new_start, new_end = adjust_dates_to_smoothed_range(start, end, step-1)
# shift the dates to check for correlation
# shifted_start, shifted_end = shift_dates(new_start, new_end, shift)
nino = nino_range('detrend.nino34.ascii.txt', new_start, new_end)
# pick out consecutive anomalies
oni_anoms = consecutive_anomalies(nino, new_start, new_end)
en = np.array(oni_anoms[2])
ln = np.array(oni_anoms[3])
tot = np.ones(shape=oni_anoms[2].shape, dtype=bool)  # idxs for all data
idxs = [tot, en, ln]

# narrow swio to shifted range, add and subtract a month to account
# for the fact that new_start and new_end are shorter by a month (as
# three monthly mean is taken later for Nino)
narrowed_swio = narrow_swio(swio_tmm, add_month(new_start),
                            subtract_month(new_end))
narrowed_wtio = narrow_swio(wtio_tmm, add_month(new_start),
                            subtract_month(new_end))

narrowed_io = [[narrowed_swio, narrowed_wtio], ['SWIO', 'WTIO']]

# titles = ['All', 'El Nino', 'La Nina']
# corr_labels = ['CF', 'RF', 'ONI']
# plotting_data = [cf_anoms_smoothed, rf_anoms_smoothed, np.array(anoms[-1])]
# plot_idxs_with_inset_correlations(plotting_data, idxs, titles, corr_labels)
# plt.show()

# corr_labels = ['CF', 'ONI']
# plotting_data = [cf_anoms_smoothed, np.array(oni_anoms[-1])]
# plot_with_inset_correlations(plotting_data, corr_labels)
# plt.suptitle('ONI shifted by {} months \n {}'.format(shift, region))
# #plt.savefig('insetcorr_{}_onishift_{}'.format(region, shift))
# plt.show()

# take every third label from the SWIO index for use on the xticks
month_step = 4
month_labels = narrowed_io[0][east_or_south][1][::month_step]
year_labels = np.arange(min(narrowed_io[0][east_or_south][0]),
                        max(narrowed_io[0][east_or_south][0])+1)


x_labels = month_and_year_labels(month_labels, year_labels, month_step)
corr_labels = ['CF', 'ONI', narrowed_io[1][east_or_south]]
plotting_data = [cf_anoms_smoothed, np.array(oni_anoms[-1]),
                 np.array(narrowed_io[0][east_or_south][2])]
# plot_three_with_inset_correlations(plotting_data, corr_labels)

# plot_three_with_one_barred(plotting_data, corr_labels)
plot_three_with_one_fill_between(plotting_data, corr_labels, x_labels, month_step)
plt.title('{}'.format(region_to_string(region)))
plt.axhline(linewidth=1, color='k')
plt.savefig(figure_dir + 'oni_' + narrowed_io[1][east_or_south] +
            '_{}_{}window'.format(region, 2*step+1))
plt.show()
