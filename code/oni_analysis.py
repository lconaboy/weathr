import datetime
import numpy as np
import matplotlib.pyplot as plt
from util import *
from coverage_analysis_functions import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from netCDF4 import Dataset
import scipy.stats

"""Mark: to use this, just replace cf with ndvi data."""

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

# load DMI
dmi_datetime, dmi_anoms = load_swio('dmi.nc', 'DMI')
dmi_tmm = swio_three_monthly_means(dmi_datetime, dmi_anoms)

# load the anomaly data and narrow to range
# match the central anomaly months to the smoothed data (i.e. if the
# data is smoothed with a window of 2, then start is actually 2 months
# later)
# step-1 accounts for the ONI being smoothed with a step of 1
new_start, new_end = adjust_dates_to_smoothed_range(start, end, step-1)
# shift the dates to check for correlation
# shift = -2
# new_start, new_end = shift_dates(new_start, new_end, shift)
nino = nino_range('detrend.nino34.ascii.txt', new_start, new_end)
# pick out consecutive anomalies
oni_anoms = consecutive_anomalies(nino)
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
narrowed_dmi = narrow_swio(dmi_tmm, add_month(new_start),
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
# first plot the SWIO/WTIO and DMI (Indian Ocean)
# plot both ONI and DMI
# corr_labels = ['CF', 'ONI', 'DMI']
# plotting_data = [cf_anoms_smoothed, np.array(oni_anoms[-1]),
#                  np.array(narrowed_dmi[2])]
# plot DMI and sst anomalies
corr_labels = ['CF', 'DMI', narrowed_io[1][east_or_south]]
plotting_data = [cf_anoms_smoothed, np.array(narrowed_dmi[2]),
                 np.array(narrowed_io[0][east_or_south][2])]

# plot_three_with_inset_correlations(plotting_data, corr_labels)
plot_three_with_one_fill_between(plotting_data, corr_labels, x_labels, month_step)
plt.title('{}'.format(region_to_string(region)))
plt.axhline(linewidth=1, color='k')
plt.axhline(y=0.5, linewidth=1, color='k', linestyle='dashed')
plt.axhline(y=-0.5, linewidth=1, color='k', linestyle='dashed')
plt.savefig(figure_dir + 'cf_oni_dmi_{}_{}window'.format(region, 2*step+1))

# now plot the ONI (El Nino)
corr_labels = ['CF', 'ONI']
plotting_data = [cf_anoms_smoothed, np.array(oni_anoms[-1])]
plot_two_with_one_fill_between(plotting_data, corr_labels, x_labels, month_step)
plt.title('{}'.format(region_to_string(region)))
plt.axhline(linewidth=1, color='k')
plt.axhline(y=0.5, linewidth=1, color='k', linestyle='dashed')
plt.axhline(y=-0.5, linewidth=1, color='k', linestyle='dashed')
plt.savefig(figure_dir + 'cf_oni_{}_{}m'.format(region, 2*step+1))


plt.figure(figsize=(4,4))
plt.hist((cf_anoms_smoothed[0][~en&~ln], cf_anoms_smoothed[0][en],
          cf_anoms_smoothed[0][ln]), range=([-0.6, 0.6]))
plt.legend([r'Neutral', r'El Ni$\mathrm{\tilde{n}}$o', r'La Ni$\mathrm{\tilde{n}}$a'])
plt.title(region_to_string(region))
plt.xlabel(r'CF$_{\sigma}$')
plt.savefig(figure_dir + 'cf_anoms_hist_' + region)

# now rainfall
end_rainfall = datetime.datetime.strptime('201512','%Y%m')
fname = ['sa_rainfall_1991_2015.txt', 'eth_rainfall_1991_2015.txt']
raw_rf = np.loadtxt(fname[east_or_south], skiprows=1, usecols=(0, 1, 2))
# need to convert to list format to use with yearly_mean, what a bore
# yearly_mean does not play well with rainfall data
rf = [[x for x in raw_rf[:, y]] for y in [1, 2, 0]]
rf_means = yearly_mean(rf, err=False)
# now we need to narrow rf to the range of our data
rf_narr = narrow_to_range(rf, start, end_rainfall)
rf_anoms = cloud_fraction_anomalies(rf_narr, rf_means, start, end_rainfall, err=False)
rf_anoms_smoothed = smooth_data_with_window(rf_anoms, step, err=False)
cf = load_cloud_fraction_period(start, end_rainfall, region)
cf_means = yearly_mean(cf)
cf_anoms = cloud_fraction_anomalies(cf, cf_means, start, end_rainfall)
cf_anoms_smoothed = smooth_data_with_window(cf_anoms, step)
legends = ['CF', 'RF']

month_step = 6
month_labels = cf_anoms[1][::month_step]
year_labels = np.arange(min(cf_anoms[0]), max(cf_anoms[0])+1)
x_labels = month_and_year_labels(month_labels, year_labels, month_step)

def plot_rainfall_and_cloud(cf_anoms_smoothed, rf_anoms_smoothed, legends, x_labels):
    fig, ax1 = plt.subplots(figsize=(5.2, 5))
    plt.axhline(linewidth=1, color='k')
    ax1.plot(cf_anoms_smoothed[0], label=legends[0])
    ax1.set_ylim([-0.8, 0.8])
    ax1.set_ylabel(legends[0]+r'$_{\sigma}$')
    ax2 = ax1.twinx()
    ax2.plot(rf_anoms_smoothed, label=legends[1], color='r')
    ax2.set_ylabel(legends[1]+r'$_{\sigma}$')
    ax2.set_ylim([-0.6, 0.6])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc=2)
    ax1.set_xticks([])
    ax2.set_xticks(np.linspace(ax1.get_xbound()[0], ax1.get_xbound()[1], len(x_labels)))
    ax2.set_xticklabels(x_labels)
    ax1.set_xlabel('Months')
    ax2.set_xlabel('Months')
    plt.title(region_to_string(region))
    plt.tight_layout()
    plt.savefig(figure_dir + 'rf_cf_anomalies_{}'.format(region))

plot_rainfall_and_cloud(cf_anoms_smoothed, rf_anoms_smoothed, legends, x_labels)
