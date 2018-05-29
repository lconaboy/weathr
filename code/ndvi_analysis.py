import numpy as np
import cloud_free as cf
from util import *
from ndvi import *
import os
import glob
import matplotlib.pyplot as plt

ndvi_dir = 'results/ndvi/'
# yearmonth_region.png
# e.g. 201708_capetown.png
ndvi_fmt = '{}{}_{}'

threshold_dir = 'data/thr/'
# region_year_band_thr.npy
# e.g. capetown_2017_vis8_thr.npy
threshold_fmt = '{}_{}_{}_thr.npy'

def ndvi_for_year_and_region(year, region):
    """Calculates NDVI for every month in given year and region. Output is
saved as numpy array into configured threshold_dir and threshold_fmt."""
    # Assume that thresholds exist.
    thresholds = {band: np.load(threshold_dir + threshold_fmt.format(region, year, band))
                  for band in ('vis6', 'vis8')}

    for month in np.arange(1, 13):
        m = '{:02d}'.format(month) # pad with zero

        nir = load_images_with_region(glob.glob("./data/eumetsat/{}{}*_vis8.png".format(year, m)),
                                           weathr_regions[region])
        vis = load_images_with_region(glob.glob("./data/eumetsat/{}{}*_vis6.png".format(year, m)),
                                           weathr_regions[region])

        nir_avg = average_with_thresholds(nir, thresholds['vis8'])
        vis_avg = average_with_thresholds(vis, thresholds['vis6'])

        nir_avg_cal = calibrate(nir_avg, 'vis8')
        # Make sure non-land pixels don't interfere with plotting.
        nir_avg_cal *= (1 - image_region(land_mask, weathr_regions[region]))
        vis_avg_cal = calibrate(vis_avg, 'vis6')
        vis_avg_cal *= (1 - image_region(land_mask, weathr_regions[region]))

        ndvi_month = ndvi(nir_avg_cal, vis_avg_cal)
        np.save(ndvi_dir + ndvi_fmt.format(year, m, region), ndvi_month)

    return None

def calibration_comparison(year, month, region):
    from ndvi_spatial_analysis_functions import colourmaps
    # Try load NDVI data
    ndvi_uncal = np.load(ndvi_dir + ndvi_fmt.format(year, month, region) + "_uncalibrated.npy")
    ndvi_cal   = np.load(ndvi_dir + ndvi_fmt.format(year, month, region) + ".npy")

    import matplotlib.pyplot as plt
    cmap = colourmaps(15)[0]
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,4.5), dpi=150)
    # ax1.set_title('Uncalibrated NDVI for {} {}, {}'.format(region_to_string(region), month, year))
    ax1.axis('off')
    plot1 = ax1.imshow(ndvi_uncal, cmap=cmap, vmin=-np.max(ndvi_uncal),
                       vmax=np.max(ndvi_uncal), origin='lower', interpolation='nearest')
    # ax2.set_title('Calibrated NDVI for {} {}, {}'.format(region_to_string(region), month, year))
    ax2.axis('off')
    plot2 = ax2.imshow(ndvi_cal, cmap=cmap, vmin=-np.max(ndvi_cal),
                       vmax=np.max(ndvi_cal), origin='lower', interpolation='nearest')
    f.subplots_adjust(right=0.8)
    cb_ax = f.add_axes([0.85, 0.15, 0.025, 0.7])
    f.colorbar(plot2, cax=cb_ax, orientation='vertical', extend='both').set_label('NDVI')
    plt.savefig(figure_dir + 'ndvi_calibration_comparison.pdf')

    return None

ndvi_monthly_means_fmt = 'monthly_means_{}'
def ndvi_monthly_means(region):
    """Calculate monthly means for region."""
    mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)

    avg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in np.arange(1, 13):
        m = '{:02d}'.format(i) # pad with zero
        f = np.dstack([np.load(fname) for fname in glob.glob(ndvi_dir + ndvi_fmt.format('*', m, region) + '.npy')])
        avg[i - 1] = np.mean(f[mask])

    return avg

def plot_ndvi_monthly_and_means(region):
    mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)
    path = ndvi_dir + ndvi_fmt

    avgs = ndvi_monthly_means(region)

    fnames = sorted(glob.glob(ndvi_dir + ndvi_fmt.format('*', '*', region) + '.npy'))
    monthlys = np.dstack([np.mean(np.load(fname)[mask]) for fname in fnames]).ravel()

    plt.figure(figsize=(10.5,6))
    # plt.bar(np.arange(len(monthlys)), monthlys, label='Monthly NDVI')
    plt.plot(monthlys, label='Monthly NDVI')
    plt.plot(np.tile(avgs, 10), c='r', label='Average monthly NDVI for whole dataset')
    plt.ylim([np.min(monthlys)-0.05*np.min(monthlys), np.max(monthlys) + 0.05*np.max(monthlys)])
    plt.xlim(0, len(monthlys))
    plt.xticks(np.arange(0, 10)*12, np.arange(2008, 2018), rotation=45)
    plt.title('NDVI for {} 2008-2018'.format(region_to_string(region)))
    plt.xlabel('Month')
    plt.ylabel('NDVI')
    plt.legend()
    plt.savefig(figure_dir + 'ndvi_monthly_and_means_{}.png'.format(region))

    return None

def fit_sine_to_ndvi_means(region):
    """Returns a tuple of fitting parameters (std, phase, mean). Fitting method is least squares."""
    avgs = ndvi_monthly_means(region)

    # know that period is 1 year
    period = 12 # months
    freq = 1/period

    guess_mean = 0.235 # Guessed by looking at data
    guess_std = np.std(avgs)
    guess_phase = 0

    t = 2*np.pi*freq * np.arange(12)
    data_guess = guess_std*np.sin(t + guess_phase) + guess_mean

    from scipy.optimize import leastsq
    optimize_func = lambda x: x[0] * np.sin(t + x[1]) + x[2] - avgs

    return leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]

def ndvi_anomalies(region, smooth=6):
    mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)
    path = ndvi_dir + ndvi_fmt

    avgs = ndvi_monthly_means(region)

    fnames = sorted(glob.glob(ndvi_dir + ndvi_fmt.format('*', '*', region) + '.npy'))
    monthlys = np.dstack([np.mean(np.load(fname)[mask]) for fname in fnames]).ravel()

    # Fitting a sine doesn't actually help too much, and is even worse
    # for the eastafrica region.

    # fit_std, fit_phase, fit_mean = fit_sine_to_ndvi_means(region)
    # fit_data = fit_std * np.sin(2 * np.pi * 1/12 * np.arange(12) + fit_phase) + fit_mean

    # anom_fit = monthlys / np.tile(fit_data, 10) - 1
    # anom_fit_smoothed = np.convolve(anom_fit, np.ones((smooth,))/smooth, mode='valid')

    anom_mean = monthlys / np.tile(avgs, 10) - 1
    anom_mean_smoothed = np.convolve(anom_mean, np.ones((smooth,))/smooth, mode='same')

    return anom_mean_smoothed, anom_mean

def plot_ndvi_anomalies(region, smooth=6):
    # mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)
    # path = ndvi_dir + ndvi_fmt

    # avgs = ndvi_monthly_means(region)

    # fnames = sorted(glob.glob(ndvi_dir + ndvi_fmt.format('*', '*', region) + '.npy'))
    # monthlys = np.dstack([np.mean(np.load(fname)[mask]) for fname in fnames]).ravel()

    anom_mean_smoothed, anom_mean = ndvi_anomalies(region, smooth=smooth)

    plt.figure(figsize=(10.5,6))
    # plt.bar(np.arange(len(anom_mean_smoothed)), anom_mean_smoothed, label='Anomaly from dataset mean')
    plt.plot(anom_mean_smoothed, label='Anomaly from dataset mean')
    plt.fill_between(np.arange(len(anom_mean_smoothed)), anom_mean_smoothed, 0, color='olive',
                     where=anom_mean_smoothed>=0, interpolate=True)
    plt.fill_between(np.arange(len(anom_mean_smoothed)), anom_mean_smoothed, 0, color='brown',
                     where=anom_mean_smoothed<0, interpolate=True)
    # plt.bar(np.arange(len(anom_fit_smoothed)), anom_fit_smoothed, label='Difference from fitted sine wave')
    # plt.ylim([0.2, np.max(monthlys) + 0.005])
    plt.xlim(0, len(anom_mean_smoothed))
    plt.xticks(np.arange(0, 10)*12, np.arange(2008, 2018), rotation=45)
    plt.title('{}-month smoothed NDVI anomalies for {} 2008-2018'.format(smooth, region_to_string(region)))
    plt.xlabel('Month')
    plt.ylabel(r'NDVI anomaly $x_i/\mu - 1$')
    plt.legend()
    plt.savefig(figure_dir + 'ndvi_anomalies_{}_smoothed_{}_months.png'.format(region, smooth))

    return None

# Reproduced here from coverage_analysis_functions.py without permission.
def plot_two_with_one_fill_between(plotting_data, corr_labels, x_labels, month_step):
    """plotting_data[0] will be errorbars, plotting_data[1] will be a line"""
    n = len(plotting_data[0][0])
    nind = np.arange(n)
    nx = nind+0.5
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plotting_data[0][0], label=corr_labels[0])
    ax.fill_between(nind, plotting_data[0][0] + plotting_data[0][1],
                    plotting_data[0][0] - plotting_data[0][1], alpha=0.25)
    ax.set_xlim([0, n])
    ax.set_ylim([-0.2, 0.2])
    ax.set_ylabel(r'NDVI$_{\sigma}$')
    ax1 = ax.twinx()
    ax1.plot(plotting_data[1], label=corr_labels[1], color='r')
    ax1.set_ylim([-2.75, 2.75])
    ax1.set_ylabel('SST anomalies ($^{\circ}$C)')

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)

    # now set x ticks
    x_tick_range = len(plotting_data[0][0][::month_step])
    ax.set_xticks(np.linspace(ax.get_xbound()[0], ax.get_xbound()[1], x_tick_range))
    ax.minorticks_off()
    ax1.set_xticks(np.linspace(ax1.get_xbound()[0], ax1.get_xbound()[1], x_tick_range))
    ax1.minorticks_off()
    ax.set_xticklabels([])
    ax1.set_xticklabels(x_labels)
    ax.set_xlabel('Month')
    ax1.set_xlabel('Month')

    return None

def plot_three_with_one_fill_between(plotting_data, corr_labels, x_labels, month_step):
    """plotting_data[0] will be errorbars, plotting_data[[1,2]] will be a line"""
    n = len(plotting_data[0][0])
    nind = np.arange(n)
    nx = nind+0.5
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plotting_data[0][0], label=corr_labels[0])
    ax.fill_between(nind, plotting_data[0][0] + plotting_data[0][1],
                    plotting_data[0][0] - plotting_data[0][1], alpha=0.25)
    ax.set_xlim([0, n])
    ax.set_ylim([-0.2, 0.2])
    ax.set_ylabel(r'NDVI$_{\sigma}$')
    ax1 = ax.twinx()
    ax1.plot(plotting_data[1], label=corr_labels[1], color='r')
    ax1.plot(plotting_data[2], label=corr_labels[2], color='g')
    ax1.set_ylim([-0.2, 0.2])
    ax1.set_ylabel('SST anomalies ($^{\circ}$C)')

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)

    # now set x ticks
    x_tick_range = len(plotting_data[0][0][::month_step])
    ax.set_xticks(np.linspace(ax.get_xbound()[0], ax.get_xbound()[1], x_tick_range))
    ax.minorticks_off()
    ax1.set_xticks(np.linspace(ax1.get_xbound()[0], ax1.get_xbound()[1], x_tick_range))
    ax1.minorticks_off()
    ax.set_xticklabels([])
    ax1.set_xticklabels(x_labels)
    ax.set_xlabel('Month')
    ax1.set_xlabel('Month')

def replace_with_nans(x, idx):
    """Useful for plotting time series data not plotting certain points. x
is the data, idx is a Boolean array for indexing."""
    data = x.copy()
    data[idx] = np.nan
    return data


def plot_three_fb_ds(plotting_data, idxs, corr_labels, x_labels, month_step):
    """plotting_data[0] will be fill between, plotting_data[[1,2]] will be a line.
    plotting_data[1] should be ONI and will be plotted solid at points defined by idxs and dashed elswhere. idxs[0] should be El Nino indices, idxs[1] La Nina. corr_labels should now include entries for ONI (neutral) and ONI (event)."""
    n = len(plotting_data[0][0])
    nind = np.arange(n)
    nx = nind+0.5
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plotting_data[0][0], label=corr_labels[0])
    ax.fill_between(nind, plotting_data[0][0] + plotting_data[0][1],
                    plotting_data[0][0] - plotting_data[0][1], alpha=0.25)
    ax.set_xlim([0, n])
    ax.set_ylim([-0.2, 0.2])
    ax.set_ylabel(r'NDVI$_{\sigma}$')
    ax1 = ax.twinx()
    # plot all th
    neu = replace_with_nans(plotting_data[1], np.logical_or(idxs[0], idxs[1]))
    enso = replace_with_nans(plotting_data[1], np.logical_and(~idxs[0], ~idxs[1]))
    # plot dashed curve underneath solid curve for continuity 
    ax1.plot(plotting_data[1], color='r', label=corr_labels[1],
             linestyle='dashed', alpha=0.75)
    ax1.plot(enso, label=corr_labels[2], color='r', linestyle='solid')
#    ax1.plot(neu, label=corr_labels[2], color='r', linestyle='dashed')
    ax1.plot(plotting_data[2], label=corr_labels[3], color='g')
    ax1.set_ylim([-2.75, 2.75])
    ax1.set_ylabel('SSTA ($^{\circ}$C)')

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)

    # now set x ticks
    x_tick_range = len(plotting_data[0][0][::month_step])
    ax.set_xticks(np.linspace(ax.get_xbound()[0], ax.get_xbound()[1], x_tick_range))
    ax.minorticks_off()
    ax1.set_xticks(np.linspace(ax1.get_xbound()[0], ax1.get_xbound()[1], x_tick_range))
    ax1.minorticks_off()
    ax.set_xticklabels([])
    ax1.set_xticklabels(x_labels)
    ax.set_xlabel('Month')
    ax1.set_xlabel('Month')


def plot_ndvi_with_dmi(region, smooth=6):
    from coverage_analysis_functions import (nino_range,
                                             consecutive_anomalies,
                                             month_and_year_labels,
                                             load_swio, narrow_swio,
                                             swio_three_monthly_means,
                                             subtract_month, add_month)

    start_date = datetime.datetime.strptime('20081','%Y%m')
    end_date = datetime.datetime.strptime('20181','%Y%m')

    dmi_datetime, dmi_anoms = np.array(load_swio('dmi.nc', 'DMI'))
    narrowed_dmi_datetime = dmi_datetime[((dmi_datetime > start_date) & (dmi_datetime < end_date))]
    narrowed_dmi = dmi_anoms[((dmi_datetime > start_date) & (dmi_datetime < end_date))]
    # DMI comes with multiple values per month, so the following takes
    # all values for a particular month and averages them. List
    # comprehensions for the win. This would not work as expected if
    # there were some months without data, but hopefully that isn't
    # the case here.
    narrowed_dmi = np.array([
        np.mean(narrowed_dmi[[((d.year == year) & (d.month == month)) for d in narrowed_dmi_datetime]])
        for year in np.arange(2008, 2018)
        for month in np.arange(1, 13)])
    # dmi_tmm = swio_three_monthly_means(dmi_datetime, dmi_anoms)
    dmi_tmm = np.convolve(narrowed_dmi, np.ones((3,))/3, mode='same')

    corr_labels = ['NDVI', 'DMI']
    plotting_data = [[ndvi_anomalies(region, smooth=smooth)[0],
                      np.zeros_like(ndvi_anomalies(region, smooth=smooth)[0])],
                     dmi_tmm]

    month_step = 4
    x_labels = month_and_year_labels(np.tile([1, 5, 9], 10), np.arange(2008, 2018), 4)

    plot_two_with_one_fill_between(plotting_data, corr_labels, x_labels, month_step)

    plt.title('{}'.format(region_to_string(region)))
    plt.axhline(linewidth=0.75, color='k')
    plt.axhline(y=0.5, linewidth=0.75, color='k', linestyle='dashed')
    plt.axhline(y=-0.5, linewidth=0.75, color='k', linestyle='dashed')
    plt.savefig(figure_dir + 'ndvi_dmi_{}_smoothed_{}.png'.format(region, smooth))

    return None

def ndvi_uncertainty(month, region):
    fnames = sorted(glob.glob(ndvi_dir + ndvi_fmt.format('*', month, region) + '.npy'))
    mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)
    ndvi = np.dstack([np.load(fname)[mask] for fname in fnames]).ravel()

    return abs(np.diff(np.percentile(ndvi, [75, 25]))[0]/2)


def plot_ndvi_with_oni_io(region, smooth=5):
    from coverage_analysis_functions import (nino_range,
                                             consecutive_anomalies,
                                             month_and_year_labels,
                                             load_swio,
                                             swio_three_monthly_means,
                                             narrow_swio)
    # Load ONI
    start = datetime.date(2008, 1, 1)
    end = datetime.date(2018, 1, 1)
    # Columns are: year, month, anomaly
    oni = nino_range('detrend.nino34.ascii.txt', start, end)

    if region == 'capetown':
        io_datetime, io_anoms = load_swio('swio.nc', 'SWIO')
        io_label = 'SWIO'
    else:
        io_datetime, io_anoms = load_swio('wtio.nc', 'WTIO')
        io_label = 'WTIO'

    io_tmm = swio_three_monthly_means(io_datetime, io_anoms) 
    
    # Find 3-month anomaly excess
    oni_tmm = np.convolve(oni[:, 2], np.ones((3,))/3, mode='same')
    en = np.zeros_like(oni_tmm, dtype=bool)
    ln = np.zeros_like(oni_tmm, dtype=bool)
    for i in np.arange(len(oni_tmm)):
        if np.all(oni_tmm[i:(i+5)] >= 0.5):  en[i:(i+5)] = True
        if np.all(oni_tmm[i:(i+5)] <= -0.5): ln[i:(i+5)] = True

    # Uncertainty is simply IQR
    errors = np.tile([ndvi_uncertainty('{:02d}'.format(m), region)
                      for m in np.arange(1,13)],
                     10)

    corr_labels = ['NDVI', 'ONI (neutral)', 'ONI (event)', io_label]
    # TODO Add error data here for NDVI, using IQR.
    plotting_data = [[ndvi_anomalies(region, smooth=smooth)[0], errors],
                     oni_tmm, io_tmm[2]]
    month_step = 4
    x_labels = month_and_year_labels(np.tile([1, 5, 9], 10), np.arange(2008, 2018), 4)    
    plot_three_fb_ds(plotting_data, [en, ln], corr_labels, x_labels, month_step)

    plt.title('{} NDVI smoothed {}-months'.format(region_to_string(region), smooth))
    plt.axhline(linewidth=0.75, color='k')
    plt.axhline(y=0.5, linewidth=0.75, color='k', linestyle='dashed')
    plt.axhline(y=-0.5, linewidth=0.75, color='k', linestyle='dashed')
    plt.savefig(figure_dir + 'ndvi_oni_io_{}_smoothed_{}.pdf'.format(region, smooth))

    return None

def plot_ndvi_distributions(region):
    fig, axs = plt.subplots(4, 3, sharey=True, figsize=(8.27,11.69))
    for ax, month in zip(axs.ravel(), np.arange(1, 13)):
        m = '{:02d}'.format(month)
        mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool);
        fnames = sorted(glob.glob(ndvi_dir + ndvi_fmt.format('*', m, region) + '.npy'));
        monthlys = np.dstack([np.mean(np.load(fname)[mask]) for fname in fnames]).ravel();
        ax.set_title(m)
        ax.vlines(np.mean(monthlys), 0, 1, transform=ax.get_xaxis_transform(),
                  color='r', label='mean')
        ax.vlines(np.median(monthlys), 0, 1, transform=ax.get_xaxis_transform(),
                  color='k', label='median')
        ax.hist(monthlys, bins=6); 

    axs[3, 0].legend()
    axs[3, 0].set_xlabel('NDVI')
    axs[3, 0].set_ylabel('Count')
    plt.suptitle('NDVI distribution for months in range 2008-2017')
    plt.savefig(figure_dir + 'ndvi_distributions_{}.pdf'.format(region))

    return None

def do_analysis():
    regions = ('capetown', 'eastafrica')
    smoothings = (3, 5, 6, 12)

    for region in regions:
        plot_ndvi_monthly_and_means(region)
        plt.close()

        for smooth in smoothings:
            plot_ndvi_anomalies(region, smooth=smooth)
            plot_ndvi_with_oni(region, smooth=smooth)
            plot_ndvi_with_dmi(region, smooth=smooth)
            plt.close()

    return None

# e.g usage
# for year in np.arange(2013, 2018):
#     for region in ('capetown', 'eastafrica'):
#         print('Producing NDVI data for {} {}'.format(region, year))
#         ndvi_for_year_and_region(year, region)
