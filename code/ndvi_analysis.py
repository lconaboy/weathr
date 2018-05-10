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
    # Try load NDVI data
    ndvi_uncal = np.load(ndvi_dir + ndvi_fmt.format(year, month, region) + "_uncalibrated.npy")
    ndvi_cal   = np.load(ndvi_dir + ndvi_fmt.format(year, month, region) + ".npy")

    import matplotlib.pyplot as plt
    import matplotlib.colors as clr
    cmap = clr.LinearSegmentedColormap.from_list('', ['red', 'white', 'darkgreen'])
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,4.5), dpi=150)
    ax1.set_title('Uncalibrated NDVI for {} {}, {}'.format(region_to_string(region), month, year))
    ax1.axis('off')
    plot1 = ax1.imshow(ndvi_uncal, cmap=cmap, vmin=-np.max(ndvi_uncal), vmax=np.max(ndvi_uncal))
    ax2.set_title('Calibrated NDVI for {} {}, {}'.format(region_to_string(region), month, year))
    ax2.axis('off')
    plot2 = ax2.imshow(ndvi_cal, cmap=cmap, vmin=-np.max(ndvi_cal), vmax=np.max(ndvi_cal))
    f.subplots_adjust(right=0.8)
    cb_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(plot2, cax=cb_ax)
    plt.savefig(figure_dir + 'ndvi_calibration_comparision.pdf')

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
    plt.bar(np.arange(len(monthlys)), monthlys, label='Monthly NDVI')
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
    plt.bar(np.arange(len(anom_mean_smoothed)), anom_mean_smoothed, label='Anomaly from dataset mean')
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


def plot_ndvi_with_oni(region, smooth=6):
    from coverage_analysis_functions import (nino_range,
                                             consecutive_anomalies,
                                             month_and_year_labels)
    # Load ONI
    start = datetime.date(2008, 1, 1)
    end = datetime.date(2018, 1, 1)
    # Columns are: year, month, anomaly
    nino = nino_range('detrend.nino34.ascii.txt', start, end)
    # Find 3-month anomaly excess
    tmm = np.convolve(nino[:, 2], np.ones((3,))/3, mode='same')
    en = np.zeros_like(tmm, dtype=bool)
    ln = np.zeros_like(tmm, dtype=bool)
    for i in np.arange(len(tmm)):
        if np.all(tmm[i:(i+5)] >= 0.5):  en[i:(i+5)] = True
        if np.all(tmm[i:(i+5)] <= -0.5): ln[i:(i+5)] = True

    corr_labels = ['NDVI', 'ONI']
    plotting_data = [[ndvi_anomalies(region, smooth=smooth)[0],
                      np.zeros_like(ndvi_anomalies(region, smooth=smooth)[0])],
                     tmm]
    month_step = 4
    x_labels = month_and_year_labels(np.tile([1, 5, 9], 10), np.arange(2008, 2018), 4)    
    plot_two_with_one_fill_between(plotting_data, corr_labels, x_labels, month_step)

    plt.title('{}'.format(region_to_string(region)))
    plt.axhline(linewidth=0.75, color='k')
    plt.axhline(y=0.5, linewidth=0.75, color='k', linestyle='dashed')
    plt.axhline(y=-0.5, linewidth=0.75, color='k', linestyle='dashed')
    plt.savefig(figure_dir + 'ndvi_oni_{}_smoothed_{}.png'.format(region, smooth))
    plt.show()

    return None


def do_analysis():
    regions = ('capetown', 'eastafrica')
    smoothings = (3, 6, 12)

    for region in regions:
        plot_ndvi_monthly_and_means(region)
        plt.close()

        for smooth in smoothings:
            plot_ndvi_anomalies(region, smooth=smooth)
            plot_ndvi_with_oni(region, smooth=smooth)
            plt.close()

    return None

# e.g usage
# for year in np.arange(2013, 2018):
#     for region in ('capetown', 'eastafrica'):
#         print('Producing NDVI data for {} {}'.format(region, year))
#         ndvi_for_year_and_region(year, region)
