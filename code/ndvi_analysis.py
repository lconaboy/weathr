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

figure_dir = 'results/figures/'

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
    ax1.set_title('Uncalibrated NDVI for {} {}, {}'.format(region.capitalize(), month, year))
    ax1.axis('off')
    plot1 = ax1.imshow(ndvi_uncal, cmap=cmap, vmin=-np.max(ndvi_uncal), vmax=np.max(ndvi_uncal))
    ax2.set_title('Calibrated NDVI for {} {}, {}'.format(region.capitalize(), month, year))
    ax2.axis('off')
    plot2 = ax2.imshow(ndvi_cal, cmap=cmap, vmin=-np.max(ndvi_cal), vmax=np.max(ndvi_cal))
    f.subplots_adjust(right=0.8)
    cb_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(plot2, cax=cb_ax)
    plt.savefig(figure_dir + 'calibration_comparision.pdf')

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
    plt.title('NDVI for {} 2008-2018'.format(region))
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

def plot_ndvi_anomalies(region):
    mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)
    path = ndvi_dir + ndvi_fmt
    smooth = 6

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
    anom_mean_smoothed = np.convolve(anom_mean, np.ones((smooth,))/smooth, mode='valid')

    plt.figure(figsize=(10.5,6))
    plt.bar(np.arange(len(anom_mean_smoothed)), anom_mean_smoothed, label='Anomaly from dataset mean')
    # plt.bar(np.arange(len(anom_fit_smoothed)), anom_fit_smoothed, label='Difference from fitted sine wave')
    # plt.ylim([0.2, np.max(monthlys) + 0.005])
    plt.xlim(0, len(monthlys))
    plt.xticks(np.arange(0, 9)*12, np.arange(2009, 2018), rotation=45)
    plt.title('{}-month smoothed NDVI anomalies for {} 2008-2018'.format(smooth, region))
    plt.xlabel('Month')
    plt.ylabel(r'NDVI anomaly $x_i/\mu - 1$')
    plt.legend()
    plt.savefig(figure_dir + 'ndvi_anomalies_{}_smoothed_{}_months.png'.format(region, smooth))

# e.g usage
# for year in np.arange(2013, 2018):
#     for region in ('capetown', 'eastafrica'):
#         print('Producing NDVI data for {} {}'.format(region, year))
#         ndvi_for_year_and_region(year, region)
