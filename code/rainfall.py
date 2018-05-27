import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from coverage_analysis_functions import *
from util import *

# this is a truly awful bit of code. to change plots between regions
# the following things must be adjusted:
# 1. region
# 2. fname (rainfall)
# 3. p0 (rainfall) ([30, 1/2, 50] for SA, [15, 1/2, 30] for EA)
# 4. ax2.set_ylim([min, max]) ([0, 80] for SA, [17, 27] for EA)
# 5. legend loc (9 for SA, 2 for EA)

T = 24  # period in months
def sine(x, a, c, T=T):
    return a*np.sin((2*np.pi/T)*x) + c

def harmonic_anomalies(data, mean_rf, year):
    """This function calculates anomalies in rainfall over 6 months
    before and after an ENSO event, as in Nicholson & Kim."""
    
    en_diff = np.zeros(24)
    # look at the year prior (Jul-Dec)
    idxs = np.argwhere((data[:, 1] == year-1.)*(data[:, 2] >= 7.))
    en_diff[0:6] =  data[idxs, 0].ravel() - mean_rf[6:12]
    print('done before')
    # look at the year of EN (all year)
    idxs = np.argwhere(data[:, 1] == year)
    en_diff[6:18] =  data[idxs, 0].ravel() - mean_rf
    print('done during')
    # finally look at the following year (Jan-Jun)
    idxs = np.argwhere((data[:, 1] == year+1.)*(data[:, 2] < 7.))
    en_diff[18:24] =  data[idxs, 0].ravel() - mean_rf[0:6]
    print('done after')
    
    return en_diff

# CLOUD FRACTION
region = 'eastafrica'
start = datetime.datetime.strptime('20081','%Y%m')
end = datetime.datetime.strptime('201712','%Y%m')
cf = load_cloud_fraction_period(start, end, region)
cf_means = yearly_mean(cf)
cf_medians = yearly_median(cf)

# RAINFALL
# fnames = ['sa_rainfall_1991_2015.txt', 'sa_rainfall_1901_2015.txt',
#          'eth_rainfall_1991_2015.txt', 'eth_rainfall_1901_2015.txt']

fname = 'eth_rainfall_1991_2015.txt'

data = np.loadtxt(fname, skiprows=1, usecols=(0,1,2))
# calculate mean and sigma
mean_rf = np.zeros(12)
sig_rf = np.zeros(12)
for i in range(0, 12):
    mean_rf[i] = np.mean(data[i::12, 0])
    sig_rf[i] = np.std(data[i::12, 0])

# PLOTTING
fig, ax1 = plt.subplots(figsize=(4, 4))
x = np.arange(12)
ax1.set_title(region_to_string(region))
# plot cloud fraction on left axes
ax1.errorbar(x, cf_medians[0], cf_medians[1], fmt='x', color='b', label='CF')
p0 = [0.4, 0.2]
params, params_cov = optimize.curve_fit(sine, x, cf_medians[0], p0, cf_medians[1])
ax1.plot(x, sine(x, params[0], params[1], T), color='b')
ax1.set_ylabel('CF')
ax1.set_ylim([0, 0.8])
# plot the mean rainfall, fitted with a sine wave
ax2 = ax1.twinx()
labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
p0 = [15, 1/2, 30]
params, params_cov = optimize.curve_fit(sine, x, mean_rf, p0, sig_rf)

#ax2.errorbar(x+0.2, mean_rf, sig_rf, fmt='rx', label='Rainfall')
ax2.plot(x, sine(x, params[0], params[1], params[2]), color='r')
ax2.plot(x, mean_rf, 'rx', label='Rainfall')
ax2.set_xticks(x)
ax2.set_ylabel('Rainfall (mm)')
ax2.set_xticklabels(labels)
ax2.set_ylim([17, 27])
ax1.set_xlabel('Month')
ax2.set_xlabel('Month')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc=2)
plt.tight_layout()
plt.savefig(figure_dir+'cf_rainfall_{}'.format(region))

# HARMONIC ANOMALIES

# ln_diff = harmonic_anomalies(data, mean_rf, 2011.)
# neu_diff = harmonic_anomalies(data, mean_rf, 2013.)

# xvals = np.arange(0, len(en_diff))
# labels = ['J', 'A', 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J',
#           'J', 'A', 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J']
# p0 = [30, 20]  # initial guess

# ylims = (-50, 50)  # force all axes to have same y limits

# plt.figure(figsize=(8, 6))
# plt.suptitle('Rainfall Anomalies (Southern Africa)')
# plt.subplot(2, 1, 1)
# ln_params, params_covariance = optimize.curve_fit(sine, xvals, ln_diff, p0)
# plt.step(xvals, ln_diff, color='k')
# plt.plot(xvals, sine(xvals, ln_params[0], ln_params[1]), color='k')
# plt.ylabel(r'$x-\mu$ (mm)')
# plt.title('2011 La Nina')
# plt.axvline(6, color='k', linestyle='dashed')
# plt.axvline(18, color='k', linestyle='dashed')
# plt.ylim(ylims)
# plt.xticks(np.arange(24))
# ax = plt.gca()
# ax.set_xticklabels(labels)
# ax.tick_params(axis='x',which='minor',bottom='off', top='off')

# plt.subplot(2, 1, 2)
# neu_params, params_covariance = optimize.curve_fit(sine, xvals, neu_diff, p0)
# plt.step(xvals, neu_diff, color='k')
# plt.plot(xvals, sine(xvals, neu_params[0], neu_params[1]), color='k')
# plt.ylabel(r'$x-\mu$ (mm)')
# plt.xlabel('Month')
# plt.title('2013 Neutral')
# plt.axvline(6, color='k', linestyle='dashed')
# plt.axvline(18, color='k', linestyle='dashed')
# plt.ylim(ylims)
# plt.xticks(np.arange(24))
# ax = plt.gca()
# ax.set_xticklabels(labels)
# ax.tick_params(axis='x',which='minor',bottom='off', top='off')

# #plt.savefig(r'sa_$T={}$_rfanom'.format(T))
# #plt.show()

# # try correlating
# diffs = np.load('cf_anom_2011_2013_capetown.npy')

# corr_ln = np.corrcoef(diffs[0], ln_diff)
# ccorr_ln = np.correlate(diffs[0], ln_diff)
# corr_neu = np.corrcoef(diffs[1], neu_diff)
# ccorr_neu = np.correlate(diffs[1], neu_diff)
