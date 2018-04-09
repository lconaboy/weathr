import datetime
import calendar
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def add_month(start):
    date = start
    delta = datetime.timedelta(days=31)

    while start.month == date.month:
        date += delta

    new_date = datetime.datetime.strptime('{}{}1'.format(date.year,
                                                          date.month), '%Y%m%d')
    
    return new_date

def load_month(date, region):
    """x[0] is the cloud mask, x[1] is [mean cloud fraction, standard deviation]"""
    return np.load('{}_{}_multiband_{}_cloud.npy'.format(date.month, date.year, region))


def to_idx(nino, arg):
    years = nino[:, 0][arg]
    years_idx = (years-2009.).astype(int).ravel()

    months = nino[:, 1][arg]
    months_idx = (months-1.).astype(int).ravel()

    # pair years and months
    idxs = list(zip(months_idx, years_idx))

    # sort by months
    idxs = sorted(idxs, key=lambda x: x[0])

    print(months)
    print(years)
    
    return idxs


def consecutive_anomalies(nino):
    """Takes the Nino 3.4 anomaly data (text file) where the first column
    is YR data, second is M data and final is ANOM data. Converts to
    ONI anomalies, which are where 3-month running means of the Nino
    3.4 SST anomalies exceed +/- 0.5C for 5 consecutive
    months. Returns an array where the first column is the year,
    second column is the ccentral month (i.e. 1 = DJF), third column
    is EN and fourth column is LN.

    """
    
    # first calculate three-monthly means for nino
    nino3 = [sum(nino[i-1:i+2, 2])/3 for i in range(1, len(nino)-1)]
    # now return the value of the middle month in the mean as an integer
    season3 = [np.int(nino[i, 1]) for i in range(1, len(nino)-1)]
    # finally return the year as an integer
    years = [np.int(nino[i, 0]) for i in range(1, len(nino)-1)]

    # the idea for counting consecutive anomalies is that we count
    # along one direction, then back again along the reverse direction
    # and sum the two counts. any positions in the resulting array
    # that are >= 6 then correspond to points where there are at least
    # 5 consecutive anomalies.
    
    # first find anomalies > 0.5
    en3 = np.array(nino3) >= 0.5
    en_anom = np.zeros(len(en3))
    en_anom_rev = np.zeros(len(en3))
    # count from 1 to len()
    for i in range(1, len(en3)):
        en_anom[i] = (en_anom[i-1] + en3[i])*en3[i]
    # now count from len() to 1
    for i in reversed(range(0, len(en3)-1)):
        en_anom_rev[i] = (en_anom_rev[i+1] + en3[i])*en3[i]

    en_idxs = (en_anom + en_anom_rev) >= 6  # and voila!
        
    # as above but for anomalies < -0.5
    ln3 = np.array(nino3) <= -0.5
    ln_anom = np.zeros(len(ln3))
    ln_anom_rev = np.zeros(len(ln3))
    for i in range(1, len(ln3)):
        ln_anom[i] = (ln_anom[i-1] + ln3[i])*ln3[i]
    for i in reversed(range(0, len(ln3)-1)):
        ln_anom_rev[i] = (ln_anom_rev[i+1] + ln3[i])*ln3[i]

    ln_idxs = (ln_anom + ln_anom_rev) >= 6

    return [years, season3, en_idxs, ln_idxs]


start = datetime.datetime.strptime('200901','%Y%M')
end = datetime.datetime.strptime('201801', '%Y%M')
region = 'eastafrica'

# now to calculate means for EN/LN/neither years from table
# load el nino years
nino = np.loadtxt('detrend.nino34.ascii.txt', skiprows=1, usecols=(0, 1, 4))
# narrow to our range
start = np.argwhere((nino[:, 0] == 2009.)*(nino[:, 1] == 1.))
end = np.argwhere((nino[:, 0] == 2018.)*(nino[:, 1] == 1.))
nino = nino[np.arange(start, end), :]

    
# indices for El Nino, La Nina and neither months
en = np.argwhere(nino[:, 2] >= 0.5).ravel()
ln = np.argwhere(nino[:, 2] <= -0.5).ravel()
nei = np.argwhere((nino[:, 2] > -0.5)*(nino[:, 2] < 0.5)).ravel()
args = [en, ln, nei]

# # now separate the cloud data into different years
# sep = np.zeros(shape=(12, 3)) 
# for i, arg in enumerate(args):
#     idxs = to_idx(nino, arg)
#     for n in range(0,12):
#         sep[n, i] = np.mean([cloud[x] for x in idxs if x[0] == n])

# # now plot anomalies
# plt.figure()
# for i in range(0,3):
#     plt.plot(sep[:, i] - cloud_mean)
#     plt.legend(['EN', 'LN', "neither"])
#     plt.title('Monthly - Cape Town')
#     plt.ylabel(r'$c - \mu$')
#     labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
#     plt.xticks(np.arange(0,12))
#     ax = plt.gca()
#     ax.set_xticklabels(labels)
#     plt.tight_layout()

# plt.figure()
# for i in range(0,3):
#     plt.plot((sep[:, i] - cloud_mean)/cloud_std)
#     plt.legend(['EN', 'LN', "neither"])
#     plt.title('Monthly - Cape Town (Normalised)')
#     plt.ylabel(r'$(c - \mu)/\sigma$')
#     labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
#     plt.xticks(np.arange(0,12))
#     ax = plt.gca()
#     ax.set_xticklabels(labels)
#     plt.tight_layout()
#     plt.show()


### new ###
# TODO: convert to loops/functions
# pick out consecutive anomalies
anoms = np.array(consecutive_anomalies(nino))

# now pick out seasonal anomalies
djf = anoms[:, anoms[1] == 1]
jja = anoms[:, anoms[1] == 7]

# now pick out EN/LN/neutral
en_djf = djf[:, djf[2] == 1]
ln_djf = djf[:, djf[3] == 1]
neu_djf = djf[:, djf[2]+djf[3] == 0]

seasons = [djf, en_djf, ln_djf, neu_djf]

def load_month_from_anoms(month, year, region):
    """x[0] is the cloud mask, x[1] is [mean cloud fraction, standard deviation]"""
    return np.load('{}_{}_multiband_{}_cloud.npy'.format(month, year, region))

def load_seasonal(seas, region, frac):
    """frac=0 to load cloud masks, frac=1 to load cloud fraction data"""
    return [load_month_from_anoms(seas[1][i], seas[0][i], region)[frac]
                     for i in range(0, len(seas[0]))]

cloud_masks = [np.mean(load_seasonal(s, region, 0), axis=0) for s in seasons]
titles = ['ALL', 'EN', 'LN', 'NEUTRAL']

### cloud masks
fig, axes = plt.subplots(2, 2)
for idx, ax in enumerate(axes.flat):
    im = ax.imshow(cloud_masks[idx])
    ax.set_title(titles[idx])
    ax.axis('off')
fig.colorbar(im, ax=axes.ravel().tolist())
# # plt.show()

cloud_frac = [load_seasonal(s, region, 1) for s in seasons]
seas_frac = np.zeros(shape=(len(cloud_frac), 2))

### seasonal difference plot
# plt.figure()
# for idx, frac in enumerate(cloud_frac):
#     seas_frac[idx, 0] = np.mean(frac[0])
#     seas_frac[idx, 1] = np.sqrt(np.mean(frac[1])**2 + np.std(frac[0])**2)
# plt.errorbar(-1, seas_frac[1, 0], seas_frac[1, 1])
# plt.errorbar(0, seas_frac[2, 0], seas_frac[2, 1])
# plt.errorbar(1, seas_frac[3, 0], seas_frac[2, 1])
# plt.legend(titles[1::])


total_mean_frac = np.array([np.mean([load_month_from_anoms(month, year, region)[1][0]
                            for year in range(2009, 2018)]) for month in range(1, 13)])
# total_mean_frac = np.ones(12)*np.mean(total_mean_frac)  # uncomment for static mean

def harmonic_anomalies(year, region):
    """This function calculates anomalies in cloud fraction over 6 months
    before and after an ENSO event, as in Nicholson & Kim."""
    
    en_diff = np.zeros(24)
    # look at the year prior (Jul-Dec)
    en_diff[0:6] = np.array([load_month_from_anoms(month, year-1, region)[1][0]
                             for month in range(7, 13)]) - total_mean_frac[6:12]
    # look at the year of EN (all year)
    en_diff[6:18] = np.array([load_month_from_anoms(month, year, region)[1][0]
                              for month in range(1, 13)]) - total_mean_frac
    # finally look at the following year (Jan-Jun)
    en_diff[18:24] = np.array([load_month_from_anoms(month, year+1, region)[1][0]
                         for month in range(1, 7)]) - total_mean_frac[0:6]

    return en_diff


def sine_wave(x, a, b):
    return a*np.sin(b*x)

en_diff = harmonic_anomalies(2015, region)
ln_diff = harmonic_anomalies(2011, region)
neu_diff = harmonic_anomalies(2013, region)

xvals = np.arange(0, len(en_diff))
labels = ['J', 'A', 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J',
          'J', 'A', 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J']
p0 = [0.05,1/4]  # initial guess

ylims = (-0.2, 0.2)  # force all axes to have same y limits

plt.figure(figsize=(9, 6))
plt.subplot(2, 2, 1)
en_params, params_covariance = optimize.curve_fit(sine_wave, xvals, en_diff, p0)
plt.step(xvals, en_diff, color='k')
plt.plot(xvals, sine_wave(xvals, en_params[0], en_params[1]), color='k')
plt.ylabel(r'$x-\mu$')
plt.xlabel('Month')
plt.title('2015 El Nino (Eastern Africa)')
plt.axvline(6, color='k', linestyle='dashed')
plt.axvline(18, color='k', linestyle='dashed')
plt.ylim(ylim)
plt.xticks(np.arange(24))
ax = plt.gca()
ax.set_xticklabels(labels)
ax.tick_params(axis='x',which='minor',bottom='off', top='off')

plt.subplot(2, 2, 2)
ln_params, params_covariance = optimize.curve_fit(sine_wave, xvals, ln_diff, p0)
plt.step(xvals, ln_diff, color='k')
plt.plot(xvals, sine_wave(xvals, ln_params[0], ln_params[1]), color='k')
plt.ylabel(r'$x-\mu$')
plt.xlabel('Month')
plt.title('2011 La Nina (Eastern Africa)')
plt.axvline(6, color='k', linestyle='dashed')
plt.axvline(18, color='k', linestyle='dashed')
plt.ylim(ylims)
plt.xticks(np.arange(24))
ax = plt.gca()
ax.set_xticklabels(labels)
ax.tick_params(axis='x',which='minor',bottom='off', top='off')

plt.subplot(2, 2, 3)
neu_params, params_covariance = optimize.curve_fit(sine_wave, xvals, neu_diff, p0)
plt.step(xvals, neu_diff, color='k')
plt.plot(xvals, sine_wave(xvals, neu_params[0], neu_params[1]), color='k')
plt.ylabel(r'$x-\mu$')
plt.xlabel('Month')
plt.title('2013 Neutral (Eastern Africa)')
plt.axvline(6, color='k', linestyle='dashed')
plt.axvline(18, color='k', linestyle='dashed')
plt.ylim(ylim)
plt.xticks(np.arange(24))
ax = plt.gca()
ax.set_xticklabels(labels)
ax.tick_params(axis='x',which='minor',bottom='off', top='off')

plt.subplot(2, 2, 4)
xvals1 = np.arange(12)
p0 = [0.2,1/2]
plt.step(xvals1, total_mean_frac, color='k')
plt.ylabel(r'$\mu$')
plt.xlabel('Month')
plt.title('Mean (Eastern Africa)')
plt.xticks(xvals1)
ax = plt.gca()
ax.set_xticklabels(labels[6:18])
ax.tick_params(axis='x',which='minor',bottom='off', top='off')
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect((x1-x0)/(y1-y0))
plt.tight_layout()

plt.savefig('ea_harmonicdiffs_vmean')
plt.show()
