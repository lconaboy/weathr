import datetime
import calendar
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from netCDF4 import Dataset


"""Ok so here it is. All of the stuff in functions is very useful and
is being used in other scripts. The stuff commented out below that is
maybe useful, maybe not but I want to hold on to it, just in case."""

def add_month(start):
    date = start
    delta = datetime.timedelta(days=2)

    while start.month == date.month:
        date += delta

    new_date = datetime.datetime.strptime('{}{}1'.format(date.year,
                                                          date.month), '%Y%m%d')
    
    return new_date


def subtract_month(start):
    date = start
    delta = datetime.timedelta(days=2)

    while start.month == date.month:
        date -= delta

    new_date = datetime.datetime.strptime('{}{}1'.format(date.year,
                                                          date.month), '%Y%m%d')
    
    return new_date


def load_month(date, region):
    """x[0] is the cloud mask, x[1] is [mean cloud fraction, standard deviation]"""
    return np.load('data/cloud/{}_{}_multiband_{}_cloud.npy'.format(date.month,
                                                                    date.year, region))


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


def consecutive_anomalies(nino, start, end):
    """Takes the Nino 3.4 anomaly data (text file) where the first column
    is YR data, second is M data and final is ANOM data. Converts to
    ONI anomalies, which are where 3-month running means of the Nino
    3.4 SST anomalies exceed +/- 0.5C for 5 consecutive
    months. Returns an array where the first column is the year,
    second column is the ccentral month (i.e. 1 = DJF), third column
    is EN and fourth column is LN. Narrows to smoothed data range
    using start and end as datetime objects.

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

    # doesn't actually need to be narrowed
    # # narrow to smoothed range
    # a = np.argwhere((np.array(years) == start.year)&
    #                 (np.array(season3) == start.month)).ravel()
    # print(a)
    # b = np.argwhere((np.array(years) == end.year)&
    #                 (np.array(season3) == end.month)).ravel()
    # span = slice(int(a), int(b))
    # print(b)
    return [years, season3, en_idxs, ln_idxs, nino3]


def load_month_from_anoms(month, year, region):
    """x[0] is the cloud mask, x[1] is [mean cloud fraction, standard deviation]"""
    return np.load('data/cloud/{}_{}_multiband_{}_cloud.npy'.format(month,
                                                                    year, region))


def load_seasonal(seas, region, frac):
    """frac=0 to load cloud masks, frac=1 to load cloud fraction data"""
    return [load_month_from_anoms(seas[1][i], seas[0][i], region)[frac]
                     for i in range(0, len(seas[0]))]


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


def three_monthly_means(start, end, region):
    """This function calculates three monthly means for data between start
and end. Returns a list where the first and second elements are year
and central month (i.e. 1 = DJF), the third element is the mean cloud
fraction value across those months and the fourth element is the min
and max of the cloud fraction values, as a proxy for the error."""
    date = start
    yr = []
    mn = []
    val = []
    rng = []

    while date <= end:
        tmp = np.zeros(3)  # load the three months
        tmp[0] = load_month(subtract_month(date), region)[1][0]
        tmp[1] = load_month(date, region)[1][0]
        tmp[2] = load_month(add_month(date), region)[1][0]
        # return the middle month of the three month mean, the mean, and
        # the range (proxy for error)
        yr.append(date.year)
        mn.append(date.month)
        val.append(np.mean(tmp))
        rng.append([min(tmp), max(tmp)])
        
        date = add_month(date)  # step date forward

    return [yr, mn, val, rng]


def nino_range(fname, start, end):
    """Takes the fname for the Nino SST anomalies data and reduces it to
the specified range. All dates must be floats, not int."""
    # now to calculate means for EN/LN/neither years from table
    # load el nino years
    nino = np.loadtxt(fname, skiprows=1, usecols=(0, 1, 4))
    # narrow to our range
    start = np.argwhere((nino[:, 0] == start.year)*(nino[:, 1] == start.month))
    end = np.argwhere((nino[:, 0] == end.year)*(nino[:, 1] == end.month))
    nino = nino[np.arange(start, end), :]

    return nino


def stack_tmm_by_event(tmm, anoms):
    """Stack all three month periods with the same event (i.e. all DJF
EN)"""
    vals = np.zeros(shape=(12, 3))
    rngs = np.zeros(shape=(12, 3))
    if tmm[0] == anoms[0] and tmm[1] == anoms[1]:  # first check the dates match
        for m in range(1, 13):
            # find the indices for the current month
            idxs = np.array(tmm[1]) == m
            # match these indices with EN, LN or neither
            en_idxs = idxs*anoms[2]
            ln_idxs = idxs*anoms[3]
            ne_idxs = idxs*(~en_idxs)*(~ln_idxs)
            all_idxs = [en_idxs, ln_idxs, ne_idxs]
            # now determine whether EN, LN or neither and calculate means
            # and ranges
            for i, idx in enumerate(all_idxs):
                if any(idx):
                    vals[m-1, i] = np.mean(np.array(tmm[2])[idx])
                    # sort the ranges
                    r = np.sort(np.array(tmm[3])[idx].ravel())
                    # if r is big enough, we can account for outliers by
                    # taking the second smallest and second largest to be the
                    # range, however if len(r) is <= 3 then we must use min
                    # and max
                    # then convert range to uncertainty
                    if len(r) > 5:
                        rngs[m-1, i] = (r[-3] - r[2])/2
                    elif len(r) > 3:
                        rngs[m-1, i] = (r[-2] - r[1])/2
                    else:
                        rngs[m-1, i] = (max(r) - min(r))/2
                else:
                    vals[m-1, i] = np.nan
                    rngs[m-1, i] = np.nan

    return [vals, rngs]


def aiq(arr):
    "Add in quadrature"
    return np.sqrt(np.sum(x**2 for x in arr.ravel()))


def yearly_mean_from_tmm(tmm, anoms):
    val = np.zeros(shape=(12))
    err = np.zeros(shape=(12))
    if tmm[0] == anoms[0] and tmm[1] == anoms[1]:  # first check the dates match
        for m in range(1, 13):
            # find the indices for the current month
            idxs = np.array(tmm[1]) == m
            val[m-1] = np.mean(np.array(tmm[2])[idxs])
            # calculate error by adding uncertainties in quadrature
            err[m-1] = aiq(np.array(tmm[3])[idxs])

    return [val, err]


def absolute_diffs(tmm, yearly_mean):
    """For calculating the difference between a tmm value (continuously) and the
mean. Need to add in errors"""
    diffs = np.zeros(len(tmm[2]))
    for i in range(len(tmm[2])):
        diffs[i] = tmm[2][i] - yearly_mean[0][tmm[1][i] - 1]

    return [tmm[0], tmm[1], diffs]


def relative_anomalies(tmm, yearly_mean):
    """For calculating the anomaly of a tmm value (continuously) relative
to the mean. Need to add in errors"""
    rels = np.zeros(len(tmm[2]))
    for i in range(len(tmm[2])):
        rels[i] = (tmm[2][i] -
                   yearly_mean[0][int(tmm[1][i])-1])/yearly_mean[0][int(tmm[1][i])-1]

    return [tmm[0], tmm[1], rels]

    
def stacked_absolute_diffs(stacked, yearly_mean):
    """For calculating the difference between a tmm value and stacking by
month and event."""
    diffs = np.zeros(shape=(12, 3))
    diffs_err = np.zeros(shape=(12, 3))
    for i in range(0, 3):
        diffs[:, i] = stacked[0][:, i] - yearly_mean[0]
        for m in range(0, 12):
            diffs_err[m, i] = aiq(np.array(stacked[1][m, i],
                                           yearly_mean[1][m]))

    return [diffs, diffs_err]


def stacked_abs_to_rel(diffs, yearly_mean):
    """Converts (stacked) absolute differences to relative anomalies,
using the mean of all the data as a baseline
    """
    rel = np.zeros(shape=(12, 3))
    rel_err = np.zeros(shape=(12, 3))
    for i in range(0, 3):
        rel[:, i] = diffs[0][:, i]/yearly_mean[0]
    for j in range(0, 3):
        for m in range(0, 12):
            rel_err[m, j] = abs(rel[m, j])*(diffs[1][m, j]/abs(diffs[0][m, j]) +
                                            yearly_mean[1][m]/abs(yearly_mean[0][m]))
    
    return [rel, rel_err]


def x_labels(period):
    """Setting x tick labels. Can be set to 'm' (monthly), 'tm' (three monthly), 'h' (harmonic. Must be called after a Figure instance."""
    ax = plt.gca()
    if period == 'm':
        labels = ['J', 'F', 'M', 'A', 'M', 'J',
                  'J', 'A', 'S', 'O', 'N', 'D']
        plt.xticks(np.arange(12))
        ax.set_xticklabels(labels)
    elif period == 'tm':
        labels = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                  'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
        plt.xticks(np.arange(12))
        ax.set_xticklabels(labels)
    elif period == 'h':
        labels = ['J', 'A', 'S', 'O', 'N', 'D',
                  'J', 'F', 'M', 'A', 'M', 'J',
                  'J', 'A', 'S', 'O', 'N', 'D',
                  'J', 'F', 'M', 'A', 'M', 'J']
        plt.xticks(np.arange(24))
        ax.set_xticklabels(labels)
    else:
        print('Enter m, tm or h')
        return None


def plot_three_errorbars(data, data_err, ylims, ylabel, xlabel, save=False):
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
    titles = ['EN', 'LN', 'Neither']
    for i in range(0,3):
        ax = axes.ravel()[i]
        ax.errorbar(np.arange(12), data[:, i], data_err[:, i])
        ax.set_ylim(ylims)
        ax.set_title(titles[i])
        ax.set_ylabel(ylabel)
   
    x_labels(xlabel)
    plt.tight_layout()

    if not save:
        plt.show()
        return None
    else:
        fn = input('Enter filename for plot ')
        plt.savefig(fn)
        plt.show()
        return None


def rainfall_three_monthly_means(data, start, end):
    # now calculate tmm
    date = start
    idx = np.argwhere((data[:, 0] == date.year)&(data[:, 1] == date.month)).ravel()
    c = 0
    years = []
    months = []
    vals = []
    while date < end:
        vals.append(np.mean(data[[idx+c-1, idx + c, idx+c+1], 2]))
        years.append(date.year)
        months.append(date.month)
        date = add_month(date)
        c += 1

    return np.array([years, months, vals])


def reduce_to_shorter_range(short_data, long_data):
    """Reduces long range data to shorter range data (i.e. rainfall data are only available up to 2015) for producing correclations. Returns indexes of larger array to use."""
    idxs = np.zeros(short_data.shape[1])
    for n in range(0, short_data.shape[1]):
        idxs[n] = np.argwhere(((long_data[0] == short_data[0, n])&
                               (long_data[1] == short_data[1, n])).ravel())

    return idxs.astype(int)


def rainfall_monthly_means(data):
    mean_rf = np.zeros(12)
    sig_rf = np.zeros(12)
    for m in range(1, 13):
        idxs = np.argwhere(data[:, 1] == m).ravel()
        mean_rf[m-1] = np.mean(data[idxs, 2])
        sig_rf[m-1] = np.std(data[idxs, 2])

    return np.array([mean_rf, sig_rf])


def load_cloud_fraction_period(start, end, region):
    """Loads all the cloud fraction for a given period and region. Returns
a list of lists in the format [YEAR, MONTH, DATA, ERR]."""
    date = start
    # initialise
    year = []
    month = []
    data = []
    err = []
    # load the data
    while date < end:
        tmp = load_month(date, region)[1]  # [1] for cloud fraction
        
        year.append(date.year)
        month.append(date.month)
        data.append(tmp[0])
        err.append(tmp[1])
        
        date = add_month(date)

    return [year, month, data, err]


def yearly_mean(data, err=True):
    """Input data is of the form [[YEAR], [MONTH], [DATA], [ERR]] so need
to index. Returns yearly mean across months. The err argument allows
the function to be used with rainfall data, which wasn't given with an
error and would return an error when indexed as if it did.

    """
    if err:
        val = np.zeros(shape=(12))
        err = np.zeros(shape=(12))
        for m in range(1, 13):
            # find the indices for the current month
            idxs = np.array(data[1]) == m
            val[m-1] = np.mean(np.array(data[2])[idxs])
            # calculate error by adding uncertainties in quadrature
            err[m-1] = aiq(np.array(data[3])[idxs])

        return [val, err]

    else:
        val = np.zeros(shape=(12))
        for m in range(1, 13):
            # find the indices for the current month
            idxs = np.array(data[1]) == m
            val[m-1] = np.mean(np.array(data[2])[idxs])

        return val


def cloud_fraction_anomalies(data, data_means, start, end, err=True):
    """Takes monthly mean data and calculates differences and anomalies
using yearly means. The err argument allows the function to be used
with rainfall data, which wasn't given with an error and would return
an error when indexed as if it did.

    """
    date = start
    diffs = np.zeros(len(data[2]))
    anoms = np.zeros(len(data[2]))
    errs = np.zeros(len(data[2]))
    i = 0
    if err:
        while date < end:
            idxs = (np.array(data[0]) == date.year)&(np.array(data[1]) == date.month)
            diffs[i] = np.array(data[2])[idxs] - data_means[0][date.month - 1]
            anoms[i] = (np.array(data[2])[idxs] -
                        data_means[0][date.month - 1])/data_means[0][date.month - 1]
            errs[i] = data_means[1][date.month - 1]

            date = add_month(date)
            i += 1

        return [data[0], data[1], diffs, anoms, errs]

    else:
        while date < end:
            idxs = (np.array(data[0]) == date.year)&(np.array(data[1]) == date.month)
            diffs[i] = np.array(data[2])[idxs] - data_means[date.month - 1]
            anoms[i] = (np.array(data[2])[idxs] -
                        data_means[date.month - 1])/data_means[date.month - 1]
            date = add_month(date)
            i += 1

        return [data[0], data[1], diffs, anoms]


def smooth_data_with_window(cf, step, err=True):
    """Smooths the data with a window of 2*step + 1"""
    if err:
        data = cf[3]  # pull the data from cf
        ucrt = cf[4]
        start_idx = step
        end_idx = len(data) - step
        val = np.zeros(end_idx - start_idx)
        err = np.zeros(end_idx - start_idx)
        for i in range(start_idx, end_idx):
            val[i-start_idx] = np.mean(data[i-step:i+step+1])
            srt = np.sort(data[i-step:i+step+1])
            rng = (srt[-2] - srt[1])/2  # take the second smallest and
            # largest as the range
            err[i-start_idx] = aiq(np.array([ucrt[i], rng]))  # aiq
                                                              # with
                                                              # error
                                                              # on
                                                              # anomaly
    
        return [val, err]

    else:
        data = cf[3]  # pull the data from cf
        start_idx = step
        end_idx = len(data) - step
        val = np.zeros(end_idx - start_idx)
        for i in range(start_idx, end_idx):
            val[i-start_idx] = np.mean(data[i-step:i+step+1])

        return val


def adjust_dates_to_smoothed_range(start, end, step):
    """Converts dates to smoothed dates in order to account for central
months lost due to the window size."""
    new_start = start
    new_end = end
    for i in range(0, step):
        new_start = add_month(new_start)
        new_end = subtract_month(new_end)

    return [new_start, new_end]


def narrow_to_range(data, start, end):
    """Specifically for narrowing rainfall data to a specified range."""
    start_idx = np.argwhere((np.array(data[0]) == start.year)&
                            (np.array(data[1]) == start.month)).ravel()
    end_idx = np.argwhere((np.array(data[0]) == end.year)&
                          (np.array(data[1]) == end.month)).ravel()

    return [data[i][start_idx[0]:end_idx[0]] for i in range(0, len(data))]
    


def replace_with_nans(x, idx):
    """Useful for plotting time series data not plotting certain points. x
is the data, idx is a Boolean array for indexing."""
    data = x.copy()
    data[idx] = np.nan
    return data


def plot_idxs_with_inset_correlations(plotting_data, idxs, titles, corr_labels):
    fig = plt.figure(figsize=(8, 8))
    for index, pos in enumerate(idxs):
        p = index + 1
        ax = fig.add_subplot(2, 2, p)
        for data in plotting_data:
            ax.plot(replace_with_nans(data, ~pos)) # replace with nans
                                                   # to produce
                                                   # appropriate gaps
                                                   # in the time
                                                   # series
        ax.set_title(titles[index])
        ax.set_ylim([-2.5, 2.5])
        ax.set_xlim([0, len(data)])
        ax.legend(corr_labels)
    
        corrs = np.corrcoef([plotting_data[0][pos],
                             plotting_data[1][pos],
                             plotting_data[2][pos]])
        
        ax1 = inset_axes(ax, 0.75, 0.75, loc=4)
        ax1.matshow(corrs)
        for (i, j), z in np.ndenumerate(corrs):
            ax1.text(j, i, '{:0.1f}'.format(z), ha='center',
                     va='center', color='white')
            plt.xticks([0, 1, 2])
            plt.yticks([0, 1, 2])
            ax1.set_xticklabels(corr_labels)
            ax1.set_yticklabels(corr_labels)
            ax1.tick_params(axis='both', which='both', bottom=False, top=False,
                            left=False, right=False, labelsize='small')


def plot_with_inset_correlations(plotting_data, corr_labels):
    """plotting_data[0] will be barred, plotting_data[1] will be a line"""
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(np.arange(0, len(plotting_data[0])), plotting_data[0])
    ax.set_xlim([0, len(plotting_data[0])])
    ax.set_ylim([-1.5, 1.5])
    ax.set_ylabel(r'$x/\mu - 1$')
    ax1 = ax.twinx()
    ax1.plot(plotting_data[1])
    ax1.set_ylim([-2.75, 2.75])
    ax1.set_ylabel('ONI 3-monthly anomalies ($^{\circ}$C)')
    plt.legend(corr_labels)
    
    corrs = np.corrcoef(plotting_data)

    in_ax = inset_axes(ax, 0.5, 0.5, loc=4)
    in_ax.matshow(corrs)
    for (i, j), z in np.ndenumerate(corrs):
        in_ax.text(j, i, '{:0.1f}'.format(z), ha='center',
                 va='center', color='white')
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        in_ax.set_xticklabels(corr_labels)
        in_ax.set_yticklabels(corr_labels)
        in_ax.tick_params(axis='both', which='both', bottom=False, top=False,
                        left=False, right=False, labelsize='small')
        plt.subplots_adjust(left=0.2, right=0.8)


def plot_three_with_inset_correlations(plotting_data, corr_labels):
    """plotting_data[0] will be barred, plotting_data[[1,2]] will be a line"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(0, len(plotting_data[0])), plotting_data[0],
           label=corr_labels[0], color='k', fill=False, alpha=None)
    ax.set_xlim([0, len(plotting_data[0])])
    ax.set_ylim([-1.25, 1.25])
    ax.set_ylabel(r'$x/\mu - 1$')
    ax1 = ax.twinx()
    ax1.plot(plotting_data[1], label=corr_labels[1], color='b')
    ax1.plot(plotting_data[2], label=corr_labels[2], color='r')
    ax1.set_ylim([-1.75, 2.75])
    ax1.set_ylabel('SST anomalies ($^{\circ}$C)')
    # ax.set_xticks(x_labels[0])
    # ax1.set_xticks(x_labels[0])
    # ax.set_xticklabels(x_labels[1])
    # ax1.set_xticklabels(x_labels[1])
    
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)
    
    corrs = np.corrcoef(plotting_data)

    in_ax = inset_axes(ax, 0.75, 0.75, loc=4)
    in_ax.matshow(corrs)
    for (i, j), z in np.ndenumerate(corrs):
        in_ax.text(j, i, '{:0.1f}'.format(z), ha='center',
                 va='center', color='white')
        plt.xticks([0, 1, 2])
        plt.yticks([0, 1, 2])
        in_ax.set_xticklabels(corr_labels)
        in_ax.set_yticklabels(corr_labels)
        in_ax.tick_params(axis='both', which='both', bottom=False, top=False,
                        left=False, right=False, labelsize='small')
        plt.subplots_adjust(left=0.2, right=0.8)


def plot_three_with_one_barred(plotting_data, corr_labels):
    """plotting_data[0] will be barred, plotting_data[[1,2]] will be a line"""
    n = len(plotting_data[0])
    nind = np.arange(n)
    nx = nind+0.5
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(nind, plotting_data[0], label=corr_labels[0])
    ax.set_xlim([0, n])
    ax.set_ylim([-1.25, 1.25])
    ax.set_ylabel(r'CF anomaly $x_i/\mu - 1$')
    ax1 = ax.twinx()
    ax1.plot(nx, plotting_data[1], label=corr_labels[1], color='r')
    ax1.plot(nx, plotting_data[2], label=corr_labels[2], color='g')
    ax1.set_ylim([-1.75, 2.75])
    ax1.set_ylabel('SST anomalies ($^{\circ}$C)')

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)

    # ax.set_xticks(np.linspace(0, len(plotting_data[0], 11.5))*len(plotting_data[0]))
    # ax1.set_xticks(np.linspace(0, len(plotting_data[0], 11.5))*len(plotting_data[0]))
    # ax.set_xticklabels(x_labels)
    # ax1.set_xticklabels(x_labels)


def plot_three_with_one_errorbars(plotting_data, corr_labels):
    """plotting_data[0] will be errorbars, plotting_data[[1,2]] will be a line"""
    n = len(plotting_data[0][0])
    nind = np.arange(n)
    nx = nind+0.5
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.errorbar(nind, plotting_data[0][0], plotting_data[0][1],
                 label=corr_labels[0])
    ax.set_xlim([0, n])
    ax.set_ylim([-1.25, 1.25])
    ax.set_ylabel(r'CF anomaly $x_i/\mu - 1$')
    ax.axhline(linewidth=1, color='k')
    ax1 = ax.twinx()
    ax1.plot(plotting_data[1], label=corr_labels[1], color='r')
    ax1.plot(plotting_data[2], label=corr_labels[2], color='g')
    ax1.set_ylim([-2.75, 2.75])
    ax1.set_ylabel('SST anomalies ($^{\circ}$C)')

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)


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
    ax.set_ylim([-1.25, 1.25])
    ax.set_ylabel(r'CF$_{\sigma}$')
    ax1 = ax.twinx()
    ax1.plot(plotting_data[1], label=corr_labels[1], color='g')
    ax1.plot(plotting_data[2], label=corr_labels[2], color='r')
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
    ax.set_ylim([-1.25, 1.25])
    ax.set_ylabel(r'CF$_{\sigma}$')
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

def shift_dates(start, end, months):
    """Shifts a date by a given amount. Useful for shifting ONI."""
    shifted_start = start
    shifted_end = end
    if months < 0:
        for i in range(0, int(-months)):
            shifted_start = subtract_month(shifted_start)
            shifted_end = subtract_month(shifted_end)
    elif months > 0:
        for i in range(0, int(months)):
            shifted_start = add_month(shifted_start)
            shifted_end = add_month(shifted_end)

    # check that the date falls in the correct range
    if shifted_end > datetime.datetime.strptime('201512','%Y%m'):
        print('Warning: date outside rainfall data range')

    return (shifted_start, shifted_end)


def swio_three_monthly_means(swio_datetime, swio_anoms):
    vals = []
    err = []
    year = []
    month = []
    start_month = add_month(swio_datetime[0])  # start a month in
    end_month = subtract_month(swio_datetime[-1])  # finish a month early
    curr_month = start_month
    while curr_month < end_month:
        prev_month = subtract_month(curr_month)
        next_month = add_month(curr_month)
        args = np.zeros(len(swio_datetime), dtype=bool)
        for i, x in enumerate(swio_datetime):
            args[i] = ((x.year == curr_month.year)&(x.month == curr_month.month) or
                       (x.year == next_month.year)&(x.month == next_month.month) or
                       (x.year == prev_month.year)&(x.month == prev_month.month))

        vals.append(np.mean(swio_anoms[args]))
        err.append(np.std(swio_anoms[args]))
        year.append(curr_month.year)
        month.append(curr_month.month)
    
        curr_month = add_month(curr_month)

    return [year, month, vals, err]


def load_swio(fname, key):
    dataset = Dataset(fname)
    swio_time = np.asarray(dataset.variables['WEDCEN2'])  # time in days since 1/1/1900
    swio_anoms = np.asarray(dataset.variables[key])
    # convert swio_time to month and year
    init_date = datetime.datetime.strptime('111900', '%d%m%Y')
    start_delta = datetime.timedelta(days=swio_time[0])
    start_date = init_date + start_delta
    swio_datetime = [start_date + i*datetime.timedelta(days=7)
                     for i in range(0, len(swio_time))]

    return [swio_datetime, swio_anoms]


def narrow_swio(swio_tmm, start, end):
    years = np.array(swio_tmm[0])
    months = np.array(swio_tmm[1])
    start_idx = int(np.argwhere((years == start.year)&(months == start.month)))
    end_idx = int(np.argwhere((years == end.year)&(months == end.month)))
    span = slice(start_idx, end_idx)
    
    return [x[span] for x in swio_tmm]


def month_and_year_labels(month_labels, year_labels, month_step):
    # now convert from number to letter
    month_letter = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_labels = [month_letter[x-1] for x in month_labels]
    count = 0
    for i in range(0, len(month_labels)):
        if month_labels[i] == month_labels[0] + month_step:
            x_labels[i] += '\n' + str(year_labels[count])
            count += 1

    return x_labels


# start = datetime.datetime.strptime('200901','%Y%M')
# end = datetime.datetime.strptime('201801', '%Y%M')
# region = 'capetown'

# nino = nino_range('detrend.nino34.ascii.txt', 1., 2009., 1., 2018.)
    
# # indices for El Nino, La Nina and neither months
# en = np.argwhere(nino[:, 2] >= 0.5).ravel()
# ln = np.argwhere(nino[:, 2] <= -0.5).ravel()
# nei = np.argwhere((nino[:, 2] > -0.5)*(nino[:, 2] < 0.5)).ravel()
# args = [en, ln, nei]

# # # now separate the cloud data into different years
# # sep = np.zeros(shape=(12, 3)) 
# # for i, arg in enumerate(args):
# #     idxs = to_idx(nino, arg)
# #     for n in range(0,12):
# #         sep[n, i] = np.mean([cloud[x] for x in idxs if x[0] == n])

# # # now plot anomalies
# # plt.figure()
# # for i in range(0,3):
# #     plt.plot(sep[:, i] - cloud_mean)
# #     plt.legend(['EN', 'LN', "neither"])
# #     plt.title('Monthly - Cape Town')
# #     plt.ylabel(r'$c - \mu$')
# #     labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
# #     plt.xticks(np.arange(0,12))
# #     ax = plt.gca()
# #     ax.set_xticklabels(labels)
# #     plt.tight_layout()

# # plt.figure()
# # for i in range(0,3):
# #     plt.plot((sep[:, i] - cloud_mean)/cloud_std)
# #     plt.legend(['EN', 'LN', "neither"])
# #     plt.title('Monthly - Cape Town (Normalised)')
# #     plt.ylabel(r'$(c - \mu)/\sigma$')
# #     labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
# #     plt.xticks(np.arange(0,12))
# #     ax = plt.gca()
# #     ax.set_xticklabels(labels)
# #     plt.tight_layout()
# #     plt.show()


# ### new ###
# # pick out consecutive anomalies
# anoms = np.array(consecutive_anomalies(nino))

# # now pick out seasonal anomalies
# djf = anoms[:, anoms[1] == 1]
# jja = anoms[:, anoms[1] == 7]

# # now pick out EN/LN/neutral
# en_djf = djf[:, djf[2] == 1]
# ln_djf = djf[:, djf[3] == 1]
# neu_djf = djf[:, djf[2]+djf[3] == 0]
# seasons = [djf, en_djf, ln_djf, neu_djf]

# # now for calculating three monthly means of data
# start = datetime.datetime.strptime('20092','%Y%m')
# end = datetime.datetime.strptime('201711','%Y%m')
# tmm = three_monthly_means(start, end, region)


# #### plotting below here
# cloud_masks = [np.mean(load_seasonal(s, region, 0), axis=0) for s in seasons]
# titles = ['ALL', 'EN', 'LN', 'NEUTRAL']

# ### cloud masks
# # fig, axes = plt.subplots(2, 2)
# # for idx, ax in enumerate(axes.flat):
# #     im = ax.imshow(cloud_masks[idx])
# #     ax.set_title(titles[idx])
# #     ax.axis('off')
# # fig.colorbar(im, ax=axes.ravel().tolist())
# # plt.show()

# cloud_frac = [load_seasonal(s, region, 1) for s in seasons]
# seas_frac = np.zeros(shape=(len(cloud_frac), 2))

# ### seasonal difference plot
# # plt.figure()
# # for idx, frac in enumerate(cloud_frac):
# #     seas_frac[idx, 0] = np.mean(frac[0])
# #     seas_frac[idx, 1] = np.sqrt(np.mean(frac[1])**2 + np.std(frac[0])**2)
# # plt.errorbar(-1, seas_frac[1, 0], seas_frac[1, 1])
# # plt.errorbar(0, seas_frac[2, 0], seas_frac[2, 1])
# # plt.errorbar(1, seas_frac[3, 0], seas_frac[2, 1])
# # plt.legend(titles[1::])


# total_mean_frac = np.array([np.mean([load_month_from_anoms(month, year, region)[1][0]
#                             for year in range(2009, 2018)]) for month in range(1, 13)])

# T = 24  # period in months
# def sine_wave(x, a, T=T):
#     return a*np.sin((2*np.pi/T)*x)

# en_diff = harmonic_anomalies(2015, region)
# ln_diff = harmonic_anomalies(2011, region)
# neu_diff = harmonic_anomalies(2013, region)

# diffs = [en_diff, ln_diff, neu_diff]

# titles = ['2015 El Nino', '2011 La Nina',
#           '2013 Neutral']

# xvals = np.arange(0, len(en_diff))
# labels = ['J', 'A', 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J',
#           'J', 'A', 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J']
# p0 = [0.05]  # initial guess

# ylims = (-0.2, 0.2)  # force all axes to have same y limits

# #plt.figure(figsize=(9, 6))

# # for idx, diff in enumerate(diffs):
# #     plt.subplot(2, 2, idx+1)
# #     params, params_covariance = optimize.curve_fit(sine_wave, xvals, diff, p0)
# #     chi, p = stats.chisquare(diff, sine_wave(xvals, params[0]))
# #     plt.step(xvals, diff, color='k', label='_nolegend_')
# #     plt.plot(xvals, sine_wave(xvals, params[0]), color='k')
# #     plt.ylabel(r'$x-\mu$')
# #     plt.xlabel('Month')
# #     plt.title(titles[idx])
# #     plt.axvline(6, color='k', linestyle='dashed')
# #     plt.axvline(18, color='k', linestyle='dashed')
# #     plt.ylim(ylims)
# #     plt.legend([r'$\chi ^2 = {}$'.format(chi)])
# #     plt.xticks(np.arange(24))
# #     ax = plt.gca()
# #     ax.set_xticklabels(labels)
# #     ax.tick_params(axis='x',which='minor',bottom='off', top='off')
# #     plt.tight_layout()
# # plt.suptitle(r'{}, $T={}$'.format(region, T) )

# # plt.subplot(2, 2, 4)
# # xvals1 = np.arange(12)
# # p0 = [0.2,1/2]
# # plt.step(xvals1, total_mean_frac, color='k')
# # plt.ylabel(r'$\mu$')
# # plt.xlabel('Month')
# # plt.title('Mean (Eastern Africa)')
# # plt.xticks(xvals1)
# # ax = plt.gca()
# # ax.set_xticklabels(labels[6:18])
# # ax.tick_params(axis='x',which='minor',bottom='off', top='off')
# # x0,x1 = ax.get_xlim()
# # y0,y1 = ax.get_ylim()
# # ax.set_aspect((x1-x0)/(y1-y0))
# # plt.tight_layout()


