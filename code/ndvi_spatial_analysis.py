import numpy as np
import matplotlib.pyplot as plt
import datetime
from util import *
from coverage_analysis_functions import *
import matplotlib.animation as animation


# TODO - produce plots of interesting times

ndvi_dir = 'results/ndvi/'
# yearmonth_region.png
# e.g. 201708_capetown.png
ndvi_fmt = '{}{}_{}'

region = 'eastafrica'

def ndvi_total_spatial_mean(region):
    mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)
    path = ndvi_dir + ndvi_fmt

    fnames = sorted(glob.glob(ndvi_dir + ndvi_fmt.format('*', '*', region) + '.npy'))
    # instead of means we could also use median here
    mean = np.mean([np.load(fname) for fname in fnames], axis=0)

    return mean


def ndvi_monthly_spatial_medians(region):
    """Use medians instead of means due to small datasets (~10 in each median)."""
    mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)
    path = ndvi_dir + ndvi_fmt

    months = ['01', '02', '03', '04', '05', '06',
              '07', '08', '09', '10', '11', '12']

    fnames = [sorted(glob.glob(ndvi_dir + ndvi_fmt.format('*', month, region) +'.npy'))
              for month in months]

    # could use the median instead of the mean
    means = [np.median([np.load(fn) for fn in fname], axis=0) for fname in fnames]
    
    return means


def ndvi_spatial_anomalies(mean, region, means='single'):
    """Calculate anomalies as (x-u)/u using a single mean"""

    mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)

    if means == 'single':
        path = ndvi_dir + ndvi_fmt

        fnames = sorted(glob.glob(ndvi_dir + ndvi_fmt.format('*', '*',
                                                             region) + '.npy'))
        monthlys = [np.load(fname) for fname in fnames]

        anoms = np.divide((monthlys - mean), mean, where=(mask==1),
                          out=np.zeros(np.shape(monthlys)))

        titles = [fname[13:19] for fname in fnames]

    elif means == 'monthly':
        fnames = sorted(glob.glob(ndvi_dir + ndvi_fmt.format('*', '*',
                                                             region) + '.npy'))
        anoms = np.zeros(shape=(len(fnames), mean[0].shape[0], mean[0].shape[1]))
        titles = []
        for idx, fname in enumerate(fnames):
            month = int(fname[17:19])  # convert month in fname to usable number
            monthly = np.load(fname)
            anoms[idx, :, :] = np.divide((monthly - mean[month-1]), mean[month-1],
                                         where=(mask==1),
                                         out=np.zeros(monthly.shape))
            titles.append(fname[13:19])

    # now smooth the anoms
    smooth = 1
    anoms_smoothed = np.zeros(shape=(anoms.shape[0]-2*smooth,
                                     anoms.shape[1], anoms.shape[2]))
    for i in range(smooth, anoms.shape[0]-smooth):
        anoms_smoothed[i-1, :, :] = np.mean(anoms[[i-1, i, i+1], :, :], axis=0)
    
    print('smoothing done')
    
    return anoms_smoothed, titles


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def outline_region(region):
    from scipy import ndimage
    
    mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)
    edge_horizont = ndimage.sobel(mask, 0)
    edge_vertical = ndimage.sobel(mask, 1)
    magnitude = np.hypot(edge_horizont, edge_vertical)

    return magnitude > 0

def colourmaps(n=17):
    import matplotlib.colors as clr
    cm = clr.LinearSegmentedColormap.from_list('', ['darkred', 'firebrick', 'white',
                                                    'darkgreen', 'purple'])
    cm1 = discrete_cmap(n, cm)
    cm2 = clr.LinearSegmentedColormap.from_list('', [[0, 0, 0, 0], 'black'])

    return cm1, cm2

mean = ndvi_monthly_spatial_medians(region)

anoms, titles = ndvi_spatial_anomalies(mean, region,
                                       means='monthly')

# colourmaps and land outline
cm1, cm2 = colourmaps(17)  # cm1 is red-green colourmap, cm2 is transparent-black
outline = outline_region(region)

# make animation of ndvi
fig, ax = plt.subplots(1, 1)
im = ax.imshow(anoms[0], cmap=cm1, vmin=-1, vmax=1)
ax.imshow(outline, cmap=cm2)
cb = fig.colorbar(im, shrink=0.95)
cb.set_label(r'NDVI$_{\sigma}$')

def animate(i):
    im = ax.imshow(anoms[i] + outline, cmap=cm1,
                   vmin=-1, vmax=1, interpolation='bicubic')  # update the data
    plt.axis('off')
    plt.title('{}/{}'.format(int(titles[i][4:6]), titles[i][0:4]))
    return ax,

ani = animation.FuncAnimation(fig, animate, np.arange(0, len(anoms)),
                              interval=15, blit=False)
# ani.save('ndvi_anoms_{}.mp4'.format(region), writer='ffmpeg')
plt.show()
