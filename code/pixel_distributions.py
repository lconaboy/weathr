import numpy as np
import matplotlib.pyplot as plt
from util import *
from cloud_free import threshold

vis6 = load_images_with_region(glob.glob('./data/eumetsat/2015*_vis6.png'),
                               region=weathr_regions['capetown'])
otsu_vis6 = threshold(vis6)

vis8 = load_images_with_region(glob.glob('./data/eumetsat/2015*_vis8.png'),
                               region=weathr_regions['capetown'])
otsu_vis8 = threshold(vis8)

nir = load_images_with_region(glob.glob('./data/eumetsat/2015*_nir.png'),
                               region=weathr_regions['capetown'])
otsu_nir = threshold(nir)

points = [[50, 50], [150, 150], [250, 250], [330, 100]]

# Let's first take a look a sample of points in all three bands
fig, axs = plt.subplots(4, 3, figsize=(8, 12))

for (a, b), ax in zip(points, axs):
    for imgs, otsu, axe in zip([vis6, vis8, nir], [otsu_vis6, otsu_vis8, otsu_nir], ax):
        axe.hist(imgs[a, b, :], bins=10)

# Add titles to first row only
titles = ['VIS6', 'VIS8', 'NIR']
for ax, title in zip(axs[0], titles):
    ax.set_title(title)

plt.suptitle('Pixel value distributions in all bands', y=0.94)
plt.savefig(figure_dir + 'pixel_distributions.pdf')

# Now let's look at statistics in VIS6 and VIS8, so that we can
# motivate our choices of using median, etc.
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

for imgs, otsu, ax in zip([vis6, vis8], [otsu_vis6, otsu_vis8], axs):
    for (a, b), axe in zip(points, ax):
        hist = axe.hist(imgs[a, b, :], bins=10, color='lightblue')
        axe.vlines(otsu[a, b], 0, 1, transform=axe.get_xaxis_transform(), color='k',
                  label='Otsu')
        # # naive peak choice
        # peak = (hist[1][np.argmax(hist[0])] + hist[1][np.argmax(hist[0]) + 1])/2.0
        # ax.vlines(peak, 0, 1, transform=ax.get_xaxis_transform(), colors=['brown'],
        #            label='peak')

        mean1 = np.mean(imgs[a, b, :])
        axe.vlines(mean1, 0, 1, transform=axe.get_xaxis_transform(), color='g',
                  label='Mean')

        mean2 = np.mean(imgs[a, b, :][imgs[a, b, :] < otsu[a, b]])
        axe.vlines(mean2, 0, 1, transform=axe.get_xaxis_transform(), color='r',
                  label='Mean < Otsu')

        med1 = np.median(imgs[a, b, :])
        axe.vlines(med1, 0, 1, transform=axe.get_xaxis_transform(), color='b',
                  label='Median', linestyles=['dashed'])

        # med2 = np.median(imgs[a, b, :][imgs[a, b, :] < otsu[a, b]])
        # ax.vlines(med2, 0, 1, transform=ax.get_xaxis_transform(), color='darkblue',
        #            label='median without tail')

        # # maybe not best choice
        std1 = np.std(imgs[a, b, :])
        std2 = np.std(imgs[a, b, :][imgs[a, b, :] < otsu[a, b]])
        axe.vlines(med1 + std1, 0, 1, transform=axe.get_xaxis_transform(), color='g',
                  linestyle='dotted',
                  label='Median + std')

titles = ['VIS6', 'VIS8']
for title, ax in zip(titles, axs):
    ax[0].set_ylabel(title)

axs[0][0].legend()

plt.suptitle('Pixel values distributions with statistics, in VIS6 and VIS8', y=0.94)
plt.savefig(figure_dir + 'pixel_distributions_stats.pdf')

