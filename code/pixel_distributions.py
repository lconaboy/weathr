import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.mlab import normpdf
from util import *
from cloud_free import threshold

def pixel_dist_for_region(region='capetown'):
    print('Working on region {}'.format(region_to_string(region)))
    vis6 = load_images_with_region(glob.glob('./data/eumetsat/2015*_vis6.png'),
                                   region=weathr_regions[region])
    otsu_vis6 = threshold(vis6)

    vis8 = load_images_with_region(glob.glob('./data/eumetsat/2015*_vis8.png'),
                                   region=weathr_regions[region])
    otsu_vis8 = threshold(vis8)

    nir = load_images_with_region(glob.glob('./data/eumetsat/2015*_nir.png'),
                                   region=weathr_regions[region])
    otsu_nir = threshold(nir)

    points = [[50, 50], [150, 150], [250, 250], [250, 100]]

    # Let's first take a look a sample of points in all three bands
    fig, axs = plt.subplots(4, 3, figsize=(8, 12))

    for (a, b), ax in zip(points, axs):
        for imgs, otsu, axe in zip([vis6, vis8, nir], [otsu_vis6, otsu_vis8, otsu_nir], ax):
            axe.hist(imgs[a, b, :], bins=10)

    # Add titles to first row only
    titles = ['VIS0.6', 'VIS0.8', 'NIR']
    for ax, title in zip(axs[0], titles):
        ax.set_title(title)

    plt.suptitle('Pixel value distributions in all bands, {}'.format(region_to_string(region)),
                 y=0.94)
    plt.savefig(figure_dir + 'pixel_distributions_{}.pdf'.format(region))

    # Now let's look at statistics in VIS6 and VIS8, so that we can
    # motivate our choices of using median, etc.
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for imgs, otsu, ax in zip([vis6, vis8], [otsu_vis6, otsu_vis8], axs):
        for (a, b), axe in zip(points, ax):
            n, bins, hist = axe.hist(imgs[a, b, :], bins=40, color='lightblue', normed=True)
            axe.set_ylim(0, np.max(n)*1.1)
            axe.vlines(otsu[a, b], 0, 1, transform=axe.get_xaxis_transform(), color='k',
                      label='Otsu')

            mean1 = np.mean(imgs[a, b, :])
            axe.vlines(mean1, 0, 1, transform=axe.get_xaxis_transform(), color='g',
                      label=r'$\mu$')

            mean2 = np.mean(imgs[a, b, :][imgs[a, b, :] < otsu[a, b]])
            axe.vlines(mean2, 0, 1, transform=axe.get_xaxis_transform(), color='r',
                      label=r'$\mu$ < Otsu')

            med1 = np.median(imgs[a, b, :])
            axe.vlines(med1, 0, 1, transform=axe.get_xaxis_transform(), color='b',
                       label=r'$\~{x}$', linestyles=['dashed'])

            (μ, σ) = norm.fit(imgs[a, b, :][imgs[a, b, :] < otsu[a, b]])
            fit = normpdf(np.arange(255), μ, σ)
            axe.plot(np.arange(255), fit, 'r--', linewidth=1, label='Gaussian fit < Otsu')

            axe.set_xlim(0, 255)
            axe.text(0.95, 0.05, "$\sigma^* = {:.2f}$ \n $\mu^* = {:.2f}$".format(σ,μ),
                     transform=axe.transAxes, horizontalalignment='right')

    titles = ['VIS0.6', 'VIS0.8']
    for title, ax in zip(titles, axs):
        ax[0].set_ylabel(title)

    axs[0][0].legend()
    axs[1][0].set_xlabel('Pixel value')

    plt.suptitle('Pixel values distributions with statistics, in VIS0.6 and VIS0.8, {}'.format(region_to_string(region)),
                 y=0.94)
    plt.savefig(figure_dir + 'pixel_distributions_stats_{}.pdf'.format(region))

    # Now let's look at the distribution when we include *every* pixel in
    # the (masked) image
    fig, axes = plt.subplots(1, 3, figsize=(12, 8), sharex=True)
    titles = ['VIS0.6', 'VIS0.8', 'NIR']
    mask = (1 - image_region(land_mask, weathr_regions[region])).astype(bool)

    for img, otsu, axis, title in zip([vis6, vis8, nir], [otsu_vis6, otsu_vis8, otsu_nir], axes, titles):
        n, bins, hist = axis.hist(img[mask].ravel(), bins=30, normed=True, color='lightblue')
        axis.set_ylim(0, np.max(n) * 1.1)
        axis.set_title(title)

        # All those pixels over all days that are below their
        # corresponding Otsu threshold, and are also land pixels.
        # This is pure magic.
        img_otsu = img[(img <= otsu[..., np.newaxis]) & mask[..., np.newaxis]]
        # Also some magic. TODO better comment.
        otsu = threshold(img[mask].reshape((1, 1, -1)))[0, 0]
        axis.vlines(otsu, 0, 1, transform=axis.get_xaxis_transform(), label='Otsu', color='k')

        mean = np.mean(img[mask].ravel())
        axis.vlines(mean, 0, 1, transform=axis.get_xaxis_transform(), label='Mean', color='r')

        mean_otsu = np.mean(img_otsu)
        axis.vlines(mean_otsu, 0, 1, transform=axis.get_xaxis_transform(),
                    label='Mean < Otsu', color='g')

        median = np.median(img[mask].ravel())
        axis.vlines(median, 0, 1, transform=axis.get_xaxis_transform(), color='b')

        if title == 'NIR': continue

        (μ, σ) = norm.fit(img_otsu)
        fit = normpdf(np.arange(255), μ, σ)
        axis.plot(np.arange(255), fit, '--', linewidth=2, label='Gaussian fit < Otsu')

        axis.set_xlim(0, 255)
        axis.text(0.95, 0.05, "$\sigma = {:.2f}$ \n $\mu = {:.2f}$".format(σ, μ),
                  transform=axis.transAxes, horizontalalignment='right')
        # BTW, why didn't we just define the land mask to be the above in
        # util.py? That way we wouldn't have to constantly write 1 - land_mask. Dumb!

    axes[0].legend()

    plt.savefig(figure_dir + 'pixel_distributions_all_pixels_{}.pdf'.format(region))

# If running as script; i.e. not being run via import pixel_distributions
if __name__ == '__main__':
    regions = ['capetown', 'eastafrica']
    for region in regions:
        pixel_dist_for_region(region=region)
