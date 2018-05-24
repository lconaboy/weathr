import numpy as np
import matplotlib.pyplot as plt
import datetime
from util import *
from coverage_analysis_functions import *
from ndvi_spatial_analysis_functions import *
import matplotlib.animation as animation

region = 'capetown'

mean = ndvi_monthly_spatial_medians(region)

anoms, years, months = ndvi_spatial_anomalies(mean, region, means='monthly', s=False)

# load ONI data
# select date range
start = datetime.datetime.strptime(str(years[0])+str(months[0]), '%Y%m')
end = datetime.datetime.strptime(str(years[-1])+str(months[-1]), '%Y%m')
# end is not included in nino_range, so should be a month further on
end = add_month(end)
# need to account for the ONI being smoothed by three months by
# starting a month earlier and ending a month later
new_start = subtract_month(start)
new_end = add_month(end)
# shift the dates to check for correlation
# shift = -2
# new_start, new_end = shift_dates(new_start, new_end, shift)
nino = nino_range('detrend.nino34.ascii.txt', new_start, new_end)
# pick out consecutive anomalies
oni_anoms = consecutive_anomalies(nino)
en = np.array(oni_anoms[2])
ln = np.array(oni_anoms[3])
neu = np.array(np.logical_and(~oni_anoms[2], ~oni_anoms[3]))
tot = np.ones(shape=oni_anoms[2].shape, dtype=bool)  # idxs for all data
idxs = [neu, en, ln]

# # colourmaps and land outline
cm1, cm2 = colourmaps(15)  # cm1 is red-green colourmap, cm2 is transparent-black
outline = outline_region(region)

# now stack seasonal anomalies
seasons = [1, 7]                # DJF & JJA
fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69))
titles = [['DJF', 'JJA'], ['Neutral', 'El Nino', 'La Nina']]
for col, season in enumerate(seasons):
    for row, idx in enumerate(idxs):

        tmp_idx = (np.array(months)==season)&idx
        tmp_anoms = anoms[tmp_idx]
        im = axes[row][col].imshow(np.median(tmp_anoms, axis=0), cm1, vmin=-0.75,
                                   vmax=0.75)
        axes[row][col].imshow(outline, cmap=cm2)
        axes[row][col].set_title(titles[0][col] + ' ' + titles[1][row])
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])

cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.01])
cb = plt.colorbar(im, cax=cbaxes, orientation="horizontal", extend='both')
cb.set_label(r'NDVI$_{\sigma}$')
plt.suptitle(region_to_string(region))
plt.savefig('ndvi_spatial_seasonal_anomalies_{}'.format(region))
plt.show()


# BIG ANIMATION TINGS YA DUN KNO
# # colourmaps and land outline
# cm1, cm2 = colourmaps(17)  # cm1 is red-green colourmap, cm2 is transparent-black
# outline = outline_region(region)

# # make animation of ndvi
# fig, ax = plt.subplots(1, 1)
# im = ax.imshow(anoms[0], cmap=cm1, vmin=-1, vmax=1)
# ax.imshow(outline, cmap=cm2)
# cb = fig.colorbar(im, shrink=0.95)
# cb.set_label(r'NDVI$_{\sigma}$')

# def animate(i):
#     im = ax.imshow(anoms[i] + outline, cmap=cm1,
#                    vmin=-1, vmax=1, interpolation='bicubic')  # update the data
#     plt.axis('off')
#     plt.title('{}/{}'.format(int(titles[i][4:6]), titles[i][0:4]))
#     return ax,

# ani = animation.FuncAnimation(fig, animate, np.arange(0, len(anoms)),
#                               interval=15, blit=False)
# # ani.save('ndvi_anoms_{}.mp4'.format(region), writer='ffmpeg')
# plt.show()
