"""General utility stuff for our project. Put things like regions of
interest here, so that they don't clutter other files, but they're
always accessible when needed.

"""
import glob
import datetime
import os
import numpy as np
from PIL import Image


def slice_d(region):
    """The difference between region slice's stop and start."""
    start = region.start
    stop = region.stop

    return stop - start


def make_region(slicex=slice(None), slicey=slice(None)):
    """Makes a region with default empty slices."""
    return [slicex, slicey]


def image_region(image, region):
    """Returns the slice of image using region."""
    return image[region[0], region[1]]


def load_images_with_region(files, region):
    """Loads all images in files narrowed to the given region.

Note: be careful, this could raise a MemoryError exception."""
    new_shape = image_region(np.asarray(Image.open(files[0]), dtype=int), region).shape
    images = np.zeros((new_shape[0], new_shape[1], len(files)))

    for idx in np.arange(0, len(files)):
        print('Loading file {} [{}/{}]'.format(files[idx], idx+1, len(files)), end='\r')
        image = np.asarray(Image.open(files[idx]), dtype=int)
        images[:, :, idx] = image_region(image, region) * (1 - image_region(land_mask, region))

    print('All files loaded...', end='\n')
    return images


def parse_data_datetime(path, band):
    """Takes the path (i.e. './data/eumetsat/') and returns the
    datetime variables in  a list"""

    globbo = path + '*_' + band + '.png'
    # load pathnames
    pnames = glob.glob(globbo)
    # load filenames
    fnames = [os.path.splitext(os.path.basename(x))[0][0:-1-len(band)]
              for x in pnames]
    # parse filenames into datetime
    dnames = [datetime.datetime.strptime(fname, '%Y%m%d%H%M')
              for fname in fnames]

    return dnames


def parse_data_string(dnames, path, band):
    """Converts datetime back to strings (i.e. fnames)"""
    strings = [dname.strftime('%Y%m%d%H%M') for dname in dnames]

    fnames = [path + string + '_' + band + '.png' for string in strings]

    return fnames


def load_data_window(dnames, start, window):
    """Takes datetime parsed names from parse_data_datetime"""

    return [fname for fname in dnames if fname > (start-window/2)
            and fname < (start+window/2)]


def sep_months(dnames, y, m):
    """Takes datetime parsed data and separates into month and year"""
    idxs = [dn.month == m and dn.year == y for dn in dnames] 

    return idxs


def sep_years(dnames, y):
    """Takes datetime parsed data and separates into year"""
    idxs = [dn.year == y for dn in dnames] 

    return idxs


def path_to_weathr_data(band):
    path = './data/eumetsat/'
    glob_path = path + '*_' + band +'.png'

    return path, glob_path



# Our particular areas of interest. Useful for cutting down on
# processing time, and focusing in on the action.
weathr_regions = {
    'all': make_region(),
    'africa': make_region(slice(620, 3020), slice(1260, 3000)),
    'capetown': make_region(slice(2615, 3015), slice(2350, 2750)),
    'egypt': make_region(slice(800,900), slice(2725, 2825)),
    'eastafrica': make_region(slice(1500, 1800), slice(3000, 3300))}

# These are named paths/globs for data we use. We should at some point come
# up with a better way of storing our data as it's all over the place
# at the moment.
weathr_data = {'vis6': './bands13/vis6/*.jpg',
               'landmask': './landmask.gif'}

# land_mask = image_region(np.asarray(Image.open(weathr_data['landmask']), dtype=int),
#                           weathr_regions['capetown'])
land_mask = np.asarray(Image.open(weathr_data['landmask']), dtype=int)

# output configurations
figure_dir = 'results/figures/'
