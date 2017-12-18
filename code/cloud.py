import numpy as np
import matplotlib
#make figure windows active
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import glob

## works but is very slow
## probably don't want to load all 365 images at once

#get a list of filenames for the vis8 data from 2009
fnames = glob.glob("vis8/nine/*.jpg")
print('fname done')
#read in the images
X = np.array([plt.imread(fname) for fname in fnames])
print('loading done')
#flatten the data for a histogram
X_flat = X.flatten()
print('flattening done')
#show a histogram
plt.figure()
#bigger than zero to account for outside
plt.hist(X_flat[X_flat > 0], bins = 255) 
plt.show()
