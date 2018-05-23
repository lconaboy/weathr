import numpy as np
import matplotlib.pyplot as plt
from util import *
from PIL import Image
import datetime
from cloud_coverage import cloud_coverage

regions = ('capetown', 'eastafrica')
bands = ('vis6','vis8')  # bands for multiband
months = np.arange(1,13)
years = np.arange(2008, 2018)

for region in regions:
    for y in years:
        for m in months:
            cloud_coverage(region, y, m)
