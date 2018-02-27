import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import time

years = ['08/', '09/', '10/', '11/', '12/', '13/', '14/', '15/', '16/', '17/']

for year in years:
    path = 'data/' + year

    nir_path = path + 'nir/'
    vis8_path = path + 'vis8/'
    vis6_path = path + 'vis6/'
    
    if not os.path.exists(nir_path):
        os.makedirs(nir_path)

    if not os.path.exists(vis8_path):
        os.makedirs(vis8_path)

    if not os.path.exists(vis6_path):
        os.makedirs(vis6_path)
    
    im_path = path + '*.jpg'

    fnames = glob.glob(im_path)

    for idx in range(0, len(fnames)):
        print('Separating file {} [{}/{}]'.format(fnames[idx], idx+1, len(fnames)),
              end='\r')
        # image = np.asarray(Image.open(fnames[idx]), dtype=int)
        image = Image.open(fnames[idx])

        R,G,B = image.split()
        R.save(nir_path + fnames[idx][8::])
        G.save(vis8_path + fnames[idx][8::])
        B.save(vis6_path + fnames[idx][8::])
