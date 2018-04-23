import numpy as np

def ndvi(nir, vis6):
    return np.divide((nir - vis6), (nir + vis6),
                     out=np.zeros_like(nir),
                     where=((nir!=0)&(vis6!=0)))

radiance_calibration_values = {'vis6': {'offset': -1.092527,
                                        'slope': 0.021422},
                               'vis8': {'offset': -1.442708,
                                        'slope': 0.028288},
                               'nir': {'offset': -1.208853,
                                       'slope': 0.023703}}

def calibrate(data, band):
    return data * 4 * radiance_calibration_values[band]['slope'] + radiance_calibration_values[band]['offset']

def average_with_thresholds(image, thr):
    # average down the third axis counting elements that are greater
    # than thr
    sx, sy, sz = image.shape
    output = np.zeros_like(thr)

    for i in np.arange(sx):
        for j in np.arange(sy):
            loc = (image[i, j, :] <= thr[i, j])
            if np.count_nonzero(loc) == 0:
                # These seems to happen way too often. Why? Also, it
                # looks like whenever this does happen, the minimum
                # value of the pixel is only slightly above the
                # threshold. Because that is the case, here we just
                # use the min value as the "average".
                # TODO Investigate why these pixels are coming above threshold.
                mini = np.min(image[i, j, :])
                output[i, j] = mini
            else:
                output[i, j] = np.mean(image[i, j, loc])
    return output

