import numpy as np

def ndvi(nir, vis6):
    return np.divide((nir - vis6), (nir + vis6),
                     out=np.zeros_like(nir),
                     where=((nir!=0)&(vis6!=0)))
