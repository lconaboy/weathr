import traceback
import sys
import numpy as np
import glob
import re
import matplotlib.pyplot as plt

from gribapi import *

fnames = glob.glob('MSG*.grb')
fout = '{}_msk'

for fname in fnames:
    fh = open(fname)
    datime = re.search('0100-([0-9]{12})', fname).group(1)

    gid = grib_new_from_file(fh)
    values = grib_get_values(gid)

    sys.stdout.write("Input mask: %s\t Output to: %s" % (fname, fout.format(datime)))
    sys.stdout.flush()
    np.save(fout.format(datime), values.astype(int).reshape((3712, 3712)))

    plt.imshow(np.load(fout.format(datime)+'.npy') == 2, cmap='Greys')
    plt.show()
    exit()
    
    assert(len(values) == 13778944)

# INPUT='MSG3-SEVI-MSGCLMK-0100-0100-20140101120000.000000000Z-20140101121307-1268538.grb'
# VERBOSE=1 # verbose error reporting

# def example():
#     f = open(INPUT)
#     gid = grib_new_from_file(f)

#     values = grib_get_values(gid)
#     for i in xrange(len(values)):
#         print "%d %.10e" % (i+1,values[i])

#     print '%d values found in %s' % (len(values),INPUT)

#     grib_release(gid)
#     f.close()

# def main():
#     try:
#         example()
#     except GribInternalError,err:
#         if VERBOSE:
#             traceback.print_exc(file=sys.stderr)
#         else:
#             print >>sys.stderr,err.msg

#         return 1

# if __name__ == "__main__":
#     sys.exit(main())
