import itertools
import glob

# find it all
def find_incomplete_data():
    available_dates = {}
    
    for fname in glob.glob("./data/eumetsat/*.png"):
        date = os.path.basename(fname)[0:len("YYYYMMDDHHMM")]
        if date not in available_dates: 
            available_dates[date] = 1
        else: 
            available_dates[date] += 1

    d = [k for (k, v) in available_dates.items() if v != 3]

    f = [glob.glob('./data/eumetsat/{}_*.png'.format(l)) for l in d]
    
    return list(itertools.chain.from_iterable(f))

# delete it all
for f in find_incomplete_data():
    os.remove(f)
