import datetime
import numpy as np
import matplotlib.pyplot as plt
from util import *
from coverage_analysis_functions import *

# load the anomaly data and narrow to range
nino = nino_range('detrend.nino34.ascii.txt', 1., 2009., 1., 2018.)
# pick out consecutive anomalies
anoms = consecutive_anomalies(nino)

# now calculate three monthly means of data
region = 'capetown'
start = datetime.datetime.strptime('20092','%Y%m')
end = datetime.datetime.strptime('201711','%Y%m')
tmm = three_monthly_means(start, end, region)
stacked = stack_tmm_by_event(tmm, anoms)
yearly_mean = yearly_mean_from_tmm(tmm, anoms)
diffs = absolute_diffs(stacked, yearly_mean)
rel = abs_to_rel(diffs, yearly_mean)

plt.figure()
for i in range(0, 3):
    plt.errorbar(np.arange(12), rel[0][:, i], rel[1][:, i])
plt.legend(['EN', 'LN', 'N'])
x_labels('tm')
plt.show()
