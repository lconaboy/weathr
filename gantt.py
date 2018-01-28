import numpy as np
import matplotlib.pyplot as plt


wp = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8],
      [9, 9], [10, 10]]  # wp is the number of the work package
dur = np.array([4, 1, 4, 6, 4, 3, 10, 7, 1, 15],
               dtype=float)  # dur is the duration of the wp in days
start = np.array([0, 5, 15, 10, 12, 12, 14, 16, 24, 30],
                 dtype=float)  # start is the start day of the work package

# convert to weeks
dur = dur/7  
start = start/7

x = np.arange(0, 19)

for task in range(0, len(wp)):
    plt.plot([start[task], start[task] + dur[task]],
             wp[task], color='r', linewidth=10)
plt.axvline(max(x), color='r', linewidth=3)
labels = ['29/1', '5/2', '12/2', '19/2', '26/2', '5/3', '12/3', '19/3', '26/3',
          '2/4', '9/4', '16/4', '23/4', '30/4',
          '7/5', '14/5', '21/5', '28/5', '4/6']

ax = plt.gca()
plt.xticks(x, labels, rotation=45)
ax.invert_yaxis()

plt.xlim([0, max(x)])
plt.xlabel('w/c')
plt.ylabel('work package')
plt.show()
