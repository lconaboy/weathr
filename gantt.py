import numpy as np
import matplotlib.pyplot as plt

wp = np.zeros(shape=(21, 2))
for n in range(0, 21):
    wp[n][0] = n  # wp is the number of the work package
    wp[n][1] = n

# dur is the duration of the wp in days
dur = np.array([7, 51, 10, 2, 4, 4, 12, 12, 10, 7, 12, 3,
                4, 5, 19, 7, 4, 4, 4, 13, 21], dtype=float)

# start is the start day of the work package
start = np.array([0, 7, 7, 7, 9, 13, 17, 29, 41, 51, 58, 58,
                  61, 65, 70, 70, 77, 81, 85, 89, 102], dtype=float)

# gen is the type of wp
gen = ['p', 'p', 'c', 'sc', 'sc', 'sc', 'c', 'c', 'c', 'c', 'p', 'c',
       'c', 'c', 'p', 'c', 'c', 'c', 'c', 'p', 'p']

# name of wp
name = ['1', '2', '2a', '2ai', '2aii', '2aiii', '2b', '2c', '2d', '2e',
        '3', '3a', '3b', '3c', '4', '4a', '4b', '4c', '4d', '5', '6']
# convert to weeks
dur = dur/7
start = start/7

x = np.arange(0, 19)

for task in range(0, len(wp)):
    tmp = gen[task]
    if tmp == 'p':
        plt.plot([start[task], start[task] + dur[task]],
                 wp[task], color=[0.75, 0.2, 0.2], linewidth=10)
    elif tmp == 'c':
        plt.plot([start[task], start[task] + dur[task]],
                 wp[task], color=[0.2, 0.75, 0.2], linewidth=8)
    else:
        plt.plot([start[task], start[task] + dur[task]],
                 wp[task], color=[0.2, 0.2, 0.75], linewidth=6)

plt.axvline((17+6/7), color='k', linewidth=2)
plt.axvline(6, color='k', linewidth=1, linestyle='dashed')
labels = ['29/1', '5/2', '12/2', '19/2', '26/2', '5/3', '12/3', '19/3', '26/3',
          '2/4', '9/4', '16/4', '23/4', '30/4',
          '7/5', '14/5', '21/5', '28/5', '4/6']

ax = plt.gca()
plt.xticks(x, labels, rotation=45)
plt.yticks(np.arange(0, len(gen)), name)
ax.invert_yaxis()

plt.xlim([0, max(x)])
plt.grid('on')
plt.xlabel('w/c')
plt.ylabel('work package')
plt.tight_layout()
plt.savefig('gantt.pdf', dpi=400) 
plt.show()
