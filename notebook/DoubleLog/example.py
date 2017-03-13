import os
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from plotting import mapDat

tc = np.logspace(-4, -2, 15)
x = np.r_[-300.:301.:1]

ade = np.load('ATEMlineADE.npz')

t = ade['tCalc']
loc = ade['rxLoc']
ade = ade['data']*1e9

# Interpolate in time
adeT = interp1d(t, ade, kind='cubic', axis=0)(tc)

# Interpolate in space
ade = interp1d(loc[:,0], adeT, kind='cubic', axis=1)(x)

# Map it
adeM, ticks, tickLabels = mapDat(ade, 1e-9, stretch=2)
adeTM, ticks, tickLabels = mapDat(adeT, 1e-9, stretch=2)


# Plot it
fig = plt.figure(figsize=(8,4.5))
ax = fig.add_axes([0.11, 0.23, 0.86, 0.72])
ax.hlines(0., -300, 300, colors='0.5', linestyles='--')

plt.plot(x, adeM.T,  lw=2., c='k', ls='-')
plt.plot(loc[:,0], adeTM.T, c='k', marker='o', ms=3., ls='')

plt.plot(-900,-900,  lw=2., c='k', ls='-', marker='o',
        markersize=3., label='ADE')

ax.set_yticks(ticks)
ax.set_yticklabels(tickLabels)
ax.set_ylim(ticks.min(), ticks.max())
ax.set_xlim(x.min(), x.max())
ax.set_xlabel("Easting (m)")
ax.set_ylabel("$b_z$ (nT)")

handles, labels = ax.get_legend_handles_labels()
tmp = np.unique(labels, return_index=True)
labels = np.array(labels)[np.sort(tmp[1])].tolist()
handles = np.array(handles)[np.sort(tmp[1])].tolist()
legend = plt.legend(handles, labels, loc='upper center', 
           bbox_to_anchor=(0.5, -0.16), 
           ncol=4, fancybox=True,
           )

frame = legend.get_frame()
frame.set_facecolor('0.95')


fname = 'example.eps'
plt.savefig(fname, format='eps', dpi=1200)
os.system("open " + fname)
