''' 
It plots the results obtained of the method of the file given as parameter.
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
import sys

RANDOM_SEED = 123123  # fix the seed on each iteration
np.random.seed(RANDOM_SEED)
name=sys.argv[1]
f1=open(name,'r')
data1 = []
for eachLine in f1:
	newLine = eachLine.replace("\"", "")
	line = [float(x) for x in newLine.split(' ')] 
	data1.append(line)

data1 = np.array(data1)

Xr=data1[:,0:2]
yr=data1[:,-1]

n_classes = 2
plot_colors = ['#000000','#FFFFFF']
plot_step = 0.02

cmap_blw = ListedColormap(['#000000','#FFFFFF'])
cmap_blw_light = ListedColormap(['#6E6E6E','#FFFFFF'])



clf = DecisionTreeClassifier(min_samples_leaf=2).fit(Xr, yr)

x_min, x_max = 0 - 0.05, 1 + 0.05
y_min, y_max = 0 - 0.05, 1 + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=cmap_blw_light)

plt.axis("tight")

# Plot the undersampled training points
for i, color in zip([0,1], plot_colors):
	idx = np.where(yr == i)
	plt.scatter(Xr[idx, 0], Xr[idx, 1], c=color, cmap=cmap_blw )

plt.title(name.replace(".txt", ""))
plt.legend()
plt.savefig('plots/'+name+'DT.eps')
plt.show()

