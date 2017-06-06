'''
It generates the synthetic data and executes MWMOTE method ([MWMOTE GitHub repo](https://github.com/yen-von/MWMOTE)).
The parameters goes as followed:

> 1st parameter (*div*): shape of the chess board.
> 2nd parameter (*N*): number of instances for the balanced dataset (N/2 for each class).
> 3rd parameter (*per*): percentage of instances that conform the imbalanced data set.
'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
import MWMOTE
import sys

RANDOM_SEED = 123123  # fix the seed on each iteration
np.random.seed(RANDOM_SEED)

# Parameters
div = int(sys.argv[1])
N = int(sys.argv[2])
per = float(sys.argv[3])

lim = []
for i in range(1, div):
	lim.append(i/float(div))
print(lim)

X=[]
Y=[]
for i in range(0,N):
	x = [0,0]
	x[0] = np.random.uniform(0,1)
	x[1] = np.random.uniform(0,1)

	y=0

	ind = 0
	while ind < len(lim) and x[0] > lim[ind]:
		ind+=1

	y=(ind)%2


	ind = 0
	while ind < len(lim) and x[1] > lim[ind]:
		ind+=1

	if y == 0:
		y = (ind)%2
	else:
		y = (ind-1)%2

	if y==0 or (y == 1 and np.random.uniform(0,1) < per):
		X.append(x)
		Y.append(y)

Xl=X
X=np.array(X)
yl=[]
for i in Y:
	if i==1: 
		yl.append(-1) 
	else:
		yl.append(1) 

y=np.array(y)

n_classes = 2
plot_colors = ['#FFFFFF','#000000']
plot_step = 0.02

cmap_blw = ListedColormap(['#FFFFFF','#000000'])
cmap_blw_light = ListedColormap(['#FFFFFF','#6E6E6E'])

Nn=400
name='MWMOTE'
Xr,yr = MWMOTE.MWMOTE(Xl, yl, Nn, return_mode = 'append')
Xr=np.array(Xr)
yr=np.array(yr)



clf = DecisionTreeClassifier(min_samples_leaf=2).fit(Xr, yr)

x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=cmap_blw_light)

plt.axis("tight")

# Plot the undersampled training points
for i, color in zip([-1,1], plot_colors):
	idx = np.where(yr == i)
	plt.scatter(Xr[idx, 0], Xr[idx, 1], c=color, cmap=cmap_blw )

plt.title('MSMOTE #Sythetic='+str(Nn))
plt.legend()
plt.savefig('plots/'+name+str(div)+'x'+str(div)+'_n'+str(N)+'_'+str(per)+'DT.eps')
plt.show()

