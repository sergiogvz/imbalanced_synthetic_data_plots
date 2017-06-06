'''
It generates the synthetic data and executes all the sampling methods of the [imblearn package](http://contrib.scikit-learn.org/imbalanced-learn/index.html
The parameters goes as followed:

> 1st parameter (*div*): shape of the chess board.
> 2nd parameter (*N*): number of instances for the balanced dataset (N/2 for each class).
> 3rd parameter (*per*): percentage of instances that conform the imbalanced data set.
'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule,OneSidedSelection,InstanceHardnessThreshold
from imblearn.combine import SMOTEENN,SMOTETomek
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
Xo=[]
Yo=[]
#Generation of the Synthetic data
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

	Xo.append(x)
	Yo.append(y)
	if y==0 or (y == 1 and np.random.uniform(0,1) < per):
		X.append(x)
		Y.append(y)


X=np.array(X)
y=np.array(Y)
Xo=np.array(Xo)
yo=np.array(Yo)


f=open('chess'+str(div)+'x'+str(div)+'_n'+str(N)+'_'+str(per)+'.txt','w')

for xw, yw  in zip(X,y):
	f.write(str(xw[0])+" "+str(xw[1])+" "+str(float(yw))+"\n")


n_classes = 2
plot_colors = ['#000000','#FFFFFF']
plot_step = 0.02

cmap_blw = ListedColormap(['#000000','#FFFFFF'])
cmap_blw_light = ListedColormap(['#6E6E6E','#FFFFFF'])


names=['ADASYN','SMOTE','NCL','OSS','IHT', 'SMOTEENN','SMOTETomek']
methods=[ADASYN(random_state=RANDOM_SEED),SMOTE(random_state=RANDOM_SEED),NeighbourhoodCleaningRule(random_state=RANDOM_SEED,n_neighbors=20),OneSidedSelection(random_state=RANDOM_SEED,n_neighbors=1,n_seeds_S=100),InstanceHardnessThreshold(random_state=RANDOM_SEED),SMOTEENN(random_state=RANDOM_SEED),SMOTETomek(random_state=RANDOM_SEED)]

x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

# Train with Original set 
clf = DecisionTreeClassifier(min_samples_leaf=2).fit(Xo, yo)

# Prediction boundaries
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=cmap_blw_light)

plt.axis("tight")

# Plot the original training points
for i, color in zip([0.,1.], plot_colors):
    idx = np.where(yo == i)
    plt.scatter(Xo[idx, 0], Xo[idx, 1], c=color, cmap=cmap_blw )

plt.title('Chess'+str(div)+'x'+str(div)+' Original data-set')
plt.legend()
plt.savefig('plots/chess'+str(div)+'x'+str(div)+'_n'+str(N)+'_1.0'+'DT.eps')
plt.show()



# Train with Imbalanced set 
clf = DecisionTreeClassifier(min_samples_leaf=2).fit(X, y)

# Prediction boundaries
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=cmap_blw_light)

plt.axis("tight")

# Plot the training points
for i, color in zip([0.,1.], plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=cmap_blw )

plt.title('Chess'+str(div)+'x'+str(div)+' Imbalanced data-set')
plt.legend()
plt.savefig('plots/chess'+str(div)+'x'+str(div)+'_n'+str(N)+'_'+str(per)+'DT.eps')
plt.show()




# Plots of the different sampling methods
for name, sampler in zip(names, methods):
	Xr,yr = sampler.fit_sample(X,y)
	# Train with sampled set 
	clf = DecisionTreeClassifier(min_samples_leaf=2).fit(Xr, yr)

	# Prediction boundaries
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	cs = plt.contourf(xx, yy, Z, cmap=cmap_blw_light)

	plt.axis("tight")

	# Plot sampled training points
	for i, color in zip([0.,1.], plot_colors):
		idx = np.where(yr == i)
		plt.scatter(Xr[idx, 0], Xr[idx, 1], c=color, cmap=cmap_blw )

	plt.title(name)
	plt.legend()
	plt.savefig('plots/'+name+str(div)+'x'+str(div)+'_n'+str(N)+'_'+str(per)+'DT.eps')
	plt.show()

