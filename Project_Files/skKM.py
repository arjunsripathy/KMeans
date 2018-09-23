import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

d = pd.read_csv("d.csv")
X = d[["Distance_Feature","Speeding_Feature"]].values

n = len(X)

nc = 4
km = KMeans(n_clusters=nc)
ci = km.fit_predict(X)

g = []
for i in range(nc): g.append([])

for i in range(n):
	c = ci[i]
	g[c].append(X[i])

colors = ['b','g','r','y']

for i in range(nc):
	t = np.transpose(g[i])
	xv = t[0]
	yv = t[1]
	plt.scatter(xv,yv,c=colors[i])

plt.show()