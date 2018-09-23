import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import math

d = pd.read_csv("d.csv")
X = d[["Distance_Feature","Speeding_Feature"]].values

NUM_POINTS= len(X)
NUM_CLUSTERS = 4
PG_THRESH = 0.0001
MAX_ITER=30
NUM_TRIES = 10

def sqd(p1,p2):
	r = 0.0
	for a,b in zip(p1,p2):
		r+=(a-b)**2
	return r

def dpl(p,l):
	r = []
	for c in l:
		sd = sqd(p,c)
		r.append(sd)
	return r

def dMat(centroids,points):
	r = []
	for c in centroids:
		r.append(dpl(c,points))
	return np.transpose(r)

def meanD(c,dmat):
	r = 0
	for i in range(len(dmat)): r+=dmat[i][c[i]]
	return math.sqrt(r/len(dmat))

def initC(numClusters,points):

	firstCluster = points[np.random.randint(0,len(points))]
	c = [firstCluster]

	for _ in range(numClusters-1):
		dm = dMat(c,points)
		cids = np.argmin(dm,axis=1)
		minDs = []
		for i in range(len(points)): minDs.append(dm[i][cids[i]])
		pDist = minDs/np.sum(minDs)
		pI = np.random.choice(len(points),p=pDist)
		c.append(points[pI])
	return np.array(c)

def plot(points,clusters,clusterIDs):

	g = []
	for i in range(len(clusters)): g.append([])

	for i in range(len(points)):
		clid = clusterIDs[i]
		g[clid].append(points[i])

	colors = ['b','g','r','y']

	for i in range(len(clusters)):
		t = np.transpose(g[i])
		if(len(t)>0):
			xv = t[0]
			yv = t[1]
			plt.scatter(xv,yv,c=colors[i])

	plt.scatter(clusters[:,0],clusters[:,1],c='k')
	plt.show()

minMeanDistance = bestC = bestClust = None
mmdInit = False

for _ in range(NUM_TRIES):

	c = initC(NUM_CLUSTERS,X)

	distanceMatrix = dMat(c,X)
	clust = np.argmin(distanceMatrix,axis=1)
	prev = 0
	cur = meanD(clust,distanceMatrix)
	count = 0
	pG = 0
	init = False

	while(not init or (pG>PG_THRESH and count<MAX_ITER)):

		prev=cur

		d = dict()
		for i in range(NUM_CLUSTERS):d[i]=[]
		for i,cid in enumerate(clust): d[cid].append(X[i])

		for i in range(NUM_CLUSTERS):
			if(len(d[i])>0):
				d[i]=np.mean(d[i],axis=0)
			else:
				d[i]=c[i]
		newC = []
		for i in range(NUM_CLUSTERS):newC.append(d[i])

		c=np.array(newC)

		distanceMatrix = dMat(c,X)
		clust = np.argmin(distanceMatrix,axis=1)
		cur = meanD(clust,distanceMatrix)
		pG = -(cur-prev)/prev

		init = True
		count+=1
	if(not mmdInit or cur<minMeanDistance):
		minMeanDistance = cur
		bestC = c
		bestClust = clust
		mmdInit = True

print("Minimum Mean Distance: %f"%(minMeanDistance))
plot(X,bestC,bestClust)



