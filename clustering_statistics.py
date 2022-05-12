import clustering
import numpy as np
import matplotlib.pyplot as plt

def readData():
    result = []
    with open('seeds_dataset.txt') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split( )
            result.append([float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]), float(data[7])])
    
    return np.array(result)

data = readData()

graph_X, graph_Y_complete, graph_Y_average, graph_Y_single = [], [], [], []

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Average Purity For Clustering')

for i in range (1, 12): graph_X.append(i)

for i in range(1, 12):
    hierarchicalClustering = clustering.HierarchicalClustering(7, data, i, "complete")
    hierarchicalClustering.buildClusters()
    graph_Y_complete.append(hierarchicalClustering.average_cluster_purity)

ax1.set(xlabel='Cluster Count', ylabel='Average Purity For Complete Linkage')

ax1.plot(graph_X, graph_Y_complete)

for i in range(1, 12):
    hierarchicalClustering = clustering.HierarchicalClustering(7, data, i, "average")
    hierarchicalClustering.buildClusters()
    graph_Y_average.append(hierarchicalClustering.average_cluster_purity)

ax2.set(xlabel='Cluster Count', ylabel='Average Purity For Average Linkage')

ax2.plot(graph_X, graph_Y_average)

for i in range(1, 12):
    hierarchicalClustering = clustering.HierarchicalClustering(7, data, i, "single")
    hierarchicalClustering.buildClusters()
    graph_Y_single.append(hierarchicalClustering.average_cluster_purity)

ax3.set(xlabel='Cluster Count', ylabel='Average Purity For Single Linkage')

ax3.plot(graph_X, graph_Y_single)

#plt.xlim([1, 10])
plt.show()
