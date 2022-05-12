import knn
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

def getAccuracy(actual_Y, predicted_Y):
    correct_count = 0
    for i in range(len(actual_Y)):
        if actual_Y[i] == predicted_Y[i]: correct_count = correct_count + 1
    
    return correct_count / len(actual_Y)

data = readData()

training_data, test_data = [], []

for i in range(len(data)):
    if i % 10 == 0: test_data.append(data[i])
    else: training_data.append(data[i])

training_data, test_data = np.array(training_data), np.array(test_data)

test_data_actual_Y = test_data[:,-1]

graph_X, graph_Y_euclidean, graph_Y_manhattan, graph_Y_minkowski = ['complete', 'average', 'single'], [], [], []
hierarchicalClustering = clustering.HierarchicalClustering(7, training_data, 3, "average")
hierarchicalClustering.buildClusters()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Accuracy By Distance Type')

for linkage_type in graph_X:
    knn_model = knn.KNN(hierarchicalClustering, test_data, 1, linkage_type, "euclidean")
    test_data_predicted_Y = np.array(knn_model.Y)
    print(test_data_predicted_Y)
    graph_Y_euclidean.append(getAccuracy(test_data_actual_Y, test_data_predicted_Y))
    print(f"{linkage_type} linkage with euclidean distance:")
    print(getAccuracy(test_data_actual_Y, test_data_predicted_Y))

ax1.set(xlabel='Linkage Type', ylabel='Accuracy For Euclidean Distance')

ax1.plot(graph_X, graph_Y_euclidean)

for linkage_type in graph_X:
    knn_model = knn.KNN(hierarchicalClustering, test_data, 1, linkage_type, "manhattan")
    test_data_predicted_Y = np.array(knn_model.Y)
    graph_Y_manhattan.append(getAccuracy(test_data_actual_Y, test_data_predicted_Y))
    print(f"{linkage_type} linkage with manhattan distance:")
    print(getAccuracy(test_data_actual_Y, test_data_predicted_Y))

ax2.set(xlabel='Cluster Count', ylabel='Average Purity For Manhattan Distance')

ax2.plot(graph_X, graph_Y_manhattan)

for linkage_type in graph_X:
    knn_model = knn.KNN(hierarchicalClustering, test_data, 1, linkage_type, "minkowski")
    test_data_predicted_Y = np.array(knn_model.Y)
    graph_Y_minkowski.append(getAccuracy(test_data_actual_Y, test_data_predicted_Y))
    print(f"{linkage_type} linkage with minkowski distance:")
    print(getAccuracy(test_data_actual_Y, test_data_predicted_Y))

ax3.set(xlabel='Cluster Count', ylabel='Average Purity For Minkowski Distance')

ax3.plot(graph_X, graph_Y_minkowski)

plt.show()
