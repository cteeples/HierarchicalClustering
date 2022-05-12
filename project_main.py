import clustering
import numpy as np
import knn

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

hierarchicalClustering = clustering.HierarchicalClustering(7, training_data, 3, "average")
hierarchicalClustering.buildClusters()

knn_model = knn.KNN(hierarchicalClustering, test_data, 1, "average", "euclidean")
test_data_predicted_Y = np.array(knn_model.Y)

print("Test data:")
print(test_data_actual_Y)
print("\n")
print("KNN Prediction using average linkage with euclidean distance:")
print(test_data_predicted_Y)
print("\n")
print("Accuracy:")
print(getAccuracy(test_data_actual_Y, test_data_predicted_Y))

