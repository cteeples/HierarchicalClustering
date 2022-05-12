import numpy as np
import re
import math

from sklearn import cluster

class Person:
    def __init__ (self, height, weight, age, gender=None, distance=None):
        self.height = height
        self.weight = weight
        self.age = age
        self.gender = gender

class Cluster:
    def __init__(self, points):
        self.points = np.array(points)
        #print(points)
        self.distance = 0
        self.id = -1

    def addPoint(self, point):
        '''
        arr = np.array([[1,2], [3,4]])
        arr2 = np.array([[2, 3], [4,5]])
        arr = np.append(arr, arr2, axis=0)
        '''
        current_points = self.points
        point_as_list = np.array([point])
        # print(self.points)
        # print(np.append(current_points, point_as_list, axis=0))
        # print("\n")
        self.points = np.append(current_points, point_as_list, axis=0)

class HierarchicalClustering:
    def __init__(self, num_of_features, data, num_of_clusters, linkage_type = "single"):
        self.linkage_type = linkage_type
        self.num_of_features = num_of_features
        self.data = data
        self.num_of_clusters = num_of_clusters
        self.clusters = []
        self.average_cluster_purity = 0.0
        self.combine_index, self.delete_index = 0, 0 
        self.isStartingMatrix = True

        for item in data: self.clusters.append(Cluster(np.array([item])))
        self.temp_removed_cluster = self.clusters

        #initialize distance matrix to 0s
        cluster_count = len(self.clusters)
        self.distance_matrix = np.zeros((cluster_count, cluster_count))
    
    def getEuclideanDistance(self, point1, point2):
        point1 = point1[:self.num_of_features]
        point2 = point2[:self.num_of_features]
        return np.linalg.norm(point1-point2)

    def getMinimumDistance(self, cluster1, cluster2):
        distance_options = []

        for point_1st in cluster1.points:
            for point_2nd in cluster2.points:
                distance_options.append(self.getEuclideanDistance(point_1st, point_2nd))
        
        if self.linkage_type == "single": return min(distance_options)
        elif self.linkage_type == "complete": return max(distance_options)
        elif self.linkage_type == "average": return sum(distance_options) / len(distance_options)
        else:
            #default to single linkage
            print("incorrect linkage type specified")
            return min(distance_options)

    def mergeTwoClosestClusters(self, distanceMatrix):
        minimum = np.min(distanceMatrix[distanceMatrix != 0])
        index = np.where(distanceMatrix == minimum)
        #choose combine_index and delete_index index of first minimum returned by np.where
        self.combine_index, self.delete_index = index[0][0], index[1][0]
        for point in self.clusters[self.delete_index].points:
            self.clusters[self.combine_index].addPoint(point)

        self.clusters.remove(self.clusters[self.delete_index])

        #decrement combine_index if necessary to adjust for removed cluster
        if self.delete_index < self.combine_index: self.combine_index = self.combine_index - 1 

    def getDistanceMatrix(self):
        cluster_count = len(self.clusters)
        print(cluster_count)

        #creates a matrix where values are in top triangle and diagonal is 0s
        if self.isStartingMatrix:
            for i in range(cluster_count):
                for j in range(i, cluster_count):
                    minimum_distance = self.getMinimumDistance(self.clusters[i], self.clusters[j])
                    self.distance_matrix[i,j] = minimum_distance
            
            self.isStartingMatrix = False
            
            #some of the diagonals don't fill to 0 because of the way I implement the loops above. (on odd number values of clusters)
            # the code below fixes this:
            np.fill_diagonal(self.distance_matrix, 0)

        else:
            #delete the delete_index of the delete_index index and the combine_index of the delete_index index
            #this is because the delete_index index is used to delete a cluster in self.mergeTwoClosestClusters()
            column_deleted_from_distance_matrix = np.delete(self.distance_matrix, self.delete_index, 1)
            column_and_row_deleted_from_distance_matrix = np.delete(column_deleted_from_distance_matrix, self.delete_index, 0)
            self.distance_matrix = column_and_row_deleted_from_distance_matrix

            #update delete_index of combined cluster
            #stop at the diagonal of the matrix (i.e. where i = self.combine_index)
            i = 0
            while i != self.combine_index:  
                # print(i)
                # print(self.delete_index)
                minimum_distance = self.getMinimumDistance(self.clusters[i], self.clusters[self.combine_index])
                self.distance_matrix[i, self.combine_index] = minimum_distance
                i = i + 1
            
            #update combine_index of combined cluster (decrement j because the values of the matrix are in the top triangle)
            #stop at the diagonal of the matrix (i.e. whereji = self.combine_index)
            j = cluster_count - 1
            while j != self.combine_index:
                minimum_distance = self.getMinimumDistance(self.clusters[self.combine_index], self.clusters[j])
                self.distance_matrix[self.combine_index, j] = minimum_distance
                j = j - 1

    def buildClusters(self):
        while(len(self.clusters) > self.num_of_clusters):
            self.getDistanceMatrix()
            print(self.distance_matrix.shape)
            print(self.distance_matrix)
            self.mergeTwoClosestClusters(self.distance_matrix)

        self.setAveragePurity()

    def setAveragePurity(self):
        purities = []
        i = 1
        for cluster in self.clusters:
            print("Points in cluster " + str(i) + ": \n")
            for point in cluster.points:
                print(point)
            print("\n")

            Y = cluster.points[:,-1].tolist()
            classWithHighestCount = max(Y, key=Y.count)
            amountOfClassWithHighestCount = np.count_nonzero(np.array(Y) == classWithHighestCount)
            purity = amountOfClassWithHighestCount / len(cluster.points)
            purities.append(purity * (len(cluster.points) / len(self.data) ))
            print("Purity for cluster " + str(i) + "(size = " + str(len(cluster.points)) + ") : " + str(purity) + "\n")

            #set the id for the clusters to the integer class with highest count
            cluster.id = int(classWithHighestCount)
            
            i += 1

        print("Average purity for " + str(self.num_of_clusters) + " clusters: " + str(sum(purities) ) )
        print('\n')

        self.average_cluster_purity = sum(purities) 

if __name__ == "__main__":
    people = readData()
    data = []
    
    for person in people:
        data.append([person.height, person.weight, person.age, 1 if person.gender.strip() == 'W' else 0])

    np_data = np.array(data)
    X, Y = np_data[:,:-1], np_data[:,-1]

    hierarchicalClustering = HierarchicalClustering(3, np_data, 2, "single")

    hierarchicalClustering.buildClusters()

    hierarchicalClustering.printClusters()

    hierarchicalClustering = HierarchicalClustering(3, np_data, 4, "single")

    hierarchicalClustering.buildClusters()

    hierarchicalClustering.printClusters()

    hierarchicalClustering = HierarchicalClustering(3, np_data, 6, "single")

    hierarchicalClustering.buildClusters()

    hierarchicalClustering.printClusters()

    hierarchicalClustering = HierarchicalClustering(3, np_data, 8, "single")

    hierarchicalClustering.buildClusters()

    hierarchicalClustering.printClusters()

    hierarchicalClustering = HierarchicalClustering(3, np_data, 2, "complete")

    hierarchicalClustering.buildClusters()

    hierarchicalClustering.printClusters()

    hierarchicalClustering = HierarchicalClustering(3, np_data, 4, "complete")

    hierarchicalClustering.buildClusters()

    hierarchicalClustering.printClusters()

    hierarchicalClustering = HierarchicalClustering(3, np_data, 6, "complete")

    hierarchicalClustering.buildClusters()

    hierarchicalClustering.printClusters()

    hierarchicalClustering = HierarchicalClustering(3, np_data, 8, "complete")

    hierarchicalClustering.buildClusters()

    hierarchicalClustering.printClusters()
