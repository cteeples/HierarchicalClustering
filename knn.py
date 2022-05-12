import numpy as np
import random
import clustering
from math import *
from decimal import Decimal

class KNN:
    def __init__(self, cluster_model, X, k, linkage_type = "single", distance_type = "euclidean", num_of_features = 7):
        self.clusters = cluster_model.clusters
        self.X = X
        self.k = k
        self.linkage_type = linkage_type
        self.distance_type = distance_type
        self.num_of_features = num_of_features
        self.Y = []

        self.setY()
    
    def p_root(self, value, root):  
        root_value = 1 / float(root)
        return round (Decimal(value) ** Decimal(root_value), 3)
 
    def getMinkowskiDistances(self, cluster_points, point, p = 3):
        return [(self.p_root(sum(pow(abs(val1-val2), p) for val1, val2 in zip(point, cluster_point)), p)) for cluster_point in cluster_points]

    def getEuclideanDistances(self, cluster_points, point):
        return [np.linalg.norm(cluster_point - point) for cluster_point in cluster_points]
    
    def getManhattanDistances(self, cluster_points, point):
        return [sum(abs(val1-val2) for val1, val2 in zip(point, cluster_point)) for cluster_point in cluster_points]

    def getDistance(self, cluster, point):
        distances = []
        if self.distance_type == "euclidean": distances = self.getEuclideanDistances(cluster.points[:,:self.num_of_features], point[:self.num_of_features])
        elif self.distance_type == "manhattan": distances = self.getManhattanDistances(cluster.points[:,:self.num_of_features], point[:self.num_of_features])
        elif self.distance_type == "minkowski": distances = self.getMinkowskiDistances(cluster.points[:,:self.num_of_features], point[:self.num_of_features])
        else:
            #default to euclidean distance
            print("incorrect distance type specified")
            distances = self.getEuclideanDistances(cluster, point)

        if self.linkage_type == "single": return min(distances)
        elif self.linkage_type == "complete": return max(distances)
        elif self.linkage_type == "average": return sum(distances) / len(distances)
        else:
            #default to single linkage
            print("incorrect linkage type specified")
            return min(distances)

    def setY(self):
        Y = []

        for x in self.X:
            classOneCount, classTwoCount, classThreeCount = 0, 0, 0

            for cluster in self.clusters:
                dist = self.getDistance(cluster, x)
                cluster.distance = dist

            clusters = sorted(self.clusters, key=lambda clusters: clusters.distance)

            for i in range(self.k):
                # print(clusters[i].gender)
                if clusters[i].id == 1: classOneCount += 1
                elif clusters[i].id == 2: classTwoCount += 1
                elif clusters[i].id == 3: classThreeCount += 1

            if (classOneCount > classTwoCount and classOneCount > classTwoCount): Y.append(float(1))
            elif (classTwoCount > classOneCount and classTwoCount > classThreeCount): Y.append(float(2))
            elif (classThreeCount > classOneCount and classThreeCount > classTwoCount): Y.append(float(3))

            else:
                randomNum = int(random.random() * 100)
                # print(randomNum)
                if randomNum % 3 == 0: Y.append(float(3))
                elif randomNum % 2 == 0: Y.append(float(2))
                else: Y.append(float(1))
        
        self.Y = Y 
