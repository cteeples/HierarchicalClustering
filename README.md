# Instructions for Running the Project

**Version 1.0.0** 

Below is a description of the project files and instructions to run them.

The files were run using an Python 3 Anaconda Distribution. Instructions might vary if your distribution is different. If your system has not been set to default to Python 3, you may need to substitute the 'python' command with 'python3'.

## clustering_statistics.py and knn_statistics.py

### clustering_statistics.py

This file is used to generate the graphs for the various clustering options - including the 3 linkage methods and a number of clustering counts. 

To run this program and see the statistics, run 
```
python clustering_statistics.py
```
### knn_statistics.py

This file is used to generate the graphs for the various knn parameters - including the linkage types and the distance methods (euclidean, manhattan, and minkowski)

To run this program and see the statistics, run 
```
python knn_statistics.py
```
## clustering.py and knn.py

These two files contain the HierarchicalClustering and KNN classes. These two classes contain the algorithms used for the project in project_main.py.

These files are not meant to be run directly.

## project_main.py

This file is the main project. It utilizes the HierarchicalClustering class to run the hierarchical agglomerative clustering algorithm on the seeds dataset. The file takes every 10th data item as test_data and uses the rest of the data as training data.

Two run the main project and see the resulting clusters and predictions of the knn algorithm, run 
```
python project_main.py
```