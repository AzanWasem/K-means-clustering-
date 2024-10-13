
("I wrote the code myself and had a friend with me, but I got the description written by ChatGPT.")

This code is very best if you are familier with python, numpy and maths function

K-Means Clustering Code Description 
This Python script implements a K-Means clustering algorithm using NumPy and scikit-learn libraries. The code follows an object-oriented approach with a KMeansModel class that encapsulates the main functionality of the algorithm. Below are the key components and functionalities of the code:

Imports
NumPy: For numerical operations and array manipulations.
Matplotlib: For data visualization (though plotting is currently commented out).
scikit-learn: To generate synthetic datasets and preprocess data.
Math Library: For mathematical functions, specifically for calculating the Euclidean distance.
Metrics: To evaluate clustering performance using confusion matrices and accuracy scores.
KMeansModel Class
Initialization (__init__):

Takes the number of clusters (K_clusters) and input data (X) as parameters.
Initializes random cluster centers and normalizes the input data using Min-Max scaling.
Distance Calculation:

GetEuclideanDistance(v1, v2): Computes the Euclidean distance between two vectors.
FindClosestCentre(v1): Identifies the index of the nearest cluster center to a given data point.
Cluster Mapping:

MapClosestCentre(): Assigns each data point to its closest cluster center, updating the class membership.
Normalization:

Normalize(): Scales the input features to the range [0, 1] to improve clustering performance.
Centroid Updating:

ChangeCentres(): Recalculates the positions of cluster centers by averaging the positions of data points assigned to each cluster.
Main Execution
Generates synthetic data using make_blobs, creating a dataset with 1000 samples, 10 features, and 2 clusters.
Instantiates the KMeansModel class and performs clustering by iteratively updating cluster centers and class assignments.
Evaluates clustering performance by comparing predicted classes to the actual labels using a confusion matrix and accuracy score.
Output
Displays the predicted classes alongside the actual labels in a concatenated array.
Prints the confusion matrix and accuracy percentage of the clustering results.


This is a code


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import confusion_matrix, accuracy_score

class KMeansModel():
    def __init__(self, K_clusters, X):
        self.K = K_clusters
        self.X = X
        self.Centres = np.random.rand(self.K, self.X.shape[1])
        self.Normalize()
        self.Classes = None
        self.N_Samples = self.X.shape[0]

    def GetEuclideanDistance(self, v1, v2):
        return sqrt(np.sum((v1-v2)**2))

    def FindClosestCentre(self, v1):
        # Loops through all centres and returns the index of closest centre
        ClosestCentreDistance = float("inf")
        ClosestCentreIndex = None
        for i in range(self.Centres.shape[0]):
            distance = self.GetEuclideanDistance(v1, self.Centres[i])
            if distance < ClosestCentreDistance:
                ClosestCentreDistance = distance
                ClosestCentreIndex = i
        return ClosestCentreIndex

    def MapClosestCentre(self):
        # This is done for each sample and creates a column vector where the 
        # element at index i of the vector corresponds to the closest centre of
        # the ith sample
        Vec = []
        for i in range(self.N_Samples):
            Vec.append(self.FindClosestCentre(self.X[i]))

        self.Classes = np.array(Vec).reshape((-1, 1))

    def Normalize(self):
        # Normalizing function to express values between 0 and 1 
        N = MinMaxScaler()
        self.X = N.fit_transform(self.X)

    def ChangeCentres(self):
        # Updates each centre to take on the mean position of each 
        # sample which takes on the corresponding class of the centre
        for i in range(self.Centres.shape[0]): # Loop through all the centres 
            if len(self.Classes[self.Classes == i]) > 0:
                self.Centres[i] = np.sum(
                    # Use boolean indexing to only take the rows which correspond to the ith centre 
                    self.X[np.array(self.Classes == i).reshape(1, -1)[0]], axis=0
                ) / len(self.Classes[self.Classes == i]) # Divide by the number of samples in the class 

if __name__ == "__main__":
    # Creating some testing data 
    k = 2
    TrainingX, TrainingY = make_blobs(
        n_samples=1000, n_features=10, cluster_std=4, centers=k)
    km = KMeansModel(k, TrainingX)
    
    km.MapClosestCentre()

    for i in range(50):
        km.ChangeCentres()
        km.MapClosestCentre()

    data = np.concatenate((km.X, km.Classes), axis=1)
    
    # Only uncomment if n_features = 2 
    # plt.scatter(data[:, : -2], data[:, 1: -1], c="r")
    # plt.scatter(km.Centres[:, [0]], km.Centres[:, [1]], c="b")
    # plt.show()
    # print(np.concatenate((km.X, km.Classes), axis=1))

    # This code either yields a very low or high accuracy in most cases, this is just
    # because this is an unsupervised learning model and that the names of classes 
    # are completely arbitrary. A better measure of accuracy is the confusion matrix, 
    # which shows classes and their frequencies 
    pred = km.Classes
    act = TrainingY.reshape(-1, 1)
    print(np.concatenate((pred, act), axis=1))
    print(confusion_matrix(pred, act))
    print(accuracy_score(pred, act) * 100, "%")
    
