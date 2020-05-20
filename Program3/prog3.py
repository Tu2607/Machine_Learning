#Tu Vu
#Program 3

import numpy as np
import math
import collections
import sys

class kc_means(object):
    def __init__(self,filename, clusterCount, iteration):
        self.data = self.loadfile(filename)
        self.clusterCount = clusterCount
        self.iteration =  iteration
        self.K = self.selectK() 

    def loadfile(self,filename):
        data = np.loadtxt(filename)
        return data

    #Note for this, not sure if np.asarray will be needed
    def selectK(self):
        k = []
        index = np.random.choice(self.data.shape[0], self.clusterCount, replace = False)
        for i in index:
            k.append(self.data[i])

        return np.asarray(k)

    #Return a list of distance between the data input to all the K values
    def EuclideDistance(self, data):
        distance = []
        for i in range(len(self.K)):
            d = np.linalg.norm(data -  self.K[i])
            distance.append(d)
        return np.asarray(distance)

    def classification(self, distance):
        cluster = 0
        k = distance[0] 

        for i in range(len(distance)):
            if k < distance[i]: #I know it's a bit redundant but just bear with it for now
                k = distance[i]
                cluster = i

        return cluster #return the cluster index that observation belong to

    #Assigning each observation to a class
    def assignmentK(self):
        K = [[] for _ in range(self.clusterCount)]

        for i in range(self.data.shape[0]):
           distances = self.EuclideDistance(self.data[i])  #Calculate the mean distance for each observation against the Ks
           K[self.classification(distances)].append(self.data[i])
        
        return K

    #Updating the K centroid   
    #cluster is the return value from function assignment 
    def updateK(self, cluster):
        assert (len(self.K) == len(cluster)), "Check in update()" #Check if the length of K is equal to the cluster we sent in
        for i in range(len(self.K)):
            clus = np.asarray(cluster[i]) ## Turn the list into a 2d array
            meanx = np.mean(clus.T[0]) ##I know this is hard coding
            meany = np.mean(clus.T[1])
            self.K[i] = [meanx,meany]
    
    def kmeans(self):
        for i in range(self.iteration):
            cluster = self.assignmentK() #Return lists of cluster
            self.updateK(cluster)



#The main function that read in the argument of how many cluster point
def main():
    clusterCount = sys.argv[1]
    iteration = sys.argv[2]
    a = kc_means("cluster_dataset.txt", clusterCount, iteration)
    

if __name__ == '__main__':
    main()