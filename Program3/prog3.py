#Tu Vu
#Program 3

import numpy as np
import math
import collections
import matplotlib.pyplot as pl
import sys

class kc_means(object):
    def __init__(self,filename, clusterCount, iteration):
        self.data = self.loadfile(filename)
        self.clusterCount = clusterCount
        self.iteration =  iteration

        self.K = self.selectK() 
        self.store = []

    def loadfile(self,filename):
        data = np.loadtxt(filename)
        return data

    #Initial means
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
        group = [[] for _ in range(self.clusterCount)]

        for i in range(self.data.shape[0]):
           distances = self.EuclideDistance(self.data[i])  #Calculate the mean distance for each observation against the Ks
           group[self.classification(distances)].append(self.data[i])
        
        return group

    #Updating the K centroid   
    #cluster is the return value from function assignment 
    def updateK(self, cluster):
        assert (len(self.K) == len(cluster)), "Check in update()" #Check if the length of K is equal to the cluster we sent in
        #Updating each centroid value
        for i in range(len(self.K)):
            if cluster[i]:  #To make sure that there is something in the cluster
                clus = np.asarray(cluster[i]) ## Turn the list into a 2d array
                new = np.sum(clus, axis = 0) / clus.shape[0] # <-- CHECK MAH MATH HEREEEEE!!!!!
                self.K[i] = new


    #Add plotting here 
    #Still need to add in the sum square error calculation
    def kmeans(self):
        pl.scatter(self.data.T[0], self.data.T[1])
        pl.scatter(self.K.T[0], self.K.T[1], c = 'r')
        pl.show()
        for i in range(self.iteration):
            cluster = self.assignmentK() #Return lists of cluster
            self.updateK(cluster)
            pl.scatter(self.data.T[0], self.data.T[1])
            pl.scatter(self.K.T[0], self.K.T[1], c = 'r')
            pl.show()
            #Calculate Sum Square error here?


#The main function that read in the argument of how many cluster point
def main():
    clusterCount = int(sys.argv[2])
    iteration = int(sys.argv[3])
    choice = str(sys.argv[1])
    a = kc_means("cluster_dataset.txt", clusterCount, iteration)

    if choice == 'k':
        a.kmeans()
    elif choice == 'c':
        print("Wait here")
    

if __name__ == '__main__':
    main()