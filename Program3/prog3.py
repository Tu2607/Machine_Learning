#Tu Vu
#Program 3

import numpy as np
import math
import collections
import matplotlib.pyplot as pl
import sys

class kc_means(object):
    def __init__(self,filename, clusterCount, iteration):
        self.data = np.loadtxt(filename)
        self.clusterCount = clusterCount
        self.iteration =  iteration

        self.K = self.selectK() 

    #Initial means
    def selectK(self):
        k = []
        index = np.random.choice(self.data.shape[0], self.clusterCount, replace = False)
        for i in index:
            k.append(self.data[i])

        return np.asarray(k)

    #Return a list of distance between the data input to all the K means
    def EuclideDistance(self, data):
        distance = []
        for i in range(len(self.K)):
            d = np.linalg.norm(data -  self.K[i])
            distance.append(d)
        return np.asarray(distance)

    #Assigning each observation to a class
    def assignmentK(self):
        group = [[] for _ in range(self.clusterCount)] #Generate the lists for grouping, based on clusterCount

        #Go through each observation and assign it a cluster
        for i in range(self.data.shape[0]):
           distances = self.EuclideDistance(self.data[i])  #Calculate the mean distance for each observation against the Ks
           group[np.argmin(distances)].append(self.data[i]) #argmin determin which cluster it belong
        
        return group

    #Updating the K centroid   
    #cluster is the return value from function assignment 
    def updateK(self, cluster):
        assert (len(self.K) == len(cluster)), "Check in update()" #Check if the length of K is equal to the cluster we sent in
        #Updating each centroid value
        for i in range(len(self.K)):
            clus = np.asarray(cluster[i]) ## Turn the list into a 2d array
            #print(clus.shape)
            new = np.sum(clus, axis = 0) / clus.shape[0] # <-- CHECK MAH MATH HEREEEEE!!!!!
            self.K[i] = new
        #print()
        return self.K

    #sum square error calculation
    def sse(self,cluster):
        sse = 0
        for i in range(len(self.K)):
            x = np.asarray(cluster[i])
            d = 0
            for j in range(x.shape[0]):
                d += (x[j] - self.K[i])**2 # <-- sum of error in each cluster
            sse += d # <-- sum of all the error in all clusters
            
        return np.asarray(sse) 

    #Add plotting here 
    #Still need to add in the sum square error calculation
    def kmeans(self):
        allK = [[] for _ in range(self.iteration)] #Might not need this if I come up with a better way
        sse = []
        pl.scatter(self.data.T[0], self.data.T[1])
        pl.scatter(self.K.T[0], self.K.T[1], c = 'r')
        pl.show()
        cluster = self.assignmentK() #Assigning the initial clusters

        for i in range(self.iteration):
            allK[i].append(self.updateK(cluster))
            cluster = self.assignmentK()
            sse.append(self.sse(cluster)) #Calculate the sum square error 

            pl.scatter(self.data.T[0], self.data.T[1])
            pl.scatter(self.K.T[0], self.K.T[1], c = 'r')
            pl.show()
        
        print(sse)



#The main function that read in the argument of how many cluster point
def main():
    choice = str(sys.argv[1])
    clusterCount = int(sys.argv[2])
    iteration = int(sys.argv[3])

    a = kc_means("cluster_dataset.txt", clusterCount, iteration)

    if choice == 'k':
        a.kmeans()
    elif choice == 'c':
        print("Wait here")
    
if __name__ == '__main__':
    main()