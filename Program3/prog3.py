#Tu Vu
#Program 3
import numpy as np
import matplotlib.pyplot as pl
from copy import deepcopy
import sys

class kc_means(object):
    def __init__(self,filename, clusterCount, iteration):
        self.data = np.loadtxt(filename)
        self.clusterCount = clusterCount
        self.iteration =  iteration
        self.K = self.selectK() 
        self.coefficient = self.randomCoefficient()

    #Initial means
    def selectK(self):
        k = []
        index = np.random.choice(self.data.shape[0], self.clusterCount, replace = False)
        for i in index:
            k.append(self.data[i])
        return np.asarray(k)
        
    #Return a list of distance between a single data input to all the K means
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
           distances = self.EuclideDistance(self.data[i])  #Calculate the mean distance for each observation against all the K means
           group[np.argmin(distances)].append(self.data[i]) #argmin determin which cluster it belong based on how close it is
        
        return group

    #Updating the K centroid   
    #cluster argument is the return value from function assignment 
    def updateK(self, cluster):
        assert (len(self.K) == len(cluster)), "Check in update()" #Check if the length of K is equal to the cluster we sent in
        #Updating each centroid value
        for i in range(len(self.K)):
            clus = np.asarray(cluster[i]) ## Turn the list into a 2d array
            new = np.mean(clus, axis = 0) ## Calculate the mean up each cluster to be the new centroid
            self.K[i] = new
        return self.K

    #sum square error calculation
    def sse(self,cluster):
        sse = 0
        for i in range(len(self.K)):
            x = np.asarray(cluster[i])
            d = 0
            for j in range(x.shape[0]):
                d += (np.linalg.norm(x[j] - self.K[i]))**2 # <-- sum of squared error in each cluster
            sse += d # <-- sum of all the error in all clusters
        return sse

    #A little bit of hard coding 
    def kplot(self, cluster):
        for i in range(len(cluster)):
            x = np.asarray(cluster[i])
            pl.scatter(x.T[0], x.T[1], cmap = pl.get_cmap('rainbow'))
            pl.scatter(self.K.T[0], self.K.T[1], c = '#000000')
        pl.show()

    def kmeans(self):
        sse = []
        allCluster = []

        cluster = self.assignmentK() #Assigning the initial clusters
        self.kplot(cluster)

        for i in range(self.iteration):
            self.updateK(cluster)
            print(self.K)
            print()
            x = deepcopy(self.K)
            allCluster.append(x)
            cluster = self.assignmentK()
            sse.append(self.sse(cluster)) #Calculate the sum square error for each iteration
            self.kplot(cluster) 

        print("The ith iteration and its set of centroids that have the smallest sum of squared error:")
        smol = np.argmin(np.array(sse))
        print(smol + 1)
        print(allCluster[smol])


    ###################################################################
    #######################FUZZY-C MEANS SECTION#######################
    #Generate random coefficient for each observation 
    def randomCoefficient(self):
        C = []
        for i in range(self.data.shape[0]):
            coeff = np.random.uniform(0.0, 1.0, (self.clusterCount,))
            C.append(coeff)
        return np.asarray(C)

    #Centroid calculation for fuzzy-c means
    def cCentroid(self):
        centroid = []
        for i in range(self.clusterCount):
            a = (self.data.T * self.coefficient.T[i]).T 
            c = np.sum(a, axis = 0) / np.sum(self.coefficient.T[i])
            centroid.append(c)
        print(np.array(centroid))
        return np.asarray(centroid)

    #Calculate the distance of each observation from centroids
    #Remember to check this part
    def cDistance(self,centroid):
        distance = []
        for i in range(self.data.shape[0]):
            d = []
            for k in range(self.clusterCount):
                d.append(np.linalg.norm(self.data[i] - centroid[k])) 
            distance.append(np.asarray(d))
        return np.asarray(distance)

    #Check this garbage code...I developed cancer writing this function
    def updateCoeff(self, distances):
        for i in range(self.data.shape[0]):
            for j in range(self.clusterCount):
                c = 0
                n = distances[i][j]
                for k in range(self.clusterCount):
                    c += (n / distances[i][k]) ** (2 / (2 - 1)) # Check the m value in the equation
                self.coefficient[i][j] = 1 / c

    #Objective function of finding the smallest possible error
    def cErrors(self,centroid):
        error = 0
        for i in range(self.data.shape[0]):
            e = 0
            for j in range(self.clusterCount): #<-- check this part here
                e += self.coefficient[i][j] * (np.linalg.norm(self.data[i] - centroid[j]))**2
            error += e
        return error

    def cmeans(self):
        for i in range(self.iteration):
            centroid = self.cCentroid()
            distance = self.cDistance(centroid)
            self.updateCoeff(distance)
        self.cplot() 

    #Testing with plotting, this is for 3 cluster
    def cplot(self):
        for i in range(self.data.shape[0]):
            if np.argmax(self.coefficient[i]) == 0:
                pl.scatter(self.data[i][0], self.data[i][1], c = 'r')
            if np.argmax(self.coefficient[i]) == 1:
                pl.scatter(self.data[i][0], self.data[i][1], c = 'b')
            if np.argmax(self.coefficient[i]) == 2:
                pl.scatter(self.data[i][0], self.data[i][1], c = 'g')
            
            #pl.scatter(centroid.T[0], centroid.T[1], c = 'r')
        pl.show()

#The main function that read in the argument of how many cluster point
def main():
    choice = str(sys.argv[1])
    clusterCount = int(sys.argv[2])
    iteration = int(sys.argv[3])

    a = kc_means("cluster_dataset.txt", clusterCount, iteration)

    if choice == 'k':
        a.kmeans()
    elif choice == 'c':
        a.cmeans()
    
if __name__ == '__main__':
    main()