#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Precode import *
import numpy

data = np.load('AllSamples.npy')


# In[2]:


k1,i_point1,k2,i_point2 = initial_S1('7088') # please replace 0111 with your last four digit of your ID


# In[3]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[4]:


print(i_point1[:, 0])
print(i_point1[:, 1])
print(len(data))


# In[5]:


import matplotlib.pyplot as plt

#Plotting the results
plt.scatter(i_point1[:, 0],i_point1[:, 1]  , color = 'red')
plt.scatter(i_point2[:, 0], i_point2[:, 1] , color = 'black')
plt.show()


# In[6]:


plt.scatter(data[:, 0], data[:, 1] , color = 'blue')
plt.show()


# In[7]:


# Euclidean distance
def dist(centroid, point):
    return np.sqrt((centroid[0]-point[0])**2 + (centroid[1]-point[1])**2)

def compute_sse(data, centroids, point_cluster):
    # Initialise SSE 
    sse = 0
    
    # Compute the squared distance for each data point and add. 
    for i,x in enumerate(data):
    	# Get the associated centroid for data point
        centroid = centroids[point_cluster[i]]
                
        # Compute the Distance to the centroid
        d = abs(dist(x, centroid))
        
        # Add to the total distance
        sse += d**2
    
#     sse /= len(data)
    return sse

# List to store SSE for each iteration 
sse_list = []

# array to hold each index point cluster
point_cluster = [0] * len(data)
# kmeans
def kmeans(k, centroids, data):
       
    while True:
        old_centriods = centroids.copy()
        # loop on all points
        for i, point in enumerate(data):
            # min distance
            min_d = 100
            # min cluster distance index
            min_cluster = 0
            # loop over centorids and compute distance and assign min distance to points
            for c, centroid in enumerate(centroids):
                d = dist(centroid, point)
                if min_d > d :
                    min_d = d
                    min_cluster = c
            # assgin points to clusters    
            point_cluster[i] = min_cluster
        # loop over to clsuter points into groups
        for c in range(len(centroids)):
            # Get all the data points belonging to a particular cluster
            cluster_data = [data[x] for x in range(len(data)) if point_cluster[x] == c]
            # Initialise the list to hold the new centroid
            new_centroids = [0] * len(centroids[0])
            # compute new centroids
            for new_centroid in range(2):
                dim_sum = [x[new_centroid] for x in cluster_data]
                dim_sum = sum(dim_sum) / len(dim_sum)
                new_centroids[new_centroid] = dim_sum
            centroids[c] = new_centroids
        # compute sse
        sse = compute_sse(data, centroids, point_cluster)
        sse_list.append(sse)
        # stop condition
        if np.array_equal(old_centriods, centroids):
            return centroids
    return centroids


# In[8]:


centroids = kmeans(k1, i_point1, data)
print (centroids)
print(sse_list[-1])
colors = ['red', 'blue', 'green', 'black', 'yellow']
for c in range(len(centroids)):
    cluster_members = [data[i] for i in range(len(data)) if point_cluster[i] == c]    
    cluster_members = np.array(cluster_members)
    plt.scatter(cluster_members[:, 0],cluster_members[:, 1]  , color = colors[c])
plt.scatter(centroids[:, 0],centroids[:, 1],marker = "x", s=150,linewidths = 5, zorder = 10, c="black")
plt.show()


# In[ ]:




