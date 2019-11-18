# FIle to evaluate the result of crown segmentation methods.
# First, a function that receives two binary masks with the treetops as small circles
# Then transforms them into a list of points and then computes hausdorf distance between the point segmentations

import sys
from skimage import measure
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import KDTree
import numpy as np

def borderPoint(image,point):
    margin=100
    top1=image.shape[0]
    top2=image.shape[1]

    return point[0]<margin or (top1-point[0])<margin or point[1]<margin or (top2-point[1])<margin


# Function to take a binary image and output the center of masses of its connected regions
def listFromBinary(fileName):
    #open filename
    im=cv2.imread(fileName,cv2.IMREAD_GRAYSCALE)
    if im is None: return []
    else:
        mask = cv2.threshold(255-im, 40, 255, cv2.THRESH_BINARY)[1]

        #compute connected components
        numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)
        #print("crownSegmenterEvaluator, found "+str(numLabels)+" "+str(len(centroids))+" points for file "+fileName)

        #im2 = 255 * np. ones(shape=[im.shape[0], im.shape[1], 1], dtype=np. uint8)

        #print(" listFromBinary, found  "+str(len(centroids)))
        #print(centroids)

        newCentroids=[]
        for c in centroids:
            if not borderPoint(im,c):newCentroids.append(c)
        #print(" listFromBinary, refined  "+str(len(newCentroids)))
        #print(newCentroids)

        return newCentroids

def hausdorfDistance(u,v): # computes Hausdorf distance between two lists of point
    if len(u)==0 or len(v)==0: return -1
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

def matchedPercentage(list1,list2,epsilon):
    if len(list1)==0 or len(list2)==0: return -1

    newList1=np.asarray([[x,y] for x,y in list1 ])
    newList2=np.asarray([[x,y] for x,y in list2 ])

    kdt = KDTree(newList2, leaf_size=30, metric='euclidean')
    dist,ind=kdt.query(newList1, k=1)
    #print(dist)
    count=0
    for d in dist:
        if d<epsilon:count+=1

    return 100*(count/len(list1))

def main(argv):
    # argv[1] contains the distance method (0 hausdorff, 1, matched point percentage)
    # argv[2],argv[3] contains the names of the files with the first  ands second mask
    # Further parameters may contain specific information for some methods

    option=int(argv[1])
    file1=argv[2]
    file2=argv[3]

    #first, turn the binary masks of files 1 and 2 into lists of points
    list1=listFromBinary(file1)
    list2=listFromBinary(file2)

    # Now, compute the distance between sets indicated by the option
    if option == 0: # compute hausdorff distance between two masks
        # Second, compute hausdorff distance
        print(format(hausdorfDistance(list1,list2),'.2f'),end=" ")
    elif option==1: #number of matched points, in this case we need one extra parameter epsilon
        epsilon=float(argv[4])
        # The first file must be the ground truth
        print(format(matchedPercentage(list1,list2,epsilon),'.2f'),end=" ")
    elif option==2: # point difference
    # the ground truth file should be the first
        realPointsNumber=len(list1)
        predictedPointsNumber=len(list2)
        #print("real "+str(realPointsNumber)+" predicted "+str(predictedPointsNumber))
        print(format(100*(realPointsNumber-predictedPointsNumber)/realPointsNumber,'.2f'),end=" ")
    elif option==3:
        #simply count point
        print(str(len(list1))+" "+str(len(list2)))
    else: raise ("crownSegmenterEvaluator, Wrong option")


if __name__ == "__main__":
    main(sys.argv)
