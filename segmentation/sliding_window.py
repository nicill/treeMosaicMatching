# import the necessary packages
import argparse
import time
import cv2
import os
from matplotlib import pyplot as plt
import sys
import numpy as np

# peak local max imports
from skimage import data, img_as_float
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

# Clustering imports
from sklearn.cluster import KMeans, SpectralClustering
import skfuzzy as skf
from sklearn.mixture import GaussianMixture
from skimage.transform import pyramid_expand, pyramid_reduce, resize

from sklearn.cluster import DBSCAN

# for the function to turn binary files into lists of points
import crownSegmenterEvaluator as CSE
from sklearn.neighbors import KDTree
import floorExtractor as fe

def sliding_window(image, stepSize, windowSize, allImage=False):
    if allImage: yield(0,0,image[:,:])
    else:
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def resize(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def dbscan(im,eps,minS=10):
    labimg = cv2.medianBlur(im,5)
    n = 0
    while(n<4):
        labimg = cv2.pyrDown(labimg)
        n = n+1

    #feature_image=np.reshape(labimg, [-1, 3])
    rows, cols = labimg.shape

    db = DBSCAN(eps=eps, min_samples=minS, metric = 'euclidean',algorithm ='auto')
    db.fit(labimg.flatten().reshape(rows*cols, 1))
    labels = db.labels_
    nClusters=len(set(labels)) - (1 if -1 in labels else 0)
    return computeClusterMaxoids(im,np.reshape(labels, [rows, cols]),nClusters)


def k_means_clustering(inp_image, n_clusters=2):
    if inp_image is None:
        print("Empty Input. Exiting")
        return None
    # Create K Means Model
    k_means = KMeans(n_clusters=n_clusters)
    shape = inp_image.shape
    # Fit on Input Image
    k_means.fit(inp_image.flatten().reshape(shape[0]*shape[1], 1))
    # Get Cluster Labels
    #print(str(k_means.cluster_centers_))
    #return k_means.cluster_centers_.reshape((-1, 2))
    clust = k_means.labels_
    #print(str(clust))
    #return computeClusterMedoids(clust.reshape(shape[0], shape[1]),n_clusters)
    return computeClusterMaxoids(inp_image,clust.reshape(shape[0], shape[1]),n_clusters)

def fuzzy_c_means(inp_image, n_clusters=2):
    if inp_image is None:
        print("Empty Input. Exiting")
        return

    shape = inp_image.shape
    # Create and Train on FCM Model
    centers, u, u0, d, jm, n_iters, fpc = skf.cluster.cmeans(
        inp_image.flatten().reshape(shape[0]*shape[1], 1).T,
        c=n_clusters,
        m=float(3),
        error=float(0.05),
        maxiter=int(100),
        init=None,
        seed=int(21)
    )
    # Get Cluster Labels with Max Probability
    clust = np.argmax(u, axis=0)

    return computeClusterMaxoids(inp_image,clust.reshape(shape[0], shape[1]),n_clusters)

def gaussian_mixture_model(inp_image, n_clusters=2):
    shape = inp_image.shape
    inp_image = inp_image.flatten().reshape(shape[0]*shape[1], 1)
    MAX_ITER=25
    RANDOM_STATE=21
    COVARIANCE_TYPE="full"
    #WINDOW_SIZE=3
    # Create Gaussian Mixture Model with Config Parameters
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type=COVARIANCE_TYPE,
        max_iter=MAX_ITER, random_state=RANDOM_STATE)
    # Fit on Input Image
    gmm.fit(X=inp_image)
    # Get Cluster Labels
    clust = gmm.predict(X=inp_image)

    return computeClusterMaxoids(inp_image.reshape(shape[0], shape[1]),clust.reshape(shape[0], shape[1]),n_clusters)

def spectral_cluster(inp_image, n_clusters=2):
    raise ("SPECTRAL CLUSTERING NOT WORKING!!!!")
    original_shape = inp_image.shape
    downsampled_img = pyramid_reduce(inp_image, 3)
    shape = downsampled_img.shape
    downsampled_img = downsampled_img.reshape(shape[0]*shape[1], 1)
    EIGEN_SOLVER="arpack"
    AFFINITY="nearest_neighbors"
    #N_CLUSTERS=5
    #WINDOW_SIZE=2

    sp = SpectralClustering(n_clusters=n_clusters,
                            eigen_solver=EIGEN_SOLVER,
                            affinity=AFFINITY)
    sp.fit_predict(downsampled_img)
    clust = sp.labels_
    clust = clust.reshape(shape[0], shape[1])
    # Performimg kmeans to re generate clusters after resize, original segmentation remains intact.
    clust = k_means_clustering(resize(clust, (original_shape[-1])).reshape(shape[0],shape[1]),n_clusters)
    return computeClusterMaxoids(inp_image,clust,n_clusters)

def computeClusterMaxoids(inpImage,clust,nClusters):#clust is shaped as the image with labels
    # This function can be modified pretty simply so it also
    # receives a minimum label where to start the label numeration for the current window
    # Reenumerates the labels so they start at this minimum
    # Is there a "background" label?
    # This would be done to take advantage of the segmentations produced by clustering methods

    coordList=[]
    maxoids=[]
    for i in range(nClusters): maxoids.append([0,0,0])
    for x in range(clust.shape[0]):
        for y in range(clust.shape[1]):
            #print(str(x)+" "+str(y)+" "+str(clust[x][y]))
            if (clust[x][y]>0): # some algorithms output -1 values that are not really regions
                #print(inpImage[x,y])
                maxX=maxoids[clust[x][y]][0]
                maxY=maxoids[clust[x][y]][1]
                #print(inpImage[maxX,maxY])
                if inpImage[x,y]> inpImage[maxX,maxY]:
                    maxoids[clust[x][y]][0]=x
                    maxoids[clust[x][y]][1]=y
                maxoids[clust[x][y]][2]+=1
            #print("                               "+str(medoids))

    for c in range(nClusters):
        if maxoids[c][2]!=0 :
            coordList.append((maxoids[c][0],maxoids[c][1]))
            #print("IN here! "+str(coordList[-1]))

    return coordList

def refineSeedsWithMaximums(inpImage,maskImage):
    mask = cv2.threshold(255-maskImage, 80, 255, cv2.THRESH_BINARY)[1]
    #compute connected components
    numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

    coordList=[]
    maxoids=[]

    for i in range(numLabels): maxoids.append([0,0,0,-1])
    for x in range(labelImage.shape[0]):
        #if x>1000:break
        for y in range(labelImage.shape[1]):
            #print(str(x)+" "+str(y)+" "+str(labelImage[x][y]))
            if (labelImage[x][y]>0): # some algorithms output -1 values that are not really regions 0 is the background
                #print(inpImage[x,y])
                maxX=maxoids[labelImage[x][y]][0]
                maxY=maxoids[labelImage[x][y]][1]
                maxValue=maxoids[labelImage[x][y]][3]
            #print(inpImage[maxX,maxY])
                if inpImage[x,y]> maxValue:
                    #print(inpImage[x,y])
                    maxoids[labelImage[x][y]][0]=x
                    maxoids[labelImage[x][y]][1]=y
                    maxoids[labelImage[x][y]][3]=inpImage[x,y]
                maxoids[labelImage[x][y]][2]+=1

    for c in range(numLabels):
        #print("starting maxoid "+str(c))
        if maxoids[c][2]!=0 :
            # BECAUSE OPENCV RETURNS THINGS IN Y,X order!!!!!!!!!!!!!!!
            coordList.append((maxoids[c][1],maxoids[c][0]))
            #coordList.append((maxoids[c][0],maxoids[c][1]))


    return coordList

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-m", "--method", required=True, help="Clustering method, 0 Global Maximum, 1 Local Maxima")
    ap.add_argument("-d", "--dem", required=True, help="Path to Digital Elevation Map (DEM)")
    ap.add_argument("-o", "--binOut", required=False, help="Path of the resulting binary image")
    ap.add_argument("-w", "--window", required=True, help="Should we use a window or the whole image? y/n ")
    ap.add_argument("-s", "--size", required=True, help="Window size (squared)")
    ap.add_argument("-f", "--floor", required=True, help="The minimum altitude Value for which treetops are considered")
    ap.add_argument("-r", "--resize", required=False, help="Percentage of resizing")
    ap.add_argument("-th", "--LMThresh", required=False, help="Threshold for the local maximum")
    ap.add_argument("-md", "--LMMinDist", required=False, help="Minimum Distance for the local maximum")
    ap.add_argument("-eps", "--DBSCANepsilon", required=False, help="Epsilon for the DBSAN algorithm")
    ap.add_argument("-ms", "--DBSCANminS", required=False, help="Minimum Samples for the DBSAN algorithm")
    ap.add_argument("-nc", "--numClust", required=False, help="Number Of Clusters for clustering Algs")
    ap.add_argument("-ref", "--refine", required=False, help="yes/no, do we refine the results by fusing nearby points?")
    ap.add_argument("-ff", "--floorFile", required=False, help="If present, we will refine our results further by not considering points closes to nearest floor point than a threshold")
    ap.add_argument("-pf", "--pointFusion", required=False, help="We always paint a circle around every detected point in order to fuse nearby points, if present, this parameter contains the radius of said circle, default is 40")
    ap.add_argument("-lf", "--labelFile", required=False, help="If present, we will adapt our results by considering different point densities at different points")
    args = vars(ap.parse_args())

    # capture the method that we are Using
    # 0 Global Maxima (with threshold) 1 Local Maxima 2 kmeans
    method=int(args["method"])
    windowMethod = args["window"]=="y"

    # load the image and define the window width and height
    dem = cv2.imread(args["dem"],0)
    img = cv2.imread(args["image"])

    if args["resize"] is not None:
        dem = resize(dem, int(args["resize"]))

    if args["numClust"] is not None:
        numClustersOrig=int(args["numClust"])

    #gray = cv2.cvtColor(dem, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(dem,(5,5),0)
    seeds = []

    if windowMethod:(winW, winH) = (int(args["size"]), int(args["size"]))
    else:(winW, winH)=(-1,-1)

    # this is the lower value considered (to avoid picking the floor)
    floorThreshold=int(args["floor"])

    limitIterations=False
    count=0
    countLimit=10

    if args["labelFile"] is not None:
        labelIm=cv2.imread(args["labelFile"],0)
        if method>2: nClustersList=[0,max(int(numClustersOrig/2),1),numClustersOrig,numClustersOrig+2,int(numClustersOrig*(1.5))]
        circleSizeList=[0,110,80,70,60]
        # DENSER MOSAICS!
        #if method>2: nClustersList=[0,int(numClustersOrig/2),numClustersOrig,2*numClustersOrig,3*numClustersOrig]
        #circleSizeList=[0,50,40,35,35]

    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(gray, stepSize=int(args["size"]), windowSize=(winW, winH),allImage=not windowMethod):
        if limitIterations: count+=1
        #print("sliding the window to "+str((y,x)))
        # if the window does not meet our desired window size, ignore it
        if windowMethod and (window.shape[0] != winH or window.shape[1] != winW):continue

        # adjust number of clusters depending on the type of window
        if args["labelFile"] is not None and method>2:
            labelWindow=labelIm[y:y + int(args["size"]), x:x + int(args["size"])]
            #print("sliding labelWindow to "+str(y)+" "+str(x))
            avLabel=fe.averageWindow(labelWindow)+0.5
            #print("the average of the window was "+str(avLabel))
            if(int(avLabel)>len(nClustersList)-1):numClusters=0
            else: numClusters=nClustersList[int(avLabel)]
            #print("as the average of the window was "+str(avLabel)+" I will now search with num clusters "+str(numClusters))
        else:
            if args["numClust"] is not None:
                numClusters=numClustersOrig

        # Here the processing inside the window actually happens. We do different things for different methods
        if method==0: # Global Maximum
            # the area of the image with the largest intensity value
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(window)
            cv2.circle(window, maxLoc, 5, (255, 0, 0), 2)

            cords = (x+maxLoc[0], y+maxLoc[1])
            seeds.append(cords)
        elif method==1: # Local Maxima

                blur = cv2.medianBlur(window,5)

                thValue=int(args["LMThresh"])
                ret,thresh= cv2.threshold(blur,thValue,255,cv2.THRESH_BINARY)

                blur[thresh==0]=0

                im = img_as_float(blur)

                # image_max is the dilation of im with a 20*20 structuring element
                # It is used within peak_local_max function
                image_max = ndi.maximum_filter(im, size=20, mode='constant')

                # Comparison between image_max and im to f      ind the coordinates of local maxima
                # https://scikit-image.org/docs/0.7.0/api/skimage.feature.peak.html
                minDist=int(args["LMMinDist"])

                coordinates = peak_local_max(im, min_distance=minDist,threshold_rel=0.5)
                #print(str(len(seeds)))
                for a,b in coordinates:
                    #print(str((x+a,y+b)))
                    # It looks like scikit learn, AS OPENCV, returns coordinates in y,x order!!!!!
                    seeds.append((x+b,y+a))
        elif method==2: # DBSCAN
            if args["DBSCANepsilon"] is not None: dbscanEps=float(args["DBSCANepsilon"])
            else: dbscanEps=5
            if args["DBSCANminS"] is not None: dbscanMinS=int(args["DBSCANminS"])
            else: dbscanMinS=10

            coordinates=dbscan(window,dbscanEps,dbscanMinS)
            for a,b in coordinates:
                #print(str((x+a,y+b)))
                seeds.append((int(x+b),int(y+a)))

        elif method==3: # K-means
            if numClusters>0:
                coordinates = k_means_clustering(window, n_clusters=numClusters)
                for a,b in coordinates:
                    #print(str((x+a,y+b)))
                    seeds.append((int(x+b),int(y+a)))
        elif method==4: # Fuzzy C-means
            if numClusters>0:
                coordinates = fuzzy_c_means(window, n_clusters=numClusters)
                for a,b in coordinates:
                    #print(str((x+a,y+b)))
                    seeds.append((int(x+b),int(y+a)))
        elif method==5: # Gaussian mixture model
            #numClusters=10
            if numClusters>0:
                coordinates = gaussian_mixture_model(window, n_clusters=numClusters)
                for a,b in coordinates:
                    #print(str((x+a,y+b)))
                    seeds.append((int(x+b),int(y+a)))
        elif method==6: # spectral clustering
            #numClusters=10
            coordinates = spectral_cluster(window, n_clusters=numClusters)
            for a,b in coordinates:
                #print(str((x+a,y+b)))
                seeds.append((int(x+b),int(y+a)))
        else:
            print("Incorrect Method code "+str(method))
            sys.exit(0)
        if count>= countLimit:break

    # Once the method is finished, paint the detected points
    # paint them as circles over the DEM and the image
    print(" Total seeds, found "+str(len(seeds)))
    maskImage=255*np.ones((dem.shape[0],dem.shape[1],1),dtype=np.uint8)

    if args["pointFusion"] is not None:circleSize=int(args["pointFusion"])
    else:    circleSize=40

    #eliminate floor using crude threshold
    nonFloorSeeds=[]
    for seed in seeds:
        #for the first three methods, refine using the label image
        if args["labelFile"] is not None and method<3:
            currentLabel=min(labelIm[seed[1]][seed[0]],4)
            circleSize=circleSizeList[currentLabel]
        #if gray[seed[1]][seed[0]]>floorThreshold:
        if circleSize > 0:
            nonFloorSeeds.append(seed)
            cv2.circle(maskImage, seed, circleSize, 0, -1)
    print("Seeds with no floor! "+str(len(nonFloorSeeds)))
    #cv2.imwrite("noFloor.jpg", maskImage)

    # Refine using a file to indicate floor points
    if args["floorFile"] is not None:
        print("floor file "+args["floorFile"])
        fromFloorTH=30
        floorList=CSE.listFromBinary(args["floorFile"])
        # create a KDtree with all points
        kdt = KDTree(floorList, leaf_size=30, metric='euclidean')
        dist,ind=kdt.query(nonFloorSeeds, k=1)

        #print("I have "+str(len(ind))+" indices ")

        i=0
        aux=[]
        while i<len(ind):
            #check if the nearest floor point has more than fromFloorTH difference to the seed
            floorPoint=(int(floorList[ind[i]].flatten()[0]),int(floorList[ind[i]].flatten()[1]))
            seed=nonFloorSeeds[i]
            #print("floorpoint and seed "+str(floorPoint)+" "+str(seed))
            #print("First gray "+str(gray[seed[1]][seed[0]]))
            #print("Second gray "+str(gray[floorPoint[1]][floorPoint[0]]))
            dif=int(gray[seed[1]][seed[0]])-int(gray[floorPoint[1]][floorPoint[0]])
            #if dif<0:
            #    print("First gray "+str(gray[seed[1]][seed[0]]))
            #    print("Second gray "+str(gray[floorPoint[1]][floorPoint[0]]))
            #    print("dif!!! "+str(dif))
            if dif>0 and dif>fromFloorTH:aux.append(seed)
            i+=1

        nonFloorSeeds=aux
        print("Seeds with no floor filtered with file! "+str(len(nonFloorSeeds)))

    # Refine so close seeds get fused into one with the maximum
    if args["refine"] is not None and args["refine"] == "yes": outputSeeds=refineSeedsWithMaximums(gray,maskImage)
    else:
        print("Skipping seed refinement!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        outputSeeds=nonFloorSeeds
    print("Seeds refined! "+str(len(outputSeeds)))

    maskImage=255*np.ones((dem.shape[0],dem.shape[1],1),dtype=np.uint8)
    for seed in outputSeeds:
        #print("IN HERE "+str(seed))
        if args["labelFile"] is not None and method<3:
            currentLabel=min(labelIm[seed[1]][seed[0]],4)
            circleSize=circleSizeList[currentLabel]
        #if gray[seed[1]][seed[0]]>floorThreshold:
        if circleSize > 0:
            if args["refine"] is not None and args["refine"] == "yes":circleSize=10

            cv2.circle(dem, seed, circleSize, (0, 0, 255), -1)
            cv2.circle(img, seed, circleSize, (0, 0, 255), -1)
            # also create binary result image
            cv2.circle(maskImage, seed, circleSize, 0, -1)
    #
    # Finally, store all annotated images
    filename, file_extension = os.path.splitext(args["dem"])
    out_dem = filename + "_marked" + file_extension
    filename, file_extension = os.path.splitext(args["image"])
    out_img = filename + "_marked" + file_extension
    filename, file_extension = os.path.splitext(args["dem"])
    if args["binOut"] is not None: out_binary = args["binOut"]
    else: out_binary = filename + "_binaryMethod"+str(method) + file_extension

    cv2.imwrite(out_dem, dem)
    cv2.imwrite(out_img, img)
    cv2.imwrite(out_binary, maskImage)


if __name__ == "__main__":
    main()
