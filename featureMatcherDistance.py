ss


#from __future__ import print_function
import cv2
import numpy as np
import sys

MAX_FEATURES = 100000
GOOD_MATCH_PERCENT = 0.0015

def dist(pt1,pt2):
    #print(pt1)

    a=np.array((pt1[0] , pt1[1]))
    b=np.array((pt2[0] , pt2[1]))
    return np.linalg.norm(b-a)

def computeSimilarity(im1, im2,mode):

    # We should have stored the points with good matches and deform them with the transformation, then compute the distance

    if (mode==0):#Orb matching
        MAX_FEATURES = 100000
        GOOD_MATCH_PERCENT = 0.05

        # Convert images to grayscale
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    elif (mode==1):#AKAZE matching

        akaze = cv2.AKAZE_create()
        keypoints1, descriptors1 = akaze.detectAndCompute(im1, None)
        keypoints2, descriptors2 = akaze.detectAndCompute(im2, None)

    elif (mode==2):#Brisk matching

        brisk = cv2.BRISK_create()
        keypoints1, descriptors1 = brisk.detectAndCompute(im1, None)
        keypoints2, descriptors2 = brisk.detectAndCompute(im2, None)

    else:
        print("Error in getMatches, unrecognized descriptor type "+str(mode))
        sys.exit ( 1 )

    # create BFMatcher object
    #bf = cv2.BFMatcher()
    bf = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    # Match descriptors.
    matches = bf.knnMatch(descriptors1,descriptors2,2)

    # Apply ratio test
    avDistance=0.
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            avDistance=avDistance+dist(keypoints1[m.queryIdx].pt,keypoints2[m.trainIdx].pt)
            if(len(good)==100):
                #print("breaking")
                break

        #else: print(str(m.distance)+" and "+str(n.distance) )
    #print("good and total "+str(len(good))+" "+str(len(matches)) )
    #return 100*(len(good)/len(matches))
    return avDistance/len(good)

if __name__ == '__main__':

    # Read reference image
    refFilename = sys.argv[1]
    #print("Reading image1 : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = sys.argv[2]
    #print("Reading image2 : ", imFilename);
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    #do something!
    print(computeSimilarity(imReference,im,int(sys.argv[3])),end=" ")
