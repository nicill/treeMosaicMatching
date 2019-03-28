import cv2
import numpy as np
import sys

#these two parameters need to be adjusted for the descriptors
MAX_FEATURES = 100000
GOOD_MATCH_PERCENT = 0.05

#0 ORB 1 AKAZE 2 BRISK
def alignImages(im1, im2,mode,matchesFile="NO"):

    #print(matchesFile)

    if (mode==0):#Orb matching
        MAX_FEATURES = 100000
        #GOOD_MATCH_PERCENT = 0.05

        # Convert images to grayscale
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES,1.1)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    elif (mode==1):#AKAZE matching
        MAX_FEATURES = 100000
        #GOOD_MATCH_PERCENT = 0.0015

        akaze = cv2.AKAZE_create()
        keypoints1, descriptors1 = akaze.detectAndCompute(im1, None)
        keypoints2, descriptors2 = akaze.detectAndCompute(im2, None)

    elif (mode==2):#Brisk matching
        MAX_FEATURES = 100000
        #GOOD_MATCH_PERCENT = 0.15

        brisk = cv2.BRISK_create()
        keypoints1, descriptors1 = brisk.detectAndCompute(im1, None)
        keypoints2, descriptors2 = brisk.detectAndCompute(im2, None)

    else:
        print("Error in alignImages, unrecognized descriptor type "+str(mode))
        sys.exit ( 1 )


    print("number of keypoints "+str(len(keypoints1))+" "+str(len(keypoints2)))

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    # choose according to lowe7s Threshold
    matches = matcher.knnMatch(descriptors1,descriptors2,2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])


    # Match features.no threshold
    #matches = matcher.match(descriptors1, descriptors2, None)
    #print("number of matches "+str(len(matches)))
    # Sort matches by score
    #matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    #numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    #matches = matches[:numGoodMatches]
    #print("number of goodmatches  "+str(numGoodMatches))

    imMatches = cv2.drawMatchesKnn(im1, keypoints1, im2, keypoints2, good, None)
    if(matchesFile!="NO"): cv2.imwrite(matchesFile, imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)

    for i, match in enumerate(good):
        points1[i, :] = keypoints1[match[0].queryIdx].pt
        points2[i, :] = keypoints2[match[0].trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    return im1Reg, h

    #estimate rigid SetTransform
    #affTransf,inliers=cv2.estimateAffine2D(points1, points2)
    #im1Reg=cv2.warpAffine(im1, affTransf, (width, height))
    #return im1Reg, affTransf


if __name__ == '__main__':

  # Read reference image
  refFilename = sys.argv[1]
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

  # Read image to be aligned
  imFilename = sys.argv[2]
  print("Reading image to align : ", imFilename);
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

  print("Aligning images ...")
  # Registered image will be resotred in imReg.
  # The estimated homography will be stored in h.
  #imReg, h = alignImagesORB(im, imReference)
  if(len(sys.argv)>6): imReg, h = alignImages(im, imReference,int(sys.argv[3]),sys.argv[6])
  else: imReg, h = alignImages(im, imReference,int(sys.argv[3]))

  # Write aligned image to disk.
  outFilename = sys.argv[5]
  print("Saving aligned image : ", outFilename);
  cv2.imwrite(outFilename, imReg)

  # Print estimated homography
  print("Estimated homography : \n",  h)
  transformFileName=sys.argv[4]
  f = open(transformFileName, "w")
  f.write(str(h))
  f.close()
