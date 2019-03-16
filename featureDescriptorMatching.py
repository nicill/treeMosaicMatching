import cv2
import numpy as np
import sys


MAX_FEATURES = 100000
GOOD_MATCH_PERCENT = 0.0015

#0 ORB 1 AKAZE
def alignImages(im1, im2,mode,matchesFile="NO"):

    print(matchesFile)

    if (mode==0):#Orb matching
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
        print("Error in alignImages, unrecognized descriptor type "+str(mode))
        sys.exit ( 1 )


    print("number of keypoints "+str(len(keypoints1))+" "+str(len(keypoints2)))

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    print("number of matches "+str(len(matches)))
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    print("number of goodmatches  "+str(numGoodMatches))

    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    if(matchesFile!="NO"): cv2.imwrite(matchesFile, imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    # h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width, channels = im2.shape
    #im1Reg = cv2.warpPerspective(im1, h, (width, height))

    #estimate rigid SetTransform
    affTransf,inliers=cv2.estimateAffine2D(points1, points2)
    im1Reg=cv2.warpAffine(im1, affTransf, (width, height))
    return im1Reg, affTransf
    #return im1Reg, h


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
  if(len(sys.argv)>5): imReg, h = alignImages(im, imReference,int(sys.argv[3]),sys.argv[5])
  else: alignImages(im, imReference,int(sys.argv[3]))

  # Write aligned image to disk.
  outFilename = sys.argv[4]
  print("Saving aligned image : ", outFilename);
  cv2.imwrite(outFilename, imReg)

  # Print estimated homography
  print("Estimated homography : \n",  h)
