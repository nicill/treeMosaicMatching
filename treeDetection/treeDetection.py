# This method takes the point coordinates of each tree top given by crownSegementerEvaluator and extracts 
# a squared patch arround each one. Then, uses the classified masks of the mosaics to know in which species belongs.
# Then stores each small labeled patch in a folder.

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def borderPoint(image,point):
    margin=100
    top1=image.shape[0]
    top2=image.shape[1]

    return point[0]<margin or (top1-point[0])<margin or point[1]<margin or (top2-point[1])<margin

# Function to take a binary image and output the center of masses of its connected regions
# THIS METHOD IS A COPY OF crownSectmenterEvaluator method! must be deleted!!!
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

        return newCentroids[1:]


def getSquare(w_size, p, img):


	height, width, channels = img.shape

	isInside = (int(p[0])-w_size//2) >= 0 and (int(p[0])+w_size//2) < width and (int(p[1])-w_size//2) >= 0 and (int(p[1])+w_size//2) < height
	
	assert isInside, "The required window is out of bounds of the input image"

	return img[int(p[0])-w_size//2:int(p[0])+w_size//2, int(p[1])-w_size//2:int(p[1])+w_size//2]

		
def main(argv):

	try:
		treetops_mask_file = argv[1]
		mosaic_file = argv[2]
		output_path = argv[3]

		treetops_mask = cv2.imread(treetops_mask_file, cv2.IMREAD_GRAYSCALE)
		mosaic = cv2.imread(mosaic_file, cv2.IMREAD_COLOR)

		centroids = listFromBinary(treetops_mask_file)
		i = 1

		for cent in centroids:

			try:
				# opencv works with inverted coords, so we have to invert ours.
				square = getSquare(100, (cent[1],cent[0]), mosaic)
			
				cv2.imwrite(output_path+"patch"+str(i)+".jpg", square) 
				i=i+1
			
			except AssertionError as error:
				print(error)
			

	except AssertionError as error:

		print(error)
		

# Exectuion example -> python treeDetection.py <path_to_top_mosaic> <path_to_mosaic> <output_folder>

if __name__ == "__main__":
    main(sys.argv)	
