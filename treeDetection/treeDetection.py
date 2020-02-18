# This method takes the point coordinates of each tree top given by crownSegementerEvaluator and extracts 
# a squared patch arround each one. Then, uses the classified masks of the mosaics to know in which species belongs.
# Then stores each small labeled patch in a folder.

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from treeMosaicMatching.segmentation.crownSegmenterEvaluator import listFromBinary



def getSquare(w_size, p, img):


	height, width, channels = img.shape

	isInside = (p[0]-w_size//2) >= 0 and (p[0]+w_size//2) < width and (p[1]-w_size//2) >= 0 or (p[0]+w_size//2) < height
	
	assert isInside, "The required window is out of bounds of the input image"


	return img[p[0]-w_size//2:p[0]+w_size//2, p[1]-w_size//2:p[1]+w_size//2]

		
	
def main(argv):

	try:
		treetops_mask_file = argv[1]
		mosaic_file = argv[2]

		treetops_mask = cv2.imread(treetops_mask_file, cv2.IMREAD_GRAYSCALE)
		mosaic = cv2.imread(mosaic_file, cv2.IMREAD_COLOR)

		# imgplot = plt.imshow(treetops_mask)
		# plt.show()

		# cv2.imshow("MOSAIC", mosaic)
		# cv2.waitKey(0)


		centroids = listFromBinary(treetops_mask_file)

		for cent in centroids:

			square = getSquare(200, cent, mosaic)
			
			cv2.imwrite(filename, img) 

		# for cent in centroids

	except AssertionError as error:

		print(error)
		







if __name__ == "__main__":
    main(sys.argv)