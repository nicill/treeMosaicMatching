import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

#given a window, compute its average
def averageWindow(window): return np.average(window)


#divide a DEM image into different density parts according to the differences in gradients.
def quantizeImage(dem, tops, floorTh,topTh,closeSize,closeIt):

    image=dem

    #First Crude threshold
    heightLabelImage=image.copy()
    heightLabelImage[image>topTh]=2
    heightLabelImage[image<topTh]=1
    heightLabelImage[image<floorTh]=0

    # For visualization
    #image[image<floorTh]=0
    #image[image>topTh]=200
    #image[tops==0]=255
    #cv2.imwrite("crudeFloor.jpg",image)

    gray = cv2.GaussianBlur(image,(5,5),0)
    laplace=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    laplace2=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
    gradientIm=laplace + laplace2
    cutOff=50
    gradientIm[gradientIm<cutOff]=0
    gradientIm[gradientIm>cutOff]=255
    #gradientIm[tops==0]=255
    cv2.imwrite("gradientThresholded.jpg",gradientIm)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gradientIm,kernel,iterations = 1)
    #erosion[tops==0]=255
    cv2.imwrite("gradientThresholdedEroded.jpg",erosion)

    kernelClosing = np.ones((closeSize,closeSize),np.uint8)

    close= cv2.morphologyEx(erosion,cv2.MORPH_CLOSE,kernelClosing,iterations = closeIt)
    gradientLabelImage=close.copy()
    gradientLabelImage[close==255]=2
    gradientLabelImage[close<255]=0
    #0 height is zero!
    gradientLabelImage[heightLabelImage==0]=0

    labelImage=heightLabelImage+gradientLabelImage
    #labelImage=gradientLabelImage

    representLabels=labelImage*61
    #representLabels[tops==255]=255
    cv2.imwrite("viewLabels.jpg",representLabels)

    return labelImage


def main(argv):
    # Receive DEM image in argv[2] and top image in argv[3],
    # then the size of the window in argv[4], and the crude floor threshols in argv[5]
    image=cv2.imread(argv[1],cv2.IMREAD_GRAYSCALE)
    tops=cv2.imread(argv[2],cv2.IMREAD_GRAYSCALE)

    labelIm=quantizeImage(image,tops,int(argv[3]),int(argv[4]),int(argv[5]),int(argv[6]))
    cv2.imwrite("labels.jpg",labelIm)


if __name__ == '__main__':
    main(sys.argv)
