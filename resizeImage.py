import cv2
import sys

filename = sys.argv[1]
W = 1000.
oriimg = cv2.imread(filename,cv2.IMREAD_COLOR)
#height, width, depth = oriimg.shape
#imgScale = W/width
#newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
print(oriimg.shape)
newX=int(float(sys.argv[2])*float(sys.argv[4]))
newY=int(float(sys.argv[3])*float(sys.argv[4]))
newimg = cv2.resize(oriimg,(int(newX),int(newY)))
#cv2.imshow("Show by CV2",newimg)
#cv2.waitKey(0)
cv2.imwrite(sys.argv[5],newimg)
