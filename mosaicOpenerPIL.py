import tifffile as tiff
import sys
import cv2
from PIL import Image


#im = cv2.imread(sys.argv[1])
#print(im.shape)

fixed = Image.open(sys.argv[1])
#moving = Image.open(sys.argv[2])

pixFixed = list(fixed.getdata())
for i in range(len(pixFixed)):
    if( pixFixed[i][3]!=255 ): print(pixFixed[i])

print(len(pixFixed))




#im2.show()


#cv2.imwrite("./sambomba.tif",im)

#a = tiff.imread(sys.argv[1])
#a.shape
