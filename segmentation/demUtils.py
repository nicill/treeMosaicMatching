
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def main(argv):

    #img = cv2.imread(argv[1],0)
    img =cv2.imread(argv[1], cv2.IMREAD_ANYDEPTH)
    print(argv[1])
    print(img.shape)
    #print(img[0[0]])

    plt.imshow(img)
    plt.show()
if __name__ == '__main__':
    main(sys.argv)
