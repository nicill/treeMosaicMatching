import cv2
import numpy as np
import sys



def main(argv):

    try:
        img=cv2.imread(argv[1],-1)
    except Exception as e:
        print(str(e))

    print(str(img.shape))


if __name__ == '__main__':
    main(sys.argv)
