import cv2
import numpy as np
import sys

def makeSnowMask(img):
    ret,thresh1 = cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
    return thresh1

def getSnowOutOfMask(snowMask,manualMask):
    aux1=snowMask.copy()
    aux2=snowMask.copy()
    aux1[manualMask==0]=1 #mark mask regions as interesting
    aux1[manualMask>0]=0 #now aux1 contains 1 in the mask regions and 0 elsewhere
    aux2[snowMask>0]=0
    aux2[snowMask==0]=1 #aux2==1 means there is snow there, zero elsewhere
    aux2=aux1+aux2
    manualMask[aux2==2]=255
    return manualMask

def getSnowOutOfGeneratedMask(snowMask,genMask):
    aux1=snowMask.copy()
    aux2=snowMask.copy()
    aux1[genMask>0]=1 #mark mask regions as interesting
    aux1[genMask==0]=0 #now aux1 contains 1 in the mask regions and 0 elsewhere
    aux2[snowMask>0]=0
    aux2[snowMask==0]=1 #aux2==1 means there is snow there, zero elsewhere
    aux2=aux1+aux2
    #print(str(genMask.shape))
    genMask[aux2==2]=0
    return genMask

def main(argv):

    #option 0, make snow mask
    if (int(argv[1])==0):
        img = cv2.imread(argv[2],0)
        thresh1=makeSnowMask(img)
        cv2.imwrite(argv[3],thresh1)
    #option 2 bitwise and two binary images
    if (int(argv[1])==1):
        print("parameters "+argv[1]+" "+argv[2]+" "+argv[3])
        snowMask = cv2.imread(argv[2],0)
        manualMask = cv2.imread(argv[3],0)
        out = getSnowOutOfMask(snowMask,manualMask)
        cv2.imwrite(argv[4],out)


if __name__ == '__main__':
    main(sys.argv)
