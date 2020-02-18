import cv2
import numpy as np
from collections import namedtuple
import os
import sys

mosaicInfo = namedtuple("mosaicInfo","path mosaicFile numClasses layerNameList layerFileList outputFolder " )

def imagePatch(image,minX,minY,size,verbose=False):
    if(verbose):print("making patch [ "+str(minX)+" , "+str(minX+size)+"] x [ "+str(minY)+" , "+str(minY+size)+"]")

    return image[minX:minX+size, minY:minY+size]

def paintImagePatch(image,minX,minY,size,color,verbose=False):
    if(verbose):print("painting patch [ "+str(minX)+" , "+str(minX+size)+"] x [ "+str(minY)+" , "+str(minY+size)+"]")

    image[minX:minX+size, minY:minY+size]=color


def imageNotEmpty(image):
    whitePixelThreshold=1000
    whitePixels=np.sum(image==255)
    blackPixels=np.sum(image==0)
    totalPixels=(image.shape[0]*image.shape[1])
    nonwhitePixels=totalPixels-whitePixels
#    return nonwhitePixels>whitePixelThreshold
    return blackPixels>whitePixelThreshold


def notWhite(pixel):
    return pixel!=255
    #return (pixel[0]!=255 or pixel[1]!=255 or pixel[2]!=255)

def listPixels(image):
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[0]):
            print ("pixel "+str(image[i][j]))
            #if(notWhite(image[i][j])):return True
    return False

def interpretParameters(paramFile,verbose=False):
    # read the parameter file line by line
    f = open(paramFile, "r")
    patchSize=-1
    layerNameList=[]
    layerList=[]
    mosaicDict={}

    for x in f:
        lineList=x.split(" ")
        # read every line
        first=lineList[0]

        if first[0]=="#": #if the first character is # treat as a comment
            if verbose:print("COMMENT: "+str(lineList))
        elif first=="\n":# account for blank lines, do nothing
            pass
        elif first=="patchSize":
            patchSize=int(lineList[1].strip())
            if verbose:print("Read Patch Size : "+str(patchSize))
        elif first=="csvFileName":
            csvFileN=lineList[1].strip()
            if verbose:print("Read csv file name : "+csvFileN)
        elif first=="mosaic":
            # read the number of layers and set up reading loop
            filePath=lineList[1]
            mosaic=lineList[2]
            numClasses=int(lineList[3])
            outputFolder=lineList[4+numClasses*2].strip()
            for i in range(4,numClasses*2+3,2):
                layerNameList.append(lineList[i])
                layerList.append(filePath+lineList[i+1])

            #make dictionary entry for this mosaic
            mosaicDict[mosaic]=mosaicInfo(filePath,mosaic,numClasses,layerNameList,layerList,outputFolder)
            if verbose:
                print("\n\n\n")
                print(mosaicDict[mosaic])
                print("\n\n\n")
                #print("Read layers and file : ")
                #print("filePath "+filePath)
                #print("mosaic "+mosaic)
                #print("num Classes "+str(numClasses))
                #print("layerName List "+str(layerNameList))
                #print("layer List "+str(layerList))
                #print("outputFolder "+outputFolder)
        else:
            raise Exception("ImagePatchAnnotator:interpretParameters, reading parameters, received wrong parameter "+str(lineList))

        if verbose:(print(mosaicDict))

    return patchSize,csvFileN,mosaicDict

def readImage(imageName,mode,verbose=False):
    if verbose: print("readImage::Reading "+imageName)
    if mode=="color":image=cv2.imread(imageName,cv2.IMREAD_COLOR)
    elif mode=="grayscale":image=cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
    else: raise Exception("imagePatcherAnnotator:readImage, wrong image mode")

    if image is None: raise Exception("ImagePatchAnnotator:readImage image not found "+imageName)
    return image


def mosaicToPatches(mInfo, patchsize, csvFileName,firstMosaic=False,verbose=False):
    #mosaicInfo = namedtuple("mosaicInfo","path mosaicFile numClasses layerNameList layerFileList outputFolder " )

    layerNames=mInfo.layerNameList
    imageName=mInfo.path+"/"+mInfo.mosaicFile

    f = open(csvFileName, "a")
    image = readImage(imageName,"color",verbose)

    shapeX=image.shape[0]
    shapeY=image.shape[1]

    outputDir=mInfo.path+mInfo.outputFolder+"/"
    if firstMosaic:
        try:
            # Create target Directory
            os.mkdir(outputDir)
            print("Directory " , outputDir ,  " Created ")
        except FileExistsError:
            print("Directory " , outputDir ,  " already exists, this should probably not be happening, erase it first unless you have a good reason not to!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    outputPrefix=mInfo.mosaicFile[:-4]

    if verbose:print("OUTPUTDIR "+outputDir)
    if verbose:print("OUTPUTPrefix "+outputPrefix)

    #print("image shape "+str(shapeX)+","+str(shapeY))

    #first, make the patches fully inside the image
    numStepsX=int(shapeX/patchsize)
    numStepsY=int(shapeY/patchsize)

    if verbose: print("steps "+str(numStepsX)+" "+str(numStepsY))
    count=0
    if firstMosaic: f.write("image,tags")

    #read all layer images
    layerList=[]
    for x in range(0,len(layerNames)):
        # check if the layer in this patch is empty
        layerName=mInfo.layerFileList[x]
        if verbose: print("Reading "+layerName)
        #layerList.append(cv2.imread(layerName,cv2.IMREAD_GRAYSCALE))
        layerList.append(readImage(layerName,"grayscale",verbose))

    for i in range(0,numStepsX):
        if verbose:print("i is now "+str(i)+" of "+str(numStepsX))
        for j in range(0,numStepsY):
            if verbose:print("             j is now "+str(j)+" of "+str(numStepsY))
            # create patch image and write it

            # also check how many layers are non-empty in this patch
            layerString=""
            for x in range(0,len(layerList)):
                # check if the layer in this patch is empty
                if layerList[x] is None :break

                layer=layerList[x]
                layerPatch=imagePatch(layer ,i*patchsize,j*patchsize,patchsize,False  )

                #cv2.imwrite("./wm1/patches/patch"+str(count)+"Layer"+str(x)+".jpg",layerPatch)
                if( imageNotEmpty( layerPatch )):
                    layerString+=layerNames[x]+" "
                    #print("updating layer string for "+layerString)

            if(not layerString=="") :
                outputImageName=outputDir+"/"+outputPrefix+"patch"+str(count)+".jpg"
                #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GOING TO WRITE IMAGE TO FILE "+outputImageName)
                cv2.imwrite(outputImageName, imagePatch(image,i*patchsize,j*patchsize,patchsize) )
                if verbose: print("for patch "+str(count)+" string is "+layerString)
                f.write("\n"+outputPrefix+"patch"+str(count)+","+layerString.strip())
                f.flush()
            count+=1

    f.close()



def main(argv):

    verbose=False
    patchSize,csvFileName,mosaicDict=interpretParameters(argv[1])

    #if verbose: print(mosaicDict)
    firstMosaic=True
    for k,v in mosaicDict.items():
        if verbose: print("\n\nstarting processing of first mosaic and layers "+str(v)+"\n\n")
        mosaicToPatches(v, patchSize, csvFileName,firstMosaic,verbose)
        firstMosaic=False





if __name__== "__main__":
  main(sys.argv)
