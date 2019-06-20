import cv2
import numpy as np
import sys


def imagePatch(image,minX,minY,size,verbose=False):
    if(verbose):print("making patch [ "+str(minX)+" , "+str(minX+size)+"] x [ "+str(minY)+" , "+str(minY+size)+"]")

    return image[minX:minX+size, minY:minY+size]

def paintImagePatch(image,minX,minY,size,color,verbose=False):
    if(verbose):print("painting patch [ "+str(minX)+" , "+str(minX+size)+"] x [ "+str(minY)+" , "+str(minY+size)+"]")

    image[minX:minX+size, minY:minY+size]=color


def imageNotEmpty(image):
    whitePixelThreshold=50
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


def main(argv,csvFileName=""):

    #hardcoded number of layers and names
    layerNames=["river","decidious","uncovered","evergreen","manmade"]
    #layerNames=["river","decidious","uncovered","evergreen"]

    # load the image
    imagePrefix=argv[1]
    imageName=imagePrefix+".jpg"
    if (csvFileName==""):csvFile=imagePrefix+".csv"
    else:csvFile=csvFileName

    f = open(csvFile, "a")
    print(imageName)
    image = cv2.imread(imageName,cv2.IMREAD_COLOR)
    patch_size = int(argv[2])
    shapeX=image.shape[0]
    shapeY=image.shape[1]
    print(str(image.shape))

    outputDir=argv[3]
    outputPrefix=argv[4]

    print("OUTPUTDIR "+outputDir)

    #print("image shape "+str(shapeX)+","+str(shapeY))

    #first, make the patches fully inside the image
    numStepsX=int(shapeX/patch_size)
    numStepsY=int(shapeY/patch_size)

    print("steps "+str(numStepsX)+" "+str(numStepsY))
    count=0
    if(argv[5]=="True"): f.write("image,tags")

    layerList=[]
    for x in range(0,len(layerNames)):
        # check if the layer in this patch is empty
        layerName=imagePrefix+"layer"+str(x)+".jpg"
        print("Reading "+layerName)
        layerList.append(cv2.imread(layerName,cv2.IMREAD_GRAYSCALE))

    for i in range(0,numStepsX):
        print("i is now "+str(i)+" of "+str(numStepsX))
        for j in range(0,numStepsY):
            print("             j is now "+str(j)+" of "+str(numStepsY))
            # create patch image and write it

            # also check how many layers are non-empty in this patch
            layerString=""
            for x in range(0,len(layerList)):
                # check if the layer in this patch is empty
                if layerList[x] is None :break

                layer=layerList[x]
                layerPatch=imagePatch(layer ,i*patch_size,j*patch_size,patch_size,False  )

                #cv2.imwrite("./wm1/patches/patch"+str(count)+"Layer"+str(x)+".jpg",layerPatch)
                if( imageNotEmpty( layerPatch )):
                    layerString+=layerNames[x]+" "
                    #print("updating layer string for "+layerString)

            if(not layerString=="") :
                outputImageName=outputDir+"/"+outputPrefix+"patch"+str(count)+".jpg"
                #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GOING TO WRITE IMAGE TO FILE "+outputImageName)
                cv2.imwrite(outputImageName, imagePatch(image,i*patch_size,j*patch_size,patch_size) )
                print("for patch "+str(count)+" string is "+layerString)
                f.write("\n"+outputPrefix+"patch"+str(count)+","+layerString.strip())
                f.flush()
            count+=1

    f.close()


if __name__== "__main__":
  main(sys.argv)
