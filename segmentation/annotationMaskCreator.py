import cv2
import numpy as np
import sys
import imagePatcherAnnotator as impa

def main():
    # Take a mosaic, a csv file containing predictions for its labels and the patch size used for the annotations
    # 1) Create trentative automatic mask images (all affected patches are black)

    #hardcoded number of layers and names
    layerNames=["river","decidious","uncovered","evergreen","manmade"]
    layerDict={} #dictionary so that we know what number corresponds to each layer
    for i in range(len(layerNames)):layerDict[layerNames[i]]=i

    # Read parameters
    patch_size = int(sys.argv[1])
    csvFile=sys.argv[2]
    imageDir=sys.argv[3]

    #read also all the prefixes of all the images that we have
    imagePrefixes=[]
    for x in range(4,len(sys.argv)):imagePrefixes.append(sys.argv[x])
    imageDict={}
    for i in range(len(imagePrefixes)):imageDict[imagePrefixes[i]]=i

    print("AnnotationMask creator main, parameters: csv files: "+str(csvFile)+" image directory"+str(imageDir)+" image prefixes "+str(imagePrefixes))

    f = open(csvFile, "r")
    firstImageName=imageDir+imagePrefixes[0]+".jpg"
    print(firstImageName)
    image = cv2.imread(firstImageName,cv2.IMREAD_COLOR)

    shapeX=image.shape[0]
    shapeY=image.shape[1]

    print(str(image.shape))

    #create a blank image for each layers
    layerList=[]
    i=0
    for pref in imagePrefixes:
        layerList.append([])
        for x in range(len(layerNames)):
            layerList[i].append(np.zeros((shapeX,shapeY),dtype=np.uint8))
            #layerList[i][x][:]=255
            #cv2.imwrite(imageDir+pref+"layer"+str(x)+".jpg",layerList[x])
        i+=1

    #first, make the patches fully inside the image
    # consider the dimensions of the image to set up the number of patches
    numStepsX=int(shapeX/patch_size)
    numStepsY=int(shapeY/patch_size)

    # go over the csv file, for every line
        # extract the image prefixes
        # extract the lables
        # for every label found, paint a black patch in the correspoding image layer
    print("steps "+str(numStepsX)+" "+str(numStepsY))
    for line in f:
        #process every line
        #print(line)
        pref=line.split("p")[0]
        patchNumber=int(line.split("h")[1].split(" ")[0])
        labelList=line.split(" ")[1].strip().split(";")
        #print("           "+pref+" patchNum "+str(patchNumber))
        #print(str(labelList))
        for x in labelList:
            if x=="":break
            #now, paint the information of each patch in the layer where it belongs
            xJump=patchNumber//numStepsY
            yJump=patchNumber%numStepsY

            #print("patch "+str(patchNumber)+" will go to ("+str(xJump)+","+str(yJump)+")")
            #now find the proper layer (once in the im)
            #print("searching for layer with "+pref+" "+x)
            #print("and will modify the layer "+str(imageDict[pref])+" "+str(layerDict[x]))
            currentLayerIm=layerList[imageDict[pref]][layerDict[x]]
            impa.paintImagePatch(currentLayerIm,xJump*patch_size,yJump*patch_size,patch_size,255)


    #finally, write the resulting images
    i=0
    #kernel = np.ones((3,3),dtype=np.uint8)
    kernel = np.zeros((3,3),np.uint8)
    kernel[:]=255
    #initialize mask accumulator

    for pref in imagePrefixes:
        layerList.append([])
        maskImage=np.zeros((shapeX,shapeY),dtype=np.uint8)
        for x in range(len(layerNames)):
            if layerNames[x] in ["decidious","evergreen"]:
                #merge these mask with the ones before

                # also, try to refine the segmenation
                # noise removal
                #ret, thresh = cv2.threshold(layerList[i][x],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                opening = cv2.morphologyEx(layerList[i][x],cv2.MORPH_OPEN,kernel, iterations = 2)

                # sure background area
                sure_bg = cv2.dilate(opening,kernel,iterations=3)
                #cv2.imwrite(imageDir+pref+"ErodedLayer"+str(x)+".jpg",opening)

                # Finding sure foreground area
                #dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                #ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
                sure_fg = cv2.erode(layerList[i][x],kernel,iterations=300)
                #cv2.imwrite(imageDir+pref+"dilatedLayer"+str(x)+".jpg",sure_fg)
                cv2.imwrite(str(x)+"sf.jpg",sure_fg)
                cv2.imwrite(str(x)+"sb.jpg",sure_bg)


                # Finding unknown region
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg,sure_fg)

                maskImage=maskImage|sure_fg
                #cv2.imwrite(imageDir+pref+"GeneratedLayer"+str(x)+".jpg",layerList[i][x])
                cv2.imwrite(str(x)+"auauua.jpg",maskImage)
            else:
                print("skypping layer "+layerNames[x])




        # Marker labelling
        ret, markers = cv2.connectedComponents(maskImage)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        print("starting watershed ")
        markers = cv2.watershed(image,markers)
        image[markers == -1] = [0,0,255]

        cv2.imwrite("sambomba3.jpg",image)

        i+=1

        sys.exit()





if __name__== "__main__":
  main()
