import cv2
import numpy as np
import sys
import imagePatcherAnnotator as impa

#def addNewMaskLayer(newMask,mainMask):
#    shapeX=mainMask.shape[0]
#    shapeY=mainMask.shape[1]
#    for i in range(shapeX):
#        if i%100==0:print(str(i)+"/"+str(shapeX))
#        for j in range(shapeY):
            # Unknown + background turns to unknown, label with unknown o background is propagated
#            if newMask[i][j]==0 and mainMask[i][j]==1:mainMask[i][j]=0
#            elif newMask[i][j]>1 and (mainMask[i][j]==0 or mainMask[i][j]==1):mainMask[i][j]=newMask[i][j]

def addNewMaskLayer(newMask,mainMask):
    aux1=newMask.copy()
    aux2=mainMask.copy()

    aux1[newMask==1]=2# now we only have unknown as zero and other things at least 2
    aux1[mainMask==1]+=1 #now 1 contains the pixels that where 0 in new and 1 in main
    mainMask[aux1==1]=0 # unknown + background is unknown

    aux1=newMask.copy()
    aux1[aux2>1]=0 #erase everything not touching mainmask unknown or background
    aux1[aux1==1]=0 #erase also background
    mainMask=mainMask|aux1 # add the labels

    return mainMask

def main():
    # Take a mosaic, a csv file containing predictions for its labels and the patch size used for the annotations
    # 1) Create trentative automatic mask images (all affected patches are black)
    # 2) Find a clear background and clear foreground part, find unknown part, find the connected components of the foregroung
    # 3) Accumulate labels for all cathegories, carefull to keep the unkwnon updated as it is where the segmentation can grow
    # 4) run watershed

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

            #now find the proper layer (once in the im)
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
        firstLabel=0 #counter so labels from different masks have different labels
        for x in range(len(layerNames)):
            if layerNames[x] in ["river","decidious","uncovered","evergreen"]:
                print("starting "+layerNames[x])
                #merge these mask with the ones before
                cv2.imwrite(str(x)+"before.jpg",layerList[i][x])

                # also, try to refine the segmenation
                # noise removal
                #ret, thresh = cv2.threshold(layerList[i][x],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                opening = cv2.morphologyEx(layerList[i][x],cv2.MORPH_OPEN,kernel, iterations = 2)

                # sure background area
                sure_bg = cv2.dilate(opening,kernel,iterations=100)
                #cv2.imwrite(imageDir+pref+"ErodedLayer"+str(x)+".jpg",opening)

                # Finding sure foreground area
                dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
                #iterations=350
                #if layerNames[x]=="evergreen": iterations =200
                #sure_fg = cv2.erode(layerList[i][x],kernel,iterations=iterations)
                #cv2.imwrite(imageDir+pref+"dilatedLayer"+str(x)+".jpg",sure_fg)

                # Finding unknown region
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg,sure_fg)

                cv2.imwrite(str(x)+"sf.jpg",sure_fg)
                cv2.imwrite(str(x)+"sb.jpg",sure_bg)
                cv2.imwrite(str(x)+"unk.jpg",unknown)

                # Marker labelling
                ret, markers = cv2.connectedComponents(sure_fg)

                # Add one to all labels so that sure background is not 0, but 1, also add firstLabel so label numbers are different
                markers = markers+firstLabel+1

                #remark sure background as 1
                markers[markers==(firstLabel+1)]=1

                firstLabel+=ret
                # Now, mark the region of unknown with zero
                markers[unknown==255] = 0

                maskImage=addNewMaskLayer(markers,maskImage)

                #cv2.imwrite(imageDir+pref+"GeneratedLayer"+str(x)+".jpg",layerList[i][x])
                cv2.imwrite(str(x)+"auauua.jpg",cv2.applyColorMap(np.uint8(markers*50),cv2.COLORMAP_JET))
            else:
                print("skypping layer "+layerNames[x])

        cv2.imwrite("finalMask.jpg",cv2.applyColorMap(np.uint8(maskImage*50),cv2.COLORMAP_JET))

        print("starting watershed ")
        markers = cv2.watershed(image,maskImage)
        image[markers == -1] = [0,0,255]

        cv2.imwrite("watershed.jpg",image)

        i+=1

        sys.exit()





if __name__== "__main__":
  main()
