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
    aux3=mainMask.copy()

    # First, the new unknown part touching background in the accumulated image is copied
    # things that were previously known but are unknown to me stay known
    #aux1[newMask==1]=2# now we only have unknown as zero and other things at least 2
    #aux1[mainMask==1]+=1 #now 1 contains the pixels that where 0 in new and 1 in main
    #mainMask[aux1==1]=0 # unknown + background is unknown
    #aux1=newMask.copy()
    #aux1[aux2>1]=0 #erase everything not touching mainmask unknown or background
    #aux1[aux1==1]=0 #erase also background
    #mainMask=mainMask|aux1 # add the labels

    #1) copy all the unknown (kills background and old lables)
    mainMask[newMask==0]=0

    #2) now add the labels (at some points double adding!)
    aux1[aux1==1]=0 #erase the background
    aux2[aux2==1]=0 #erase the background
    mainMask=mainMask|aux1 # add the labels

    # 3) now, if there is a label in both, put unknown (kills new and old labels)
    aux2[aux1>1]=1
    aux2[aux3>1]+=1 #now 2 in aux2 marks labels in the two
    mainMask[aux2==2]=0 #label plus label is unknown, correcting souble added layers

    return mainMask

def buildBinaryMask(markers,firstLabel,lastLabel):
    if firstLabel==0:firstLabel=2
    aux=markers.copy()
    #erase background
    aux[markers==1]=0
    #mark the pertinent labels
    for x in range(firstLabel,lastLabel+1):
        aux[markers==x]=1

    #erase all other labels
    aux[aux!=1]=255

    #maybe make it 255 instead of 1?

    return aux

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
    for pref in imagePrefixes:
        layerList.append([])
        #mask accumulator image
        maskImage=np.ones((shapeX,shapeY),dtype=np.uint8)
        firstLabel=0 #counter so labels from different masks have different labels
        firstLabelList=[0]
        for x in range(len(layerNames)):
            if layerNames[x] in ["river","decidious","uncovered","evergreen"]:
                print("starting "+layerNames[x])

                # Try to refine the segmenation
                opening = cv2.morphologyEx(layerList[i][x],cv2.MORPH_OPEN,kernel, iterations = 2)

                # sure background area
                sure_bg = cv2.dilate(opening,kernel,iterations=100)
                #cv2.imwrite(imageDir+pref+"ErodedLayer"+str(x)+".jpg",opening)

                # Finding sure foreground area
                dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)

                # Finding unknown region
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg,sure_fg)

                # Marker labelling
                ret, markers = cv2.connectedComponents(sure_fg)

                # Add one to all labels so that sure background is not 0, but 1, also add firstLabel so label numbers are different
                markers = markers+firstLabel+1

                #remark sure background as 1
                markers[markers==(firstLabel+1)]=1

                firstLabel+=ret
                firstLabelList.append(firstLabel)
                # Now, mark the region of unknown with zero
                markers[unknown==255] = 0

                maskImage=addNewMaskLayer(markers,maskImage)

                #cv2.imwrite(imageDir+pref+"GeneratedLayer"+str(x)+".jpg",layerList[i][x])
                cv2.imwrite(str(x)+"layerMask.jpg",cv2.applyColorMap(np.uint8(markers*50),cv2.COLORMAP_JET))
                cv2.imwrite(str(x)+"AccumMask.jpg",cv2.applyColorMap(np.uint8(maskImage*50),cv2.COLORMAP_JET))
            else:
                print("skypping layer "+layerNames[x])

        cv2.imwrite("finalMask.jpg",cv2.applyColorMap(np.uint8(maskImage*50),cv2.COLORMAP_JET))

        print("starting watershed ")
        markers = cv2.watershed(image,maskImage)
        image[markers == -1] = [0,0,255]

        cv2.imwrite(str(i)+"watershed.jpg",image)
        cv2.imwrite(str(i)+"markers.jpg",cv2.applyColorMap(np.uint8(markers*50),cv2.COLORMAP_JET))

        # now we should reconstruct the individual mask segmenations from the final marker
        print(" list Of first labels"+str(firstLabelList))

        #now, make layer images, for every interval of layers, only include markers inside of it
        for j in range(1,len(firstLabelList)):
            im=buildBinaryMask(markers,firstLabelList[j-1],firstLabelList[j])
            cv2.imwrite(str(i)+"and"+str(j-1)+"binaryMask.jpg",np.uint8(im))


        i+=1






if __name__== "__main__":
  main()
