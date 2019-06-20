import cv2
import numpy as np
import sys
import imagePatcherAnnotator as impa
import dice as dice
import skimage.segmentation as seg

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
        aux[markers==x]=255

    #erase the other markers
    aux[aux<255]=0

    return aux

def main(argv):
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
    patch_size = int(argv[1])
    csvFile=argv[2]
    #imageDir, full path!
    imageDir=argv[3]

    #read also all the prefixes of all the images that we have
    imagePrefixes=[]
    for x in range(4,len(argv)):imagePrefixes.append(argv[x])
    imageDict={}
    for i in range(len(imagePrefixes)):imageDict[imagePrefixes[i]]=i

    #hardcoded output dir
    outputDir="./outputIm/"

    #print("AnnotationMask creator main, parameters: csv files: "+str(csvFile)+" image directory"+str(imageDir)+" image prefixes "+str(imagePrefixes))

    f = open(csvFile, "r")
    shapeX={}
    shapeY={}
    image={}
    for pref in imagePrefixes:
        image[pref] = cv2.imread(imageDir+pref+".jpg",cv2.IMREAD_COLOR)
        print("Image "+imageDir+pref+".jpg")
        shapeX[pref]=image[pref].shape[0]
        shapeY[pref]=image[pref].shape[1]

    #create a blank image for each layers
    layerList=[]
    i=0
    for pref in imagePrefixes:
        layerList.append([])
        for x in range(len(layerNames)):
            layerList[i].append(np.zeros((shapeX[pref],shapeY[pref]),dtype=np.uint8))
        i+=1

    # go over the csv file, for every line
        # extract the image prefixes
        # extract the lables
        # for every label found, paint a black patch in the correspoding image layer
    for line in f:
        #process every line
        #print(line)
        pref=line.split("p")[0]
        patchNumber=int(line.split("h")[1].split(" ")[0])
        labelList=line.split(" ")[1].strip().split(";")
        numStepsX=int(shapeX[pref]/patch_size)
        numStepsY=int(shapeY[pref]/patch_size)

        for x in labelList:
            if x=="":break
            #now, paint the information of each patch in the layer where it belongs
            xJump=patchNumber//numStepsY
            yJump=patchNumber%numStepsY

            #now find the proper layer (once in the im)
            currentLayerIm=layerList[imageDict[pref]][layerDict[x]]
            impa.paintImagePatch(currentLayerIm,xJump*patch_size,yJump*patch_size,patch_size,255)

    i=0
    for pref in imagePrefixes:
        for x in range(len(layerNames)):
            #print("shape of current layer image "+repr(layerList[i][x].shape))
            cv2.imwrite(outputDir+pref+"layer"+str(x)+".jpg",layerList[i][x])
        i+=1

    i=0
    kernel = np.zeros((3,3),np.uint8)
    kernel[:]=255
    for pref in imagePrefixes:
        print("starting with prefix "+pref)
        layerList.append([])
        #mask accumulator image
        maskImage=np.ones((shapeX[pref],shapeY[pref]),dtype=np.uint8)
        firstLabel=0 #counter so labels from different masks have different labels
        firstLabelList=[0]
        for x in range(len(layerNames)):
            if layerNames[x] in ["river","decidious","uncovered","evergreen"]:
                #print("starting "+layerNames[x])

                # Try to refine the segmenation
                opening = cv2.morphologyEx(layerList[i][x],cv2.MORPH_OPEN,kernel, iterations = 2)

                # sure background area
                sure_bg = cv2.dilate(opening,kernel,iterations=10)

                # Finding sure foreground area
                dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                ret, sure_fg = cv2.threshold(dist_transform,0.17*dist_transform.max(),255,0)

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

                cv2.imwrite(outputDir+pref+"CoarseMaskLayer"+str(layerNames[x])+".jpg",layerList[i][x])
                #cv2.imwrite(outputDir+str(x)+"layerMask.jpg",cv2.applyColorMap(np.uint8(markers*50),cv2.COLORMAP_JET))
                #cv2.imwrite(outputDir+pref+str(x)+"AccumMask.jpg",cv2.applyColorMap(np.uint8(maskImage*50),cv2.COLORMAP_JET))
            else:
                pass
                #print("skypping layer "+layerNames[x])

        cv2.imwrite(outputDir+"finalMask.jpg",cv2.applyColorMap(np.uint8(maskImage*50),cv2.COLORMAP_JET))

        #print("starting watershed ")
        markers = cv2.watershed(image[pref],maskImage)
        #markers = seg.random_walker(image[pref],maskImage)

        image[pref][markers == -1] = [0,0,255]

        cv2.imwrite(outputDir+pref+str(i)+"watershed.jpg",image[pref])
        cv2.imwrite(outputDir+pref+str(i)+"markers.jpg",cv2.applyColorMap(np.uint8(markers*50),cv2.COLORMAP_JET))

        # now we should reconstruct the individual mask segmenations from the final marker
        #print(" list Of first labels"+str(firstLabelList))

        #now, make layer images, for every interval of layers, only include markers inside of it
        #while we are doing it, we can also compute the DICE coefficient
        for j in range(1,len(firstLabelList)):
            refinedLayer=buildBinaryMask(markers,firstLabelList[j-1],firstLabelList[j])
            refinedLayer=np.uint8(refinedLayer)
            coarseLayer=layerList[i][j-1]
            manualLayer=np.invert(cv2.imread(imageDir+pref+"layer"+str(j-1)+".jpg",cv2.IMREAD_GRAYSCALE))
            #cv2.imwrite(outputDir+pref+str(i)+str(j-1)+"manual.jpg",manualLayer)
            #cv2.imwrite(outputDir+pref+str(i)+str(j-1)+"coarse.jpg",coarseLayer)
            cv2.imwrite(outputDir+pref+str(layerNames[j-1])+"refined.jpg",refinedLayer)
            print(" LAYER "+layerNames[j-1])
            currentDice=dice.dice(coarseLayer,manualLayer )
            print("*******************************************dice coarse mask "+str(currentDice))
            currentDice=dice.dice(refinedLayer,manualLayer )
            print("*******************************************dice refined mask "+str(currentDice))
        i+=1


        #now. compute the



if __name__== "__main__":
  main(sys.argv)
