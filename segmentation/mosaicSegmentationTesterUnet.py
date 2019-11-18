#import imagePatcherSemSegFastai as patch
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
import dice
import sys
import cv2

#Define classes and image names
codes = ["river","decidious","uncovered","evergreen","manmade","Void"]
layerNames=codes
layerDict={} #dictionary so that we know what number corresponds to each layer
for i in range(len(layerNames)):layerDict[layerNames[i]]=i
imagePrefixes=["wM1","wM2","wM3","wM4","wM5","wM6","wM7"]
imageDict={}
for i in range(len(imagePrefixes)):imageDict[imagePrefixes[i]]=i
#hardcoded output dir
outputDir="./outputIm/"

## TODO (maybe): change this so that only decidious and evergreen are taken into account
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

#parameters,
#compute is a boolean value to decide whether we compute the model or only load it
#validationFile is the name of a file containing the validation set (in our case, names of images corresponding to one mosaic)
#mosaicPrefix contains the information of the mosaic we are computing
#unetDir contains the data, validation files in the root and then two folders with images and patches
def trainUnet(validationFile,mosaicPrefix,unetDir,compute=False,lr=5e-3):

    # Define path for the data, needto be png with labels from 0 to Nclasses-1 including "Void"
    path=Path(unetDir)
    path_lbl = path/'labels'
    path_img = path/'images'

    #print("VALID! "+validationFile)
    # Get files and labels
    fnames = get_image_files(path_img)
    lbl_names = get_image_files(path_lbl)

    # translate the name of the image into its label file!!!!!!!!!!
    # at this moment, the names are the same with the addition of "Real" before file extension for real images
    get_y_fn = lambda x: path_lbl/f'{x.stem[:-4]}{x.suffix}'

    img_f = fnames[0]
    mask = open_mask(get_y_fn(img_f))

    src_size = np.array(mask.shape[1:])
    src_size,mask.data

    # ## Datasets
    size = src_size//2
    #size = src_size
    print("size is "+str(size))
    free = gpu_mem_get_free_no_cache()
    # the max size of bs depends on the available GPU RAM
    if free > 8200: bs=8
    else:           bs=4
    bs=32
    print(f"using bs={bs}, have {free}MB of GPU RAM free")

    src = (SegmentationItemList.from_folder(path_img)
           .split_by_fname_file("../"+validationFile)
           .label_from_func(get_y_fn, classes=codes))
    data = (src.transform(get_transforms(), size=size, tfm_y=True)
            .databunch(bs=bs)
            .normalize(imagenet_stats))
    # ## Model
    metrics=acc_camvid
    # Compute with different types of models and freezing/not freezing
    wd=1e-2
    learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
    #lr=5e-3
    modelFileName=mosaicPrefix+"Unet"+str(lr)
    if compute:
        epochs=10
        print(" training model with epochs "+str(epochs)+" and learning rate "+str(lr))
        learn.fit_one_cycle(epochs, slice(lr), pct_start=0.9)
        learn.save(modelFileName)
    else:
        learn.load(modelFileName);
        #print("LOAD/SAVE file name is wrong! uncomment previous line!")
        #learn.load('stage-1');

    return learn

def createValidationFile(allFileList,pref,outF):
    fIn = open(allFileList, "r")
    for line in fIn:
        if pref in line:
            outF.write(line)

def replaceImagePatch(image,minX,minY,size,origPatch,verbose=False):
    if(verbose):print("painting patch [ "+str(minX)+" , "+str(minX+size)+"] x [ "+str(minY)+" , "+str(minY+size)+"]")

    if(origPatch.shape[0]!=size or origPatch.shape[1]!=size ):patch=cv2.resize(origPatch,(size,size))
    else: patch=origPatch
    for i in range(size):
        for j in range(size):
            image[minX+i, minY+j]=patch[i][j]
    #image[minX:minX+size, minY:minY+size]=patch
    #cv2.imwrite("./outImg/ORIGPATCH"+str(minX)+str(minY)+str(size)+".jpg",origPatch)
    #cv2.imwrite("./outImg/PATCH"+str(minX)+str(minY)+str(size)+".jpg",patch)
    #cv2.imwrite("./outImg/im"+str(minX)+str(minY)+str(size)+".jpg",image)


def main(argv):
    # Read parameters
    # mosaic and layer prefix
    imagePrefix=argv[1]
    patch_size = int(argv[2])
    dataDir=argv[3]
    # outputFileDir should contain a file with the names of all images called "allFilesList.txt"
    allImagesFile="allFilesList.txt"
    seasonPrefix="wm"

    #this can be used to make the patches here, not active, not tested
    numMosaics=7
    #for i in range(1,numMosaics+1):
        # Break mosaics into patches, join all patches in the same folder
        #patch.main(["",imagePrefix+str(i),patch_size,outputDir,outputPrefix+str(i),str(i==1)],csvFileName)

    #now,we have all the patches and have one single file with all the names of the files
    # now break into files with the files of one mosaic each
    # Store the names of the files in a list
    validFileNameList=[]
    for i in range(1,numMosaics+1):
        validFileName="valid"+seasonPrefix+str(i)+".txt"
        outF= open(dataDir+validFileName, "w")
        validFileNameList.append(validFileName)
        createValidationFile(dataDir+allImagesFile,seasonPrefix+str(i),outF)
        outF.close()

    #initialize the list of mosaics and make a list with their shapes
    shapeX={}
    shapeY={}
    image={}
    for pref in imagePrefixes:
        imageDir=""
        for x in imagePrefix.split("/")[:-1]:imageDir=imageDir+x+"/"
        image[pref] = cv2.imread(imageDir+pref+".jpg",cv2.IMREAD_COLOR)
        print("Image "+imageDir+pref+".jpg")
        shapeX[pref]=image[pref].shape[0]
        shapeY[pref]=image[pref].shape[1]

    #create a blank image for each layer of each mosaic
    layerList=[]
    i=0
    for pref in imagePrefixes:
        layerList.append([])
        for x in range(len(layerNames)):
            layerList[i].append(np.zeros((shapeX[pref],shapeY[pref]),dtype=np.uint8))
        i+=1

    #now, take every mosaic, exclude it and train a Unet model with the images of all other mosaics
    # to do this, simply use the corresponding validationFile as the validation file in the unet
    compute=True
    lrValues=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009]
    for lr in lrValues:
        print("Starting with LR "+str(lr))
        for i in range(1,numMosaics+1):
            mosaicName=seasonPrefix+str(i)
            print("Starting Unet computation with mosaic "+mosaicName)
            learn=trainUnet(validFileNameList[i-1],mosaicName,dataDir,compute,lr)

            #now, for the trained model,
            # take all the files in the validation list
            # Predict each file
            pref=imagePrefixes[i-1]
            f=open(dataDir+validFileNameList[i-1],"r")
            for line in f:
                #predict image
                imName=line.strip()
                img = open_image(dataDir+"images/"+imName)
                patchNumber=int(line.split("h")[1].split("R")[0])

                #print("patch "+str(patchNumber))

                #predicted:1) class, 2) id (in this case pixel labels), 3) probabilities
                pred_class,pred_idx,outputs=learn.predict(img)
                #print("predicate shape "+str(pred_idx.shape)+" "+str(pred_idx))
                predicate=pred_idx.reshape((pred_idx.shape[1],pred_idx.shape[2]))
                #print("predicate shape "+str(predicate.shape))

                numStepsX=int(shapeX[pref]/patch_size)
                numStepsY=int(shapeY[pref]/patch_size)

                for l in codes:

                    if l not in ["Void"]:

                        #print("Code "+str(l))
                        #now, paint the information of each patch in the layer where it belongs
                        xJump=patchNumber//numStepsY
                        yJump=patchNumber%numStepsY

                        #print("imageDict[pref]"+str(imageDict[pref]))
                        #print("layerDict[l]"+str(layerDict[l]))

                        currentLayerIm=layerList[imageDict[pref]][layerDict[l]]
                        patch=np.zeros((predicate.shape[0],predicate.shape[1]),dtype=np.uint8)
                        patch.fill(0)
                        for a in range(predicate.shape[0]):
                            for b in range(predicate.shape[1]):
                                if predicate[a][b]==layerDict[l]:
                                    #print("found a pixel "+str(a)+" "+str(b)+" "+str(l))
                                    patch[a][b]=255

                        #patch[predicate==layerDict[l]]=255
                        #cv2.imwrite("./outImg/REALPATCH"+str(patchNumber)+".jpg",patch)
                        #cv2.imwrite(outputDir+pref+"patch"+str(patchNumber)+l+".jpg",patch)
                        replaceImagePatch(currentLayerIm,xJump*patch_size,yJump*patch_size,patch_size,patch)

            for l in codes:

                if l not in ["Void"]:

                    # We are finished, store all layer images
                    predictedLayer=layerList[imageDict[pref]][layerDict[l]]
                    cv2.imwrite(outputDir+pref+"UnetMaskLayer"+str(l)+".jpg",predictedLayer)

                    #Also output DICE
                    #load manual annotation
                    manualLayer=cv2.imread(imageDir+pref+"layer"+str(layerDict[l])+".jpg",cv2.IMREAD_GRAYSCALE)
                    if manualLayer is not None:
                        manualLayer=np.invert(manualLayer)
                        currentDice=dice.dice(predictedLayer,manualLayer )
                        print("*******************************************dice Unet "+l+" "+str(currentDice))
                    else:
                        print("*******************************************dice Unet "+l+" NO LAYER ")



if __name__ == '__main__':
    main(sys.argv)
