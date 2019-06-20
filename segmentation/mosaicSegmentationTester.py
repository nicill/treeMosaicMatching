import imagePatcherAnnotator as patch
import evaluateMLCResults as trainer
import annotationMaskCreator as masker
import sys

def createCSVExcludingMosaic(csvFileName,pref,exc):
    fIn = open(csvFileName, "r")
    fOut = open(csvFileName[:-4]+"Excluding"+str(exc)+".csv", "a")

    for line in fIn:
        if pref not in line:
            fOut.write(line)

    return csvFileName[:-4]+"Excluding"+str(exc)+".csv"

def createOnlyMosaicPathcList(csvFileName,pref,exc):
    fIn = open(csvFileName, "r")

    ret=[]

    for line in fIn:
        if pref in line:
            ret.append(line.split(",")[0])

    return ret


def main(argv):
    # Read parameters
    # mosaic and layer prefix
    imagePrefix=argv[1]
    patch_size = int(argv[2])
    outputDir=argv[3]
    outputPrefix=argv[4]
    csvFileName=imagePrefix+"ALL.csv"

    numMosaics=7
    for i in range(1,numMosaics+1):
        # Break mosaics into patches, join all patches in the same folder
        #patch.main(["",imagePrefix+str(i),patch_size,outputDir,outputPrefix+str(i),str(i==1)],csvFileName)
        pass
    #now,we have done all the patches and have one single file with all the infomation,
    # now break into files excluding one mosaic each
    #Store the names of the files in a list
    csvFileList=[]
    singleMosaicFileLists=[]
    for i in range(1,numMosaics+1):
        csvFileList.append(createCSVExcludingMosaic(csvFileName,outputPrefix+str(i),i))
        singleMosaicFileLists.append(createOnlyMosaicPathcList(csvFileName,outputPrefix+str(i),i))

    #now, take every file in the list, train a model using it as ground truth
    # after training the model, predict all the labels in the excluded mosaic
    for i in range(1,numMosaics+1):
        path=""
        for x in imagePrefix.split("/")[:-1]:path+=x+"/"
        labelFileName=csvFileList[i-1].split("/")[-1]
        imageDirName=outputDir.split("/")[-1]
        outputFileNamePrefix=outputPrefix+str(i)

        eval=trainer.EvalMLCResults(path,labelFileName,imageDirName,outputFileNamePrefix)
        eval.readLabelFile()
        dict=eval.getDict()

        lr=0.0004
        #modelNames=[None,"PlanetRN50UNFFT","PlanetRN50FT"]
        mod="PlanetRN50UNFFT"
        unf=True
        preComp=not (mod is None)
        if preComp:
            eval.setModelFile(mod)
            preComp=True

        # now train the thing
        print("@@@@@@@@@@@@@@@@@@@@@@@@ computing for lr "+str(lr)+" with model "+str(mod)+" and unfreeze "+str(unf))

        try:
            compute=False
            predict=False
            eval.computePredictions(preComp,unf,lr,compute,predict,str(i))

            if(predict):
                print ("FULL AGREEMENT: ")
                print(str(eval.outputFullAgreementPercent())+"%")

                #print ("FULL AGREEMENT with False positives: ")
                #print(str(eval.outputFullAgreementWithFalsePositivesPercent())+"%")

                #print ("Partial AGREEMENT: ")
                #partialA=eval.outputPartialAgreementPercent()
                #print(str(partialA)+"%")

                #print (" NO AGREEMENT AT ALL! ")
                #print(str(100-partialA)+"%")

                print("TP evergreen "+str(eval.outputTPCategory("evergreen")) )
                print("FP evergreen "+str(eval.outputFPCategory("evergreen") ))

                print("TP decidious "+str(eval.outputTPCategory("decidious")) )
                print("FP decidious "+str(eval.outputFPCategory("decidious")) )

                print("Stats Evergreen "+str(eval.outputStatsCategory("evergreen")))
                print("Stats Decidious "+str(eval.outputStatsCategory("decidious")))

                eval.clearPredictions()

        except Exception as e:
            print(" THERE WAS SOME EXCEPTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "+str(e))

        #now take the new model and predict it all,
        predictionsFileName=imagePrefix+"Predictions"+str(i)+".csv"
        eval.computePredictionsFileList(singleMosaicFileLists[i-1],predictionsFileName,preComp,unf,lr,str(i))
        #call annotation mask creator with the file containing only the images and predictions of the excluded models
        masker.main(["",patch_size,predictionsFileName,path,outputPrefix+str(i)])

if __name__ == '__main__':
    main(sys.argv)
