import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import cv2
import numpy as np
import sys
from fastai import *
from fastai.vision import *

#python evaluateMLCResults.py /media/yago/workDrive/Experiments/forests/segmentation/wm1/ wM1.csv  patches

class EvalMLCResults():
    def __init__(self,path,labelFileName,imageDir,outputFilePrefix,imageSuff=".jpg"):
        self.fileDict={}
        self.validFileDict={}
        self.predList=[]
        self.path=path
        self.labelFileName=labelFileName
        self.imageDir=imageDir
        self.suffix=imageSuff
        self.outputFilePrefix=outputFilePrefix

    def setModelFile(self,modelFile):
        self.modelFile=modelFile

    def getDict(self):return self.fileDict

    def getValidationDict(self):return self.validFileDict

    def readLabelFile(self):
        f = open(self.path+self.labelFileName, "r")
        #discard first line
        firstLine=True
        for x in f:
            if firstLine:
                firstLine=False
            else:
                auxList=x.split(",")
                self.fileDict[auxList[0]]=[Label(auxList[1].split())]

    def clearPredictions(self):
        for key, val in self.fileDict.items():
            val.pop()

    def computePredictions(self,pretrainedModel=False,unfreeze=False,lr=0.01,compute=True):

        #set up and load model
        path = Config.data_path()/self.path
        path.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.path+self.labelFileName)
        df.head()

        tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
        np.random.seed(42)
        src = (ImageList.from_csv(path, self.labelFileName, folder=self.imageDir, suffix=self.suffix).split_by_rand_pct(0.2).label_from_df(label_delim=' '))
        data = (src.transform(tfms, size=128).databunch().normalize(imagenet_stats))
        arch = models.resnet50
        acc_02 = partial(accuracy_thresh, thresh=0.2)
        f_score = partial(fbeta, thresh=0.2)
        learn = cnn_learner(data, arch, metrics=[acc_02, f_score])
        if pretrainedModel : learn.load(self.modelFile) # If we are starting with a pre-trained model, load it

        numEpochs=10
        if unfreeze:
            learn.unfreeze()
            if compute: learn.fit_one_cycle(numEpochs, slice(lr, lr/5))
        else:
            if compute: learn.fit_one_cycle(numEpochs, slice(lr))

        saveFileName="treesRN50lr"+str(lr)
        if unfreeze: saveFileName="UNF"+saveFileName
        if pretrainedModel : saveFileName='Pret'+self.modelFile+saveFileName

        if compute:learn.save(saveFileName)
        else: learn.load(saveFileName)
        print("model loaded, now compute predictions ")
        outputFileName=self.outputFilePrefix+saveFileName+".csv"
        of=open(outputFileName,"w")

        #now, go over the images in self.fileDict and complete the predictions
        for key, val in self.fileDict.items():
            # we might be missing applying the transforms here!
            img = open_image(self.path+self.imageDir+"/"+key+self.suffix)
            # maybe the threshold for prediction can be changed here?
            pred_class,pred_idx,outputs =learn.predict(img)
            val.append(Label(str(pred_class).split(";")))
            #print(str(key)+" "+str(pred_class))
            of.write(str(key)+" "+str(pred_class)+"\n")
            of.flush()
            #print (pred_class)

        of.close()

        # finally, store only the validation set in the validation dictionary
        valid=data.valid_ds.x.items[:]
        print("validation data ")
        for x in valid:
            aux=x.split("/")[-1].split(".")[0]
            # now copy the correct labels on to the validation dictionary
            self.validFileDict[aux]=self.fileDict[aux]
            #print(self.validFileDict[aux])

        #also, store as a csv file for clearPredictions




    def outputFullAgreementPercent(self,valid=True):
        countAgreement=0
        countTotal=0
        if valid:dict=self.getValidationDict()
        else:dict=self.getDict()

        for key, val in dict.items():
            countTotal+=1
            if val[0].totalAgreement(val[1]) :countAgreement+=1
            #else: print("no full agreement for "+str(key)+" "+str(val[0])+" and "+str(val[1]))
        print(" ************************************************ Full agreeement in "+str(countAgreement)+" out of "+str(countTotal) )
        return countAgreement*100/len(dict)

    def outputFullAgreementWithFalsePositivesPercent(self,valid=True):
        if valid:dict=self.getValidationDict()
        else:dict=self.getDict()
        countAgreement=0
        for key, val in dict.items():
            if val[0].totalAgreementWithFP(val[1]) :countAgreement+=1
        return countAgreement*100/len(dict)

    def outputPartialAgreementPercent(self,valid=True):
        countAgreement=0
        if valid:dict=self.getValidationDict()
        else:dict=self.getDict()
        for key, val in dict.items():
            if val[0].partialAgreement(val[1]) :countAgreement+=1
            #else: print("no agreement for "+str(key)+" "+str(val[0])+" and "+str(val[1]))
        return countAgreement*100/len(dict)

    # receives a label name and returns how many true positives were computed
    def outputTPCategory(self,cat,valid=True):
        countAgreement=0
        countTotal=0
        if valid:dict=self.getValidationDict()
        else:dict=self.getDict()

        for key, val in dict.items():
            # first, check if this patch contains the category
            if val[0].containsLabel(cat):
                countTotal+=1
                if val[1].containsLabel(cat):countAgreement+=1
        #print(" ************************************************ True Positives for  "+str(cat)+"were "+str(countAgreement)+" of "+str(countTotal))
        return countAgreement*100/countTotal

    # receives a label name and returns how many false positives were computed
    def outputFPCategory(self,cat,valid=True):
        countAgreement=0
        countTotal=0
        if valid:dict=self.getValidationDict()
        else:dict=self.getDict()

        for key, val in dict.items():
            # first, check if this patch contains the category
            if val[0].containsLabel(cat):
                countTotal+=1
            elif val[1].containsLabel(cat) :countAgreement+=1
        #print(" ************************************************ False Positives for  "+str(cat)+"were "+str(countAgreement)+" of "+str(countTotal))
        return countAgreement*100/countTotal

    def outputStatsCategory(self,cat,valid=True):
        if valid:dict=self.getValidationDict()
        else:dict=self.getDict()

        listPairs=[(val[0].containsLabel(cat),val[1].containsLabel(cat)) for key, val in dict.items() ]

        TP=listPairs.count((True,True))
        FP=listPairs.count((False,True))

        TN=listPairs.count((False,False))
        FN=listPairs.count((True,False))

        return len(dict),TP,FP,TN,FN

class Label():
    #content is a list of strings
    def __init__(self,content):self.content=content

    def __str__(self):
        return str(self.content)
    def __repr__(self):
        return str(self.content)

    # everything is predicted right
    def totalAgreement(self,pred):
        for x in self.content:
            if x not in pred.content:return False
        #print("total or partial agreement "+str(self.content)+" and "+str(pred.content)+ "returning "+str(len(pred.content)==len(self.content)))
        return (len(pred.content)==len(self.content))

    # everything is predicted right but there are some false-positives (classifier found things that were not there)
    def totalAgreementWithFP(self,pred):
        for x in self.content:
            if x not in pred.content:return False
        return True

    #We got something right at least
    def partialAgreement(self,pred):
        for x in self.content:
            if x in pred.content:return True
        return False

    def containsLabel(self,l):return l in self.content

def main():

    # load label File
    path=sys.argv[1]
    labelFileName=sys.argv[2]
    imageDirName=sys.argv[3]
    outputFileNamePrefix=sys.argv[4]

    #print(labelFileName)
    eval=EvalMLCResults(path,labelFileName,imageDirName,outputFileNamePrefix)

    eval.readLabelFile()
    dict=eval.getDict()
    #print(dict["patch117"])

    lrValues=[0.1,0.2,0.3,0.4,0.5,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    #lrValues=[0.1,0.2,0.3,0.4,0.5,0.01,0.02,0.03,0.04,0.05,0.001,0.002,0.003,0.004,0.005,0.0001,0.0002,0.0003,0.0004,0.0005,0.00001,0.00002,0.00003,0.00004,0.00005]
    #lrValues=[0.5,0.25,1,0.15,0.75,0.00001,0.00005,00000.5,00000.1]
    #modelNames=[None]
    modelNames=[None,"PlanetRN50UNFFT","PlanetRN50FT"]
    unfValues=[True,False]
    for lr in lrValues:
        for mod in modelNames:
            for unf in unfValues:
                preComp=False
                if not mod is None:
                    eval.setModelFile(mod)
                    preComp=True

                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ computing for lr "+str(lr)+" with model "+str(mod)+" and unfreeze "+str(unf))

                try:
                    compute=False
                    eval.computePredictions(preComp,unf,lr,compute)

                    print ("FULL AGREEMENT: ")
                    print(str(eval.outputFullAgreementPercent())+"%")

                    print ("FULL AGREEMENT with False positives: ")
                    print(str(eval.outputFullAgreementWithFalsePositivesPercent())+"%")

                    print ("Partial AGREEMENT: ")
                    partialA=eval.outputPartialAgreementPercent()
                    print(str(partialA)+"%")

                    print (" NO AGREEMENT AT ALL! ")
                    print(str(100-partialA)+"%")

                    print("TP evergreen "+str(eval.outputTPCategory("evergreen")) )
                    print("FP evergreen "+str(eval.outputFPCategory("evergreen") ))

                    print("TP decidious "+str(eval.outputTPCategory("decidious")) )
                    print("FP decidious "+str(eval.outputFPCategory("decidious")) )

                    print("Stats Evergreen "+str(eval.outputStatsCategory("evergreen")))
                    print("Stats Decidious "+str(eval.outputStatsCategory("decidious")))

                    eval.clearPredictions()
                except Exception as e:
                    print(" THERE WAS SOME EXCEPTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "+str(e))
                #clear predictions

if __name__== "__main__":
  main()
