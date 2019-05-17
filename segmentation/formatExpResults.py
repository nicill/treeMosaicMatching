import sys
import abc

class resultFileProcesser():
    def __init__(self,fileName1,fileName2):
        self.inFileName=fileName1
        self.outFileName=fileName2
    def run(self):
        #print("Processing File")
        self.f1 = open(self.inFileName, "r")
        self.f2 = open(self.outFileName, "a")
        for line in self.f1:
            self.processLine(line)


    @abc.abstractmethod
    def  processLine(self,line):
        " Method to process whatever is done in each experiment, not implemented in the mother class "
        return

class TLresultFileProcesser(resultFileProcesser):
    NUMBER_STAGES=5 #including 2 for stats
    def __init__(self,fileName1,fileName2):
        print("model;LR;Unfreeze;TA;TAFP;PA")
        super().__init__(fileName1,fileName2)
        self.restart()

    def restart(self):
        self.modelName=""
        self.stage=0
        self.TA=0
        self.TAFP=0
        self.PA=0
        self.output=""

    def upStage(self):
        self.stage+=1

    def stageChange(linePart):
        return ("FULL AGREEMENT" in linePart) or ("Partial AGREEMENT" in linePart)

    def statsStage(linePart):
        return "Stats " in linePart

    def isFloat(linePart):
        try:
            a=float(linePart)
            return True
        except ValueError:
            return False
    def lastStage(self): return self.stage==self.NUMBER_STAGES
    def activeStage(self): return not self.stage==0

    def processLine(self,line):
        # if we find an exception, restart
        linePart=line[0:100]
        #print(line)
        #first, process stage changes
        if "EXCEPTION" in linePart:self.restart()
        elif "computing" in linePart:
            listOfWords=line.split(" ")
            self.output+=listOfWords[7]+";"+listOfWords[4]+";"+listOfWords[10].strip()
        elif TLresultFileProcesser.stageChange(linePart) :
            #print("yelou "+linePart+" "+str(self.stage))
            self.upStage()
        elif TLresultFileProcesser.isFloat(linePart.replace("%"," ").strip()):
            if self.activeStage():self.output+=";"+linePart.replace("%"," ").strip()
        elif TLresultFileProcesser.statsStage(linePart):
            listOfWords=line.strip().split("(")
            category=listOfWords[0].split(" ")[1]
            otherlistOfWords=(listOfWords[1])[0:-1]
            finalString=otherlistOfWords.split(",")

            total=finalString[0].strip()
            TP=finalString[1].strip()
            FP=finalString[2].strip()
            TN=finalString[3].strip()
            FN=finalString[4].strip()

            self.output+=";"+category+";"+total+";"+TP+";"+FP+";"+TN+";"+FN+";"+";"+";"+";"
            self.upStage()
        #check if we need to restart
        if(self.lastStage()):
                print(self.output)
                self.f2.write(self.output+"\n")
                self.restart()



def main():

    rP=TLresultFileProcesser(sys.argv[1],sys.argv[2])
    rP.run()

if __name__ == '__main__':
    main()
