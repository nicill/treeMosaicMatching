import matplotlib.pyplot as plt
import csv
import sys

def main(argv):

    x = []
    y = []

    #print("opening "+str(argv[1]))
    yLabel=""
    xLabel=[]
    plt.figure(dpi=800)
    with open(argv[1],'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            #first row, find out number of categories
            if yLabel=="":
                yLabel=row[0]
                for i in range(1,len(row)):
                    x.append([])
                    xLabel.append(row[i])
            else:
                #print(row)
                y.append(float(row[0]))
                for i in range(len(row)-1):
                    x[i].append(float(row[i+1]))

    #print(x)
    for i in range(len(x)):
        plt.plot(y,x[i], label=xLabel[i])

    code=int(argv[2])

    labelsLong=["Total Agreement","Total Agreement With False Positives","Partial Agreement","Accuracy","Specificity","Sensitivity"]
    labelsShort=["TA","TA FP","PA","ACC","SPEC","SENS"]

    plt.xlabel(yLabel)
    plt.ylabel(labelsShort[code]+" %")
    plt.xscale('log')
    plt.title(labelsLong[code]+' %')
    plt.legend()
    #plt.show()

    plt.savefig(argv[3])

if __name__ == '__main__':
    main(sys.argv)
