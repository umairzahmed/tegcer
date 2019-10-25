import seaborn as sns
import numpy as np
from termcolor import colored
import pandas as pd
import os, sys
import matplotlib.pyplot as plt

from src.Base import ConfigFile as CF, Helper as H

# ----- Begin mods ----
MaxNumClasses = 50
# ----- End mods ------

class ConfRow:
    def __init__(self, actClass, classIndex, egBefore):
        self.truePos, self.falsePos, self.falseNeg = 0, 0, 0
        self.actClass = actClass
        self.classIndex = classIndex
        self.egBefore = egBefore # Store a random example (to visualize results)
        self.predClasses = {} # {label:count}

    def updatePred(self, predClass):
        if predClass not in self.predClasses: self.predClasses[predClass] = 0
        self.predClasses[predClass] += 1
    
    def getPrec(self):
        numTotal = self.truePos + self.falsePos
        if numTotal==0: return None
        else:
            return round(float(self.truePos) / numTotal, 2)

    def getRecall(self):
        numTotal = self.truePos + self.falseNeg
        if numTotal==0: return None
        else:
            return round(float(self.truePos) / numTotal, 2)

    def getTotalAct(self):
        return self.truePos + self.falseNeg
    
    def getCSV_Acc(self):
        return [self.actClass, self.getTotalAct(), self.getPrec(), self.getRecall()]

    def getCSV_Conf(self):
        return [(k,v) for k,v in H.sortDictVal(self.predClasses, reverse=True)]
    
    def __str__(self):
        cl, tot, prec, rec = self.getCSV_Acc()
        acc = '{:3} : {:5.2} , {:5.2} / {:5d} ->'.format(cl, prec, rec, tot)
        liSorted = ' '.join(['({:3},{:3})'.format(k,v) for k,v in self.getCSV_Conf() ])
        return acc + ' ' + liSorted

class ConfusionMatrix:
    def __init__(self, deepModel):
        self.deepModel = deepModel
        self.dataset = self.deepModel.dataset
        self.folderName = CF.pathOutput + '/'
        if not os.path.exists(self.folderName):
            os.makedirs(self.folderName)

        self.fnameRead = self.folderName + deepModel.modelName  + '.txt'

        self.multiClass = self.dataset.multiClass        
        self.y_test = self.dataset.y_test

        self.initConfMatrix()

    def initConfMatrix(self):
        if self.multiClass:
            self.dict_index2class = self.dataset.dict_Index_Label
            self.dict_class2index = self.dataset.dict_Label_Index
            self.dict_class2data  = self.dataset.dict_Label_DataSamples
            self.num_classes = self.dataset.num_labels
        else:
            self.dict_index2class = self.dataset.dict_Index_Cluster
            self.dict_class2index = self.dataset.dict_Cluster_Index
            self.dict_class2data  = self.dataset.dict_ClusterRaw_DataSamples
            self.num_classes = self.dataset.num_classes

        self.confMatrix = {}
        for i in self.dict_class2index:
            dataSamples = self.dict_class2data[i]
            eg = dataSamples[0].src
            self.confMatrix[i] = ConfRow(i, self.dict_class2index[i], eg)

    def getPrecRecall(self):
        sumTruePos = sum([self.confMatrix[i].truePos for i in self.confMatrix])
        sumFalseNeg = sum([self.confMatrix[i].falseNeg for i in self.confMatrix])
        sumFalsePos = sum([self.confMatrix[i].falsePos for i in self.confMatrix])

        prec   = round(float(sumTruePos) / (sumTruePos + sumFalsePos), 2)
        recall = round(float(sumTruePos) / (sumTruePos + sumFalseNeg), 2)
        return prec, recall

    def readDict(self):
        s = open(self.fnameRead).read() # The Conf Matrix string dump
        d = {}

        for l in s.split('\n'):
            actClass = int(l.split(':')[0])
            predClassCounts = l.split('>')[1].replace('(','')
            predDict = {}                
            for predClassCount in predClassCounts.split(')'):
                if predClassCount.strip() != '':
                    predClass, count = predClassCount.split(',')
                    predDict[int(predClass)] = int(count)
            d[actClass]= predDict

        return d

    def writeConfMat(self, confMat):
        headers = ['actualClass', '#test-count', 'precision', 'recall', 'egBefore', 'predClass-1', 'predCount-1']
        rows = []
        for confRow in confMat:
            
            accs = confRow.getCSV_Acc()
            egBefore = confRow.egBefore
            liSorted = confRow.getCSV_Conf()

            row = accs + [egBefore] + [j for li in liSorted for j in li]
            rows.append(row)

        H.writeCSV(CF.fnameConfMat, headers, rows)


    def calcConfMat(self):
        print colored('\n\tConfusion Matrix: ...', 'magenta')        
        predAtK = []

        for topK in [1, 3, 5]:
            countM, countN = 0, 0
            predClasses_Tests = self.deepModel.getPrediction(topK)

            for actClasses_bin, predClasses_bin in zip(self.y_test, predClasses_Tests):
                if self.multiClass:
                    # Add +1 to index since off-by-one with dict_index2class
                    actClasses_indices = [index+1 for index in range(len(actClasses_bin)) if actClasses_bin[index]==1]
                    predClasses_indices = [index+1 for index in range(len(predClasses_bin)) if predClasses_bin[index]==1]
                else:
                    actClasses_indices, predClasses_indices = [np.argmax(actClasses_bin)], predClasses_bin

                actClasses = [self.dict_index2class[index] for index in actClasses_indices]
                predClasses = [self.dict_index2class[index] for index in predClasses_indices]
                
                for actClass in actClasses:
                    if actClass in predClasses: # True-Positive
                        countM += 1
                        if topK == 1: # Conf Matrix only for Pred@1
                            self.confMatrix[actClass].truePos += 1
                            self.confMatrix[actClass].updatePred(actClass)
                        
                    else: # False-Negative: Not predicted at all
                        countN += 1
                        if topK == 1: # Conf Matrix only for Pred@1
                            self.confMatrix[actClass].falseNeg += 1
                        
                            for predClass in predClasses: # Add all confusion labels
                                self.confMatrix[actClass].updatePred(predClass)

                if topK == 1: # Conf Matrix only for Pred@1
                    for predClass in predClasses:
                        if predClass not in actClasses: # False-Positive: Predicted, but falsely
                            self.confMatrix[predClass].falsePos += 1

            if topK == 1: # Conf Matrix only for Pred@1
                sortedConfMat = [self.confMatrix[i] for i in self.getSortedConfMat()[0]]
                strConf = H.joinList(sortedConfMat)        
                self.writeConfMat(sortedConfMat)

            prec_at_k = round(100 * float(countM) / (countM + countN), 2)
            predAtK.append(prec_at_k)
            print 'Pred@{}= {}'.format(topK, prec_at_k)

        return strConf, predAtK

    def getSortedConfMat(self):
        sortedConfs = {i:self.confMatrix[i].getTotalAct() for i in self.confMatrix}
        sortedIndices = [i for i,j in sorted(sortedConfs.items(), key=lambda x: (x[1],x[0]), reverse=True)]
        sortedLabels = [self.confMatrix[i].actClass for i in sortedIndices]

        return sortedIndices, sortedLabels

    def plotConfMat(self):
        fnameWrite = self.folderName + 'confMat ' + self.deepModel.modelName + '.png'
        sortedIndices, sortedLabels = self.getSortedConfMat()

        mat = [ [self.confMatrix[i].predClasses[j] / float(self.confMatrix[i].getTotalAct()) 
                    if j in self.confMatrix[i].predClasses else 0 
                    for j in sortedIndices] for i in sortedIndices]
        mask = [[True if i==0 else False for i in row] for row in mat]

        plt.figure(figsize=(30,30)) # Inc fig size - width,height
        sns.set(font_scale=3) # And hence inc font size of labels
        ax = sns.heatmap(np.matrix(mat), linewidths=1, cmap='Reds', mask=np.matrix(mask),
            xticklabels=sortedLabels, yticklabels=sortedLabels)
        #sns.plt.show()
        plt.yticks(rotation=0) 
        plt.xticks(rotation=90) 

        # Add horizontal and vertical lines
        #for i in range(0, len(sortedLabels), 2):
        #    plt.axvline(x=i, color='k', linestyle='dashed') 
        #    plt.axhline(y=i, color='k', linestyle='dashed') 

        fig = ax.get_figure()
        fig.savefig(fnameWrite)
        plt.clf() # Clear fig

    def getPrecRecallDict(self):
        sortedIndices, sortedLabels = self.getSortedConfMat()
        totalNumTest = sum([i.getTotalAct() for i in self.confMatrix.values()] )

        data = {} # Each 'type' contains data for Precision list, followed by recall list, and count_percent list
        data['acc'] = [self.confMatrix[i].getPrec() for i in sortedIndices]
        data['acc'] += [self.confMatrix[i].getRecall() for i in sortedIndices]
        data['acc'] += [self.confMatrix[i].getTotalAct() / float(totalNumTest) for i in sortedIndices]

        data['class'] = sortedLabels*3

        length = len(self.confMatrix)
        data['type'] = ['precision']*length + ['recall']*length + ['count_percent']*length 

        return data, sortedLabels

    def plotPrecRec(self):
        fnameWrite = self.folderName + 'precRecall ' + self.deepModel.modelName + '.png'
        data, x_order = self.getPrecRecallDict()
        
        df = pd.core.frame.DataFrame(data)
        plt.figure(figsize=(30,20)) # Inc fig size - width,height
        sns.set(font_scale=2) # And hence inc font size of labels

        ax = sns.pointplot(x="class", y="acc", hue="type", data=df,
              palette={"precision": "g", "recall": "m", "count_percent":"b"}, order=x_order, 
              size=10, markers=['^','v','o']);
        
        ax.grid(True) # Show column grids
        plt.ylim(-0.1,1) # Limit y-axis values (acc) to (-0.1,1)
        plt.xticks(rotation=90) # Rotate x-axis labels by 90 deg 

        # Add horizontal lines
        prec, recall = self.getPrecRecall()
        plt.axhline(y=0.0, color='k', linestyle='--') # k = black
        plt.axhline(y=prec, color='g', linestyle='--') 
        plt.axhline(y=recall, color='m', linestyle='--') 

        # Add vertical lines
        for x in range(0, len(x_order), 2):
            plt.axvline(x=x, color='k', linestyle='dotted') 

        fig = ax.get_figure()
        fig.savefig(fnameWrite)
        plt.clf() # Clear fig


def tempPlot():
    order = ['a'+str(i) for i in range(11)]

    data = {}
    data['acc'] = list(np.arange(-0.1, 1.0, 0.1))
    data['acc'] += list(np.arange(1.0, -0.1, -0.1))
    data['acc'] += [0.5]*11
    data['class'] = order*3
    data['type'] = ['precision']*11 + ['recall']*11 + ['count_percent']*11 
    
    df = pd.core.frame.DataFrame(data)
    plt.figure(figsize=(20,30))
    sns.set(font_scale=3)

    ax = sns.pointplot(x="class", y="acc", hue="type", data=df,
              palette={"precision": "g", "recall": "m", "count_percent":"b"}, order=order[::-1], 
              size=10, markers=['^','v','o'], join=True);
    
    ax.grid(True)
    plt.ylim(-0.1,1)
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.xticks(rotation=90) 
    
    plt.axvline(x=0, color='k', linestyle='solid') 
    plt.axvline(x=1, color='k', linestyle='dashed') 
    plt.axvline(x=2, color='k', linestyle='dashdot') 
    plt.axvline(x=3, color='k', linestyle='dotted') 

if __name__=='__main__':
    #plotHeatMap()
    pass