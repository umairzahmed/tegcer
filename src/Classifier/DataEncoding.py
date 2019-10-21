'''Defines a class/wrapper for the entire dataset along with the operations performed on it (loading of data,
pre/post-processing, and converting it to the desired format before feeding to network, etc)'''

import numpy as np
import keras, pandas
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from collections import defaultdict
from termcolor import colored
import random, math, os, sys, re

from src.Base import ConfigFile as CF, Helper as H

class DataSample:
    def addRaw_Space(self):
        punctuation = re.escape('!#$%&*+-/<=>\\^|~')
        separator = re.escape('"\'(),;:.[]{}`_@?')

        self.rawText = self.rawText.replace('\t', ' ')
        # club punctuations. == and = are 2 diff Tokens
        self.rawText = re.sub('([' + punctuation + ']+)', r' \1 ', self.rawText)  
        # split each separator. (( is 2 tokens
        self.rawText = re.sub('([' + separator + '])', r' \1 ', self.rawText)  

    def addRaw_EOL(self):
        if CF.FLAG_ADD_EOL: # Recommended to always set this flag to True (otherwise, a\nb will a treated as new Token)
            self.rawText = re.sub(r'[\r\n]', ' <EOL> ', self.rawText)

    def addRaw_ErrSet(self):
        ''' Prefix the errSet to training input (feature)'''
        errSets = [CF.ERRSET_PRE + i for i in self.errSet]
        self.rawText = ' '.join(errSets) + CF.ERRSET_SEP + self.rawText

    def addRaw_Bigram(self):
        for bigram in H.pairwise(self.rawText.split()):
            p1, p2 = bigram
            biRawText = p1 +'<BIGRAM>'+ p2
            self.rawText += ' '+ biRawText

    def addRaw(self):
        #self.addRaw_Space()
        self.addRaw_EOL()
        self.addRaw_Bigram()
        self.addRaw_ErrSet()

        #print self.rawText
    
    def getSortedErrSet(self, errSet):
        return sorted([e.strip() for e in errSet.strip().split(';') if e.strip()!=''])

    def addCluster(self):
        if 'errset' in self.predSet.lower():
            clusterList = self.clusterRaw.strip().splitlines()
            errSet = self.getSortedErrSet(clusterList[0])
            restData = '\n'.join(sorted(clusterList[1:]))

            self.labelList = [i.strip() for i in errSet if i.strip()!='']
            self.clusterRaw = (';'.join(sorted(self.labelList)) +';\n'+ restData).strip() #  
       

    def __init__(self, rawText, errSet, clusterRaw, predSet, src=None, trgt=None):
        self.src = src   # The Original "source"/buggy line, if passed
        self.trgt = trgt # The Original "target"/fixed line, if passed
        self.rawText = rawText.strip()
        self.errSet = self.getSortedErrSet(errSet)
        
        self.clusterRaw = clusterRaw
        self.labelList = clusterRaw
        self.predSet = predSet

        self.addCluster()
        self.addRaw()
        


class DataSet:
    # Object functions
    def __init__(self, trainSet, predictSet, multiClass, MIN_TRAIN_SIZE):
        self.trainSet, self.predictSet, self.multiClass = trainSet, predictSet, multiClass
        self.X_train_DataSample, self.y_train_cluster, self.y_train_label = [], [], []
        self.X_valid_DataSample, self.y_valid_cluster, self.y_valid_label = [], [], []
        self.X_test_DataSample, self.y_test_cluster, self.y_test_label = [], [], []

        # Each DataSample can have multiple labels. 
        # Multiple-labels form a "cluster-index", if self.multiClass=True
        self.dict_ClusterRaw_DataSamples, self.dict_Label_DataSamples = {}, {} 

        # Mapping of Cluster/Label to Index and vice versa 
        self.dict_Cluster_Index, self.dict_Label_Index = {}, {}
        self.dict_Index_Cluster, self.dict_Index_Label = {}, {}
        
        self.num_classes = 1 # One more than what is required
        self.num_labels = 1
        self.max_seq_length, self.max_vocab_size = 0, 0
        self.MIN_TRAIN_SIZE = MIN_TRAIN_SIZE

        self.mlb, self.tokenizer = None, None

    def shuffleClusters(self):
        print colored('\tShuffling Clusters ...', 'magenta')
        
        for clusterRaw in self.dict_ClusterRaw_DataSamples:
            l = self.dict_ClusterRaw_DataSamples[clusterRaw]
            random.seed(CF.MY_SEED) # fix random seed for reproducibility
            random.shuffle(l)
            self.dict_ClusterRaw_DataSamples[clusterRaw] = l

    def splitTrainTest(self):
        print colored('\tSplitting Train+Test ...', 'magenta')    
        self.num_classes = len(self.dict_ClusterRaw_DataSamples) + 1
        self.num_labels = len(self.dict_Label_DataSamples) 
        print 'NumClasses=', self.num_classes - 1
        print 'NumLabels=', self.num_labels - 1

        for clusterRaw, dataSamples in H.sortDictLen_Rev(self.dict_ClusterRaw_DataSamples):
            clusterIndex = self.dict_Cluster_Index[clusterRaw]
            li = dataSamples
            labelList = li[0].labelList # Pick any dataSamples labelList - would be the same for all similar clusterRaw
            labelIndices = [self.dict_Label_Index[label] for label in labelList]

            numTrain = int(math.ceil(CF.TRAIN_SPLIT * len(li)))
            numValid = int(math.floor(CF.VALIDATION_SPLIT * len(li)))
            numTest =  len(li) - numTrain - numValid
            print 'Class-',clusterIndex, 'NumTrain=', numTrain, 'NumValid=', numValid, 'NumTest=', numTest
            self.X_train_DataSample.extend(li[:numTrain])
            self.X_valid_DataSample.extend(li[numTrain : numTrain+numValid])
            self.X_test_DataSample.extend(li[numTrain + numValid :])

            self.y_train_cluster.extend([clusterIndex] * numTrain)
            self.y_valid_cluster.extend([clusterIndex] * numValid)
            self.y_test_cluster.extend([clusterIndex] * (len(li) - numTrain - numValid))
            
            self.y_train_label.extend([labelIndices] * numTrain)
            self.y_valid_label.extend([labelIndices] * numValid)
            self.y_test_label.extend([labelIndices] * (len(li) - numTrain - numValid))
    
        self.X_train_rawText = [i.rawText for i in self.X_train_DataSample]
        self.X_valid_rawText = [i.rawText for i in self.X_valid_DataSample]
        self.X_test_rawText = [i.rawText for i in self.X_test_DataSample] 

    def vectorizeInput(self):
        print colored('\tVectorizing sequences ...', 'magenta')     
        char_level = False
        # if 'text' in self.trainSet.lower(): char_level = True # If training on raw text (concrete programs), set char_level to Frue
        # else: char_level = False # Otherwise, set char_level to False (while training on abstractions)
        self.tokenizer = Tokenizer(filters='', lower=False, char_level=char_level)

        self.tokenizer.fit_on_texts(self.X_train_rawText)
        self.tokenizer.num_words = len(self.tokenizer.word_index) + 1
        self.max_vocab_size = self.tokenizer.num_words

        self.X_train_seq = self.tokenizer.texts_to_sequences(self.X_train_rawText)
        self.X_valid_seq = self.tokenizer.texts_to_sequences(self.X_valid_rawText)
        self.X_test_seq = self.tokenizer.texts_to_sequences(self.X_test_rawText)
        self.max_seq_length = max([len(i) for i in self.X_train_seq + self.X_valid_seq + self.X_test_seq])
        print 'Max-seq-length=', self.max_seq_length
        print 'Max-vocab-size=', self.max_vocab_size

        # truncate and pad input sequences
        self.X_train_seq = sequence.pad_sequences(self.X_train_seq, padding='post', maxlen=self.max_seq_length)
        self.X_valid_seq = sequence.pad_sequences(self.X_valid_seq, padding='post', maxlen=self.max_seq_length)
        self.X_test_seq = sequence.pad_sequences(self.X_test_seq, padding='post', maxlen=self.max_seq_length)

        # Binary vecotrs (occurance)
        self.X_train_bin = self.tokenizer.texts_to_matrix(self.X_train_rawText, mode='binary')
        self.X_valid_bin = self.tokenizer.texts_to_matrix(self.X_valid_rawText, mode='binary')
        self.X_test_bin = self.tokenizer.texts_to_matrix(self.X_test_rawText, mode='binary')
    
    def vectorizeIO_NewTest(self):
        self.X_test_seq = self.tokenizer.texts_to_sequences(self.X_test_rawText)
        self.X_test_seq = sequence.pad_sequences(self.X_test_seq, padding='post', maxlen=self.max_seq_length)
        self.X_test_bin = self.tokenizer.texts_to_matrix(self.X_test_rawText, mode='binary')

        self.y_test_binClass  = keras.utils.to_categorical(self.y_test_cluster, self.num_classes)
        self.y_test_binLabel  = self.mlb.transform(self.y_test_label)

    def vectorizeOutput(self):
        # Convert class vector to binary (single) one-hot matrix (for use with softmax layer)
        self.y_train_binClass = keras.utils.to_categorical(self.y_train_cluster, self.num_classes)
        self.y_valid_binClass = keras.utils.to_categorical(self.y_valid_cluster, self.num_classes)
        self.y_test_binClass  = keras.utils.to_categorical(self.y_test_cluster, self.num_classes)


        # Convert labels to binary (multi) one-hot matrix (for use with sigmoid layer)
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(self.y_train_label + self.y_valid_label + self.y_test_label)
        self.y_train_binLabel = self.mlb.transform(self.y_train_label)
        self.y_valid_binLabel = self.mlb.transform(self.y_valid_label)
        self.y_test_binLabel  = self.mlb.transform(self.y_test_label)

    def setDefaultTrainSet(self, modelName):
        # Use integer seq by default, unless specified
        self.X_train, self.X_valid, self.X_test = self.X_train_seq, self.X_valid_seq, self.X_test_seq 

        if self.multiClass:
            self.y_train, self.y_valid, self.y_test = self.y_train_binLabel, self.y_valid_binLabel, self.y_test_binLabel
        else:    
            self.y_train, self.y_valid, self.y_test = self.y_train_binClass, self.y_valid_binClass, self.y_test_binClass

        if 'bin' in modelName.lower():
            self.X_train, self.X_valid, self.X_test = self.X_train_bin, self.X_valid_bin, self.X_test_bin

    def getTotalNumPairs(self):
        return len(self.y_train) + len(self.y_valid) + len(self.y_test)

    def setDict_Indices(self):
        '''Once raw cluster/label dicts are created, assign indices to them'''
        # Assign indices to class: Order of "inverse length", then by "key ascending"
        for clusterRaw, dataSamples in H.sortDictLen_Rev(self.dict_ClusterRaw_DataSamples):
            self.dict_Cluster_Index[clusterRaw] = len(self.dict_Cluster_Index) + 1

        # Assign indices to labels: Order of "inverse length", then by "key ascending"
        for label, dataSamples in H.sortDictLen_Rev(self.dict_Label_DataSamples):
            self.dict_Label_Index[label] = len(self.dict_Label_Index) + 1

        self.dict_Index_Cluster = {v: k for k, v in self.dict_Cluster_Index.iteritems()}
        self.dict_Index_Label = {v: k for k, v in self.dict_Label_Index.iteritems()}
    
    def setDict_filterSize(self):
        '''Remove those clusters having small number of data samples (as per MIN_TRAIN_SIZE param)'''
        for clusterRaw in self.dict_ClusterRaw_DataSamples.keys():
            dataSamples = self.dict_ClusterRaw_DataSamples[clusterRaw]

            if len(dataSamples) < self.MIN_TRAIN_SIZE: # If small cluster
                # Remove it from clusterRaw
                self.dict_ClusterRaw_DataSamples.pop(clusterRaw) 

    def setDict_labels(self):
        # Add dataSamples labelwise to dict_Label_DataSamples
        for clusterRaw in self.dict_ClusterRaw_DataSamples:
            for d in self.dict_ClusterRaw_DataSamples[clusterRaw]:
                for label in d.labelList:
                    if label not in self.dict_Label_DataSamples: 
                        self.dict_Label_DataSamples[label] = []
                    self.dict_Label_DataSamples[label].append(d)

    def setDict_Cluster(self, fname, H_src=None, H_trgt=None):
        self.dict_ClusterRaw_DataSamples = {}
        if '.xlsx' in fname:
            df=pandas.read_excel(fname, converters={'clusterID': str, 'subClassID': str})
            headers, lines = df.columns.tolist(), df.values
        else:
            headers, lines = H.readCSV(fname)

        lines = lines
        hList = map(lambda x:x.lower(), headers)
        indexTrain, indexPredict = hList.index(self.trainSet.lower()), hList.index(self.predictSet.lower())
        i_errSet = hList.index('errset')

        for l in lines:
            trainRaw, predictRaw = l[indexTrain], l[indexPredict]
            errSet = l[i_errSet]
            src, trgt = None, None
            if H_src:  src  = l[headers.index(H_src)]  # If headers for source-target pairs
            if H_trgt: trgt = l[headers.index(H_trgt)] # Then, associate with the dataSample

            if predictRaw == predictRaw: # is not empty (i.e, shouldn't be a NaN, for pandas)
                d=DataSample(str(trainRaw), errSet, str(predictRaw), self.predictSet, src, trgt)

                # Append to self.dict_ClusterRaw_DataSamples
                if d.clusterRaw not in self.dict_ClusterRaw_DataSamples: 
                    self.dict_ClusterRaw_DataSamples[d.clusterRaw] = []
                self.dict_ClusterRaw_DataSamples[d.clusterRaw].append(d)

        # Once raw cluster/label dicts are created, 
        # filter out small clusters, assign labels and indices to rest
        self.setDict_filterSize()
        self.setDict_labels()
        self.setDict_Indices()

    def setNewTest(self, newTests, errSets, modelName):
        # Set cluster-ID to be a dummy -1
        self.X_test_DataSample = [DataSample(rawTest, errSet, '-1', self.predictSet) 
                                    for rawTest, errSet in zip(newTests, errSets)] 
        self.X_test_rawText = [i.rawText for i in self.X_test_DataSample] 

        print '--- Encoded input to classifier ---\n', self.X_test_rawText
        self.vectorizeIO_NewTest()
        self.setDefaultTrainSet(modelName)



def readDataset(trainSet, predictSet, multiClass=False, MIN_TRAIN_SIZE=5, H_src=None, H_trgt=None):
    print colored('\tReading DataSet: ...', 'magenta')    

    ds = DataSet(trainSet, predictSet, multiClass, MIN_TRAIN_SIZE)

    ds.setDict_Cluster(CF.fnameDataset, H_src, H_trgt)
    ds.shuffleClusters()
    ds.splitTrainTest()

    ds.vectorizeInput()
    ds.vectorizeOutput()
    return ds
