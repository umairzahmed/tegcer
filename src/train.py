'''Train TEGCER on a given dataset, and report the precision/recall scores.'''

import os, sys, datetime, csv
import cPickle as pickle

from src.Base import ConfigFile as CF, Helper as H
from Classifier import DataEncoding, DeepModels, Accuracy

# ------------ BEGIN PARAMETERS -------------

    # In the dataset.csv, which column is to be used for training model
trainSet = 'sourceLineAbs'  # sourceText | targetText | sourceAbs | targetAbs | sourceLineText | targetLineText | sourceLineAbs |   targetLineAbs
    # In the dataset.csv, which column refers to the class/label
predictSet = 'errSet_diffs' 
    # In the dataset.csv, the columns which refer to the source-target example pairs. These pairs are used during "live-deployment" phase (on invoking run.py), to suggest feedback after predicting corresponding class/label.
H_src, H_trgt = 'sourceLineText', 'targetLineText' 

multiClass = False # Treat the problem as multi-label? By splitting the predictSet on ; to predict a multi-class?
modelName = 'Dense_Bin' # Dense_Bin | embed_CNN_LSTM_Dropout_Recur | LSTM | embed_LSTM | CNN_LSTM_Dropout 

TRAIN_MULT_FACTOR = 1 # Multiply the training dataset by this factor (since the dataset is small)
MIN_TRAIN_SIZE = 10 # Ignore those classes having less than these many samples (before MULT_FACTOR)
EPOCHS = 6

# ------------ END PARAMETERS ---------------

fname_summary = CF.pathOutput + 'deepClassify_summary.csv'

def writeSummary(row):
    headers = ['time_Recorded', 'train_set', 'predict_set', 'totalNumPairs', 'Train,Valid,Test', 'numClasses', 'modelName', 'modelSummary', 'Pred@1,3,5', 'precision', 'recall', 'trainTime', 'max_seq_length', 'max_vocab_size', 'TRAIN_MULT_FACTOR', 'EPOCHS', 'EMBEDDING_VECTOR_LENGTH', 'classMapping_RawCluster', 'confusion_matrix', 'acc', 'loss', 'val_acc', 'val_loss']
    H.appendCSV(fname_summary, headers, [row])

def trainTest(modelName, dataset):
    # Train and Test
    deepModel = DeepModels.DeepModel(modelName, dataset)
    history, trainTime = deepModel.train(EPOCHS, TRAIN_MULT_FACTOR)
    acc = deepModel.test()

    # Save the model and weights
    open(CF.pathSavedModels + modelName + ".json", "w").write(deepModel.model.to_json()) # Store model
    deepModel.model.save_weights(CF.pathSavedModels + modelName + ".h5") # serialize weights to HDF5
    pickle.dump(dataset.tokenizer, open(CF.pathSavedModels + modelName + '_xNorm.pickle', "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(dataset.mlb, open(CF.pathSavedModels + modelName + '_yNorm.pickle', "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # Plots
    confMatrix = Accuracy.ConfusionMatrix(deepModel)
    strConfMat, predAtK = confMatrix.calcConfMat()
    strConfMat = ''

    prec, recall = confMatrix.getPrecRecall()
    return str(deepModel), history, trainTime, prec, recall, strConfMat, predAtK

def roundH(h, name):
    return [round(i,2) for i in h.history[name]]

def recordAccModel(modelName, dataset):
    strDeepModel, h, trainTime, prec, recall, strConfMat, predAtK = trainTest(modelName, dataset)

    currTime = datetime.datetime.now().ctime()
    numPairs = dataset.getTotalNumPairs()
    tvt = (round(CF.TRAIN_SPLIT,2), round(CF.VALIDATION_SPLIT,2), 
            round(1-CF.TRAIN_SPLIT-CF.VALIDATION_SPLIT,2) )

    dict_index, num_classes = dataset.dict_Cluster_Index, dataset.num_classes
    if dataset.multiClass:
        dict_index, num_classes = dataset.dict_Label_Index, dataset.num_labels
    classMapStr = '\n'.join([str(j)+' -> '+str(i) for i,j in H.sortDictVal(dict_index)])

    row = [currTime, trainSet, predictSet, numPairs, tvt, num_classes-1]
    row += [modelName, strDeepModel, predAtK, prec, recall, trainTime, dataset.max_seq_length, dataset.max_vocab_size]
    row += [TRAIN_MULT_FACTOR, EPOCHS, CF.EMBEDDING_VECTOR_LENGTH]
    row += [classMapStr, strConfMat, roundH(h, 'acc'), roundH(h, 'loss'), roundH(h, 'val_acc'), roundH(h, 'val_loss')]

    writeSummary(row)


if __name__=='__main__':
    dataset = DataEncoding.readDataset(trainSet, predictSet, multiClass, 
                            MIN_TRAIN_SIZE=MIN_TRAIN_SIZE, H_src=H_src, H_trgt=H_trgt)
    
    recordAccModel(modelName, dataset)
