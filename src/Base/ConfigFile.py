import os, sys

# Init

if  not ('__file__' in locals() or '__file__' in globals()):
    __file__='.'

full_path = os.path.realpath(__file__)
pathCurr, configFilename = os.path.split(full_path)

# Data Paths
pathData =  pathCurr + "/../../data/"
pathDataset = pathData + 'dataset.csv'

pathSavedModels = pathData + 'savedModels/'
fnameConfMat = pathData + 'currRun_ConfMatrix.csv'

# Clang library include (stdio, stddef, etc) PATH
# Ubuntu: /usr/lib/clang/xx.yy/include
# MacOS: '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/'
pathClangLib = "/usr/lib/clang/6.0/include" 

# Constants
MY_SEED = 3
FLAG_ADD_EOL = True
ERRSET_PRE = 'ERROR_#'
ERRSET_SEP = ' </ERRSET> '

# Parameters
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 1.0 - TRAIN_SPLIT - VALIDATION_SPLIT
EMBEDDING_VECTOR_LENGTH = 32
