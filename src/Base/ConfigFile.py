import os, sys

# --------- Init --------- #
if  not ('__file__' in locals() or '__file__' in globals()):
    __file__='.'
full_path = os.path.realpath(__file__)
pathCurr, configFilename = os.path.split(full_path)

# --------- Data Paths --------- #
pathData =  pathCurr + "/../../data/"
pathInput = pathData + 'input/'
pathOutput = pathData + 'output/'
pathSavedModels = pathData + 'savedModels/'

fnameDataset = pathInput + 'dataset.csv'
fnameKeywords = pathInput + "keywords.txt"
fnameClusterErrs = pathOutput + 'clusterErrs.csv'

fnameConfMat = pathOutput + 'currRun_ConfMatrix.csv'

fnameTmpFile = pathInput + 'temp.c'

# --------- Clang library include (stdio, stddef, etc) PATH --------- #
# Ubuntu: /usr/lib/clang/xx.yy/include
# MacOS: '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/'
pathClangLib = "/usr/lib/clang/6.0/include" 

# --------- Abstraction ------------ #
fnameErrorIDs = pathInput + 'error_IDs.csv'
fnameIncludeIdent = pathInput + 'include_identifiers.csv'
IncludeIdentifiers = [line.strip() for line in open(fnameIncludeIdent).read().split('\n') 
                                    if line.strip()!=''] 

# --------- Data Preprocess Parameters --------- #
MY_SEED = 3
FLAG_ADD_EOL = True # Recommended to always set this flag to True (otherwise, a\nb will a treated as new Token)
ERRSET_PRE = 'ERROR_#'
ERRSET_SEP = ' </ERRSET> '

# --------- Neural Network Parameters --------- #
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 1.0 - TRAIN_SPLIT - VALIDATION_SPLIT
EMBEDDING_VECTOR_LENGTH = 32
