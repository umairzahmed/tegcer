'''Generate example based hints on a given program that fails to compile, by running a previously trained model.
Usage: python -m src.run ./path_to/buggy_code.c'''
import keras
import math, sys

from src import train
from Classifier import DataEncoding, DeepModels
from src.Base import ConfigFile as CF, Helper as H
from src.CodeAbstraction import Code, Abstraction, ErrorGroup

# ---- Params: Begin

# Note: Use the same encoding as Training phase. 
    # to re-define the encoding/decoding techniques and parameters, used during training phase.
    # such as max_vocab_size, max_seq_length, num_classes, etc.

trainSet = train.trainSet # 'sourceLineAbs'
predictSet = train.predictSet # 'errSet_diffs'
H_src, H_trgt = train.H_src, train.H_trgt # 'sourceLineText', 'targetLineText'
modelName = train.modelName # 'Dense_Bin' 
MIN_TRAIN_SIZE = train.MIN_TRAIN_SIZE

# Predict TOP_K*TOP_EG number of examples per erroneous line.
TOP_K = 2 # Predict top-K #classes per erroneous line
TOP_EG = 2 # Fetch top-EG #examples per class. 

# ---- Params: End

# --- Read model and setup dataSets/deepModels
def readModel():
    '''Load model and weights'''
    json_file = open(CF.pathSavedModels + modelName + '.json', 'r')
    loaded_model = keras.models.model_from_json(json_file.read())
    json_file.close()
    
    # load weights into new model
    loaded_model.load_weights(CF.pathSavedModels + modelName + '.h5')

    return loaded_model

def setup():
    '''Load model and dataset pre-process/encoding methods'''
    # Read the stored model
    model = readModel()
    
    dataset = DataEncoding.readDataset(trainSet, predictSet, 
        MIN_TRAIN_SIZE=MIN_TRAIN_SIZE, H_src=H_src, H_trgt=H_trgt)
    
    deepModel = DeepModels.DeepModel(modelName, dataset)
    #history, trainTime = deepModel.train(EPOCHS, TRAIN_MULT_FACTOR)

    deepModel.model = model
    return deepModel, dataset

# --- Post-Processing: After running the deepModel ---

def generateExamples(dataset, predClasses):
    '''Given the entire dataset and a predicted label, print TOP_EG number of example pairs.'''
    # Init
    labels = [dataset.dict_Index_Cluster[i] for i in predClasses]
    prettyLabels = [i.replace('\n', ' ') for i in labels]
    examplesList = []
    
    print 'Top-{} Predicted Class-IDs:\n{}'.format(TOP_K, predClasses)
    print 'TOP-{} Predicted Class-Labels (Error-IDs and Unique Repairs):\n{}\nExamples:'.format(TOP_K, prettyLabels)

    # For each label
    numC = 0
    for label in labels:
        dataSamples = dataset.dict_ClusterRaw_DataSamples[label] # Fetch all training examples
        srcTrgtPairs = [(H.joinList(i.src, ''), H.joinList(i.trgt, '')) for i in dataSamples] # Create list of (before, after)
        
        topEgs = H.getTop_K(srcTrgtPairs, TOP_EG) # Sort them based on frequency and fetch Top-EG #examples

        for i in range(len(topEgs)): # For each example
            eg = topEgs[i]
            before, after = eg[0].replace('\n', ''), eg[1].replace('\n', '') # Remove new-line characters
            if before.strip() == '': before = '// Empty Line' # Deal with empty lines
            if after.strip() == '': after = '// Empty Line' # Deal with empty lines

            print 'Eg #{} class-{} before: {}'.format(i+1, predClasses[numC], before) # Print them
            print 'Eg #{} class-{} after : {}'.format(i+1, predClasses[numC], after)
            
            examplesList.append(eg)
        numC += 1

    # Extend to the examplesList, proportionately    
    return examplesList

# --- Running the Deep Model ---

def runDeepModel(deepModel, dataset, allErrs, codeText, errClang, maxNumEg=3):
    line_eg = {}
    absLines, lineNums = Abstraction.getBuggyAbsLine(codeText) # Fetch abstraction of buggy lines
    errSet_perLine = ErrorGroup.getErrSetLines(allErrs, errClang, lineNums) # Fetch Error-Groups for each buggy line

    print 'Erroneous lineNums:\n{}\nErroneous abstracted lines:\n{}'.format(lineNums, absLines)
    print 'Error-Groups per line-number:\n{}\n\n'.format(errSet_perLine)

    if len(absLines) != 0: # Only if there are errors (with lineNums) 
        dataset.setNewTest(absLines, errSet_perLine, modelName) # Set new "test-data", to invoke model on
        preds_topk = deepModel.getPrediction(PREDICT_TOP_K=TOP_K) # Get top-K class predictions by model
        
        for lineNum, predClasses in zip(lineNums, preds_topk): # For each buggy line and predicted class
            print '\nLineNum:\n{}\nActual Buggy-Line:\n{}'.format(lineNum, codeText.splitlines()[lineNum-1])
            examplesList = generateExamples(dataset, predClasses) # Pretty print example-based feedback
            line_eg[lineNum] = examplesList

    return line_eg

if __name__=='__main__':
    # Setup the model and dataset environment/params
    deepModel, dataset = setup()    

    # Fetch buggy C filename
    try:
        fname = sys.argv[1]
    except IndexError:
        sys.exit('Usage: python -m src.run ./path_to/buggy_code.c')

    # Read code-text and its compilation errors by Clang
    codeText = open(fname).read()
    errClang = ErrorGroup.fetchClangError(codeText)
    allErrs = ErrorGroup.readAllErrors()

    print 'Given buggy code:\n', codeText
    print '\nClang output on the given code ...\n', errClang
    
    # Run deep model and Generate the example based feedback
    runDeepModel(deepModel, dataset, allErrs, codeText, errClang, maxNumEg=20)
