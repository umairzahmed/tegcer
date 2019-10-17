import sys, os, csv, re, operator, collections, difflib
from subprocess import call

from src.Base import ConfigFile as CF, Helper as H

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

class Error:
    def __init__(self, errorExp, tempIndex=None, index=None):
        self.errorExp = errorExp # Explanation
        self.tempIndex = tempIndex # If the sorting of indices (based on freq) isn't done yet, assign a temporary Index
        self.index = index # Otherwise, assign the permanent index
        self.count = 0

    def setIndex(self, index):
        self.index = index
        self.tempIndex = None

    def getIndex(self):
        if self.index==None:
            return self.tempIndex
        else: return self.index

class ErrSet:
    def __init__(self, keyList):
        self.keySet = set(keyList)
        self.key = '; '.join(map(str, self.keySet )) + ';'
        self.minKey = {} # {err1:minCount1, err2: minCount2, ...} - in an errSet, how many times an error was repeated
        self.maxKey = {} # {err1:maxCount1, err2: maxCount2, ...} 
        
        self.diffs = {}
        self.totalNum = 0
        self.numInter = 0 # size of total intersection set (over all examples, how many of compilation lines matched with edited lines)
        self.numEditLines = 0 # total number (over all examples) of unique edited lines, for this comp err set
        self.numCompLines = 0 # total number (over all examples) of unique compilation lines reported, for this err set
    
    def __str__(self):
        s = ''
        for k in self.keySet:
            s += str(k) 
            if self.maxKey[k] > 1: # Only if max count of key is > 1 (that is, this error occurs more than once in any given example)
                s += '{' + str(self.minKey[k]) + ',' + str(self.maxKey[k]) + '}'
            s += '; '

        return s

    def calcRange(self, keyList):
        '''Given a list of compiler errors, calculate update the min and max times each comp error occurs'''
        for k in self.keySet:
            if k not in self.minKey: self.minKey[k] = 1
            if k not in self.maxKey: self.maxKey[k] = keyList.count(k)

            self.minKey[k] = min(self.minKey[k], keyList.count(k))
            self.maxKey[k] = max(self.maxKey[k], keyList.count(k))

    def calcIntersection(self, linesEdit, linesCompErr):
        self.numEditLines += len(linesEdit)
        self.numCompLines += len(linesCompErr)
        self.numInter += len(linesEdit.intersection(linesCompErr))

    def calcPrecRecall(self):
        self.prec, self.recall = None, None

        if self.numCompLines > 0:
            self.prec = round(float(self.numInter) / self.numCompLines, 2)
        
        if self.numEditLines > 0:
            self.recall = round(float(self.numInter) / self.numEditLines, 2)

        return self.prec, self.recall    

def createErrSet(keyList, dictErrDiff):
    errSet = ErrSet(keyList)
    if errSet.key not in dictErrDiff:
        dictErrDiff[errSet.key] = errSet
    errSet = dictErrDiff[errSet.key] # Fetch the errSet from dict (to update its count)
    errSet.totalNum += 1
    errSet.calcRange(keyList)

    return errSet

def getErrSet(allErrs, dictErrDiff, errPrutor, atLine=None):    
    errList = re.findall('(?:[a-z0-9\_\/]+\.c\:)?(\d+)\:(?:\d+\:([a-z ]+)\:)?(.*)', errPrutor)
    # ?: to ignore grouping. First one to make sure either beginning of error-string, or beg of line in error-string
    # Last ? to make the "columnNumber:type" as optional (to match linker errors)
    keyList, lineNums, errExplanationList = [], set(), []

    for lineNum, errType, errExplanation in errList:
        if atLine != None and str(lineNum) != str(atLine): continue

        lineNum, errType, errExplanation = lineNum.strip(), errType.strip(), errExplanation.strip()
        errExplanation = re.sub('(?:\'.*\')|(?:\".*\")|(?:\`.*\')','ID',errExplanation) # Abstract 'xxx', "xxx", `xxx' with ID
        errExplanation = re.sub('\d+','ID', errExplanation) # Abstract numbers with ID
        errExplanation = re.sub('\[.*\]','', errExplanation) # Remove [xxx]
        errExplanation = re.sub('[\+\=\\\/\>\<\!\&\^\*]+','OP', errExplanation) # Abstract special chars (+, =, etc) with OP

        if errType == 'note' or errType=='warning': # Ignore warnings and notes 
            continue # (consider only 'fatal error', 'error' and '' => linker errs)
            
        if errExplanation == '':
            a=0

        if errExplanation not in allErrs:
            allErrs[errExplanation] = Error(errExplanation, tempIndex=len(allErrs)+1)
        allErrs[errExplanation].count += 1

        errExplanationList.append(errExplanation)
        keyList.append(allErrs[errExplanation].getIndex())
        lineNums.add(lineNum)
    
    errSet = createErrSet(keyList, dictErrDiff)
    return errSet, errExplanationList, lineNums

def clusterErr(errSet, diffsI, diffsD):
    diffsI = ['+ '+i.strip() for i in diffsI if i.strip()!='']
    diffsD = ['- '+i.strip() for i in diffsD if i.strip()!='']
    # diffsR = ['* '+i.strip() for i in diffsR if i.strip()!='']

    for diff in diffsI + diffsD: # + diffsR
        if diff not in errSet.diffs:
            errSet.diffs[diff] = 0
        errSet.diffs[diff] += 1

def writeAllErrs(allErrs):
    f=open(CF.fnameErrorIDs, 'w')
    fcsv = csv.writer(f)
    f.write('index,error_message,count\n')
    for errExp, error in sorted(allErrs.items(), key=lambda item:item[1].getIndex()):        
        fcsv.writerow([error.getIndex(), error.errorExp, error.count])
    f.close()

def writeClusterErr(dictErrDiff):
    f=open(CF.fnameClusterErrs, 'w')
    fcsv = csv.writer(f)
    f.write('error,totalNum_srcTarget,avgEdits,precision_compiler,recall_compiler,#Count Diff\n')

    totalEdits, totalCount = 0, 0.0
    uniqueEdits = collections.defaultdict(lambda: 0)

    for e in dictErrDiff:
        errSet = dictErrDiff[e]
        prec, recall = errSet.calcPrecRecall()
        #print e, errSet.totalNum, errSet.diffs,'\n' 
        numEdits = reduce(lambda x,y:x+y, errSet.diffs.values(), 0)
        avgEdits = round(float(numEdits)/errSet.totalNum, 2)

        row = [errSet, errSet.totalNum, avgEdits,  prec, recall]
        for diff, numErr in sorted(errSet.diffs.items(), key=operator.itemgetter(1), reverse=True):
            sign, edit = diff[0], diff[1:].strip('"') # diff[0] has '+' or '-', depending on ins/del resp
            row.append(sign + str(numErr) +' '+ edit) 
            uniqueEdits[edit] += 1

        fcsv.writerow([unicode(s).encode("utf-8") for s in row])

        totalCount += errSet.totalNum
        totalEdits += numEdits
    f.close()

    print '#avgEdits = ', round(float(totalEdits)/totalCount, 2)
    print '#uniqueEdits = ', len(uniqueEdits)

    #print uniqueEdits

def readPrev_AllErrors():
    '''Check if indexing of errors (sorted based on count) is already present in the path.
    Based on some previous run (or semester). If so, use that indexing (most freq comp error gets index-1)'''
    allErrs = {}
    try:
        headers, lines = H.readCSV(CF.fnameErrorIDs)
        indexIndex, indexErrExp = headers.index('index'), headers.index('error_message')

        for line in lines:
            index, errExp = line[indexIndex], line[indexErrExp]
            allErrs[errExp] = Error(errExp, index=index)

    except IOError:
        pass

    return allErrs

def writeErrSets(fname):
    headers, lines = H.readCSV(fname)
    headers.append("ErrSet")
    dictErrDiff = {} # {CompErr1:ErrSet1, ...}
    allErrs = readPrev_AllErrors()
    count = 0
    print 'Total #src-target pairs=',len(lines)

    indexErrPrutor = headers.index("sourceErrorPrutor")
    indexErrClang = headers.index("sourceErrorClangParse")
    indexLineNums = headers.index("lineNums_Abs")
    indexDi, indexDd = headers.index("diffAbs_ins"), headers.index("diffAbs_del")

    for line in lines:        
        count += 1
        if count%1000==0:
            print count,'/',len(lines),'done ...'

        diffsI, diffsD = line[indexDi].splitlines(), line[indexDd].splitlines()
        errPrutor, errClang, diffLineNums = line[indexErrPrutor], line[indexErrClang], set(line[indexLineNums].splitlines())
        errPrutor, errClang = errPrutor.replace('\r', '\n'), errClang.replace('\r', '\n')

        errSet, errExpList, compLineNums = getErrSet(allErrs, dictErrDiff, errPrutor) # Get the err-set (unique rep for set of errors)
        clusterErr(errSet, diffsI, diffsD) # Cluster the diffs (add the diff to dictErrDiff)
        errSet.calcIntersection(compLineNums, diffLineNums) # Update counts to calc precision-recall of compiler lineNums

        line.append(errSet.key)
        
    H.writeCSV(fname, headers, lines)
    writeAllErrs(allErrs)
    writeClusterErr(dictErrDiff)


if __name__ == "__main__":
    writeErrSets(CF.fnameDataset)    
    