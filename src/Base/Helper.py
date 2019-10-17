from itertools import tee
import csv, random, inspect, time
import numpy as np
from termcolor import colored
from ConfigFile import *
from collections import Counter

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def joinList(li, joinStr='\n', func=str):
    return joinStr.join([func(i) for i in li]) 

def joinLL(lists, joinStrWord=' ', joinStrLine='\n', func=str):
    listStrs = [joinList(li, joinStrWord, func) for li in lists]
    return joinList(listStrs, joinStrLine, func) 


def stringifyL(li):
    return [str(token) for token in li]

def stringifyLL(lists):
    return [stringifyL(li) for li in lists]

def readCSV(fname):
    f = open(fname, 'rU')
    freader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    lines = list(freader)
    f.close()
    headers = [i.strip() for i in lines[0]]

    return headers, lines[1:]

def writeCSV(fname, headers, lines):    
    fwriter = csv.writer(open(fname, 'w'), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
    fwriter.writerow(headers)
    fwriter.writerows(lines)

def appendCSV(fname, headers, lines):
    try:
        origH, origLines = readCSV(fname)
        if len(origLines) > 0:
            fwriter = csv.writer(open(fname, 'a'), 
                delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) 
            fwriter.writerows(lines)
            return
    except Exception:
        pass

    # If ran into 'no file found' error, or no lines present in the CSV, write it anew
    writeCSV(fname, headers, lines) 

def sortDictLen_Rev(dicti):
    '''Returns a sorted(dictionary) based on the length of its value list (desc), and key (asc)'''
    return sorted(dicti.iteritems(), key=lambda (key,val):(-len(val), key))

def sortDictVal(dicti, reverse=False):
    '''Returns a sorted(dictionary), based on its val'''
    return sorted(dicti.iteritems() , key=lambda (k,v): (v, k), reverse=reverse)

def getShuffleKeys(dict):
    keys = dict.keys()
    print 'Shuffling keys with seed=',MY_SEED
    random.seed(MY_SEED)
    random.shuffle(keys)
    return keys

def getRandom_K(li, k):
    random.seed(MY_SEED)
    return random.sample(li, k)

def getTop_K(li, k=None):
    counter = Counter(li)
    sortedD = sortDictVal(counter, reverse=True)
    sortedL = [i[0] for i in sortedD] # Return just the 'keys' and not the count

    if k == None:
        return sortedL
    return sortedL[:k]

def rfind(li, val, start, end):
    '''Find highest index of 'val' in list 'li'. Return -1 otherwise '''
    index = -1
    li = li[start:end]
    for i in range(len(li)):
        if li[i]==val: 
            index=i
    return index

def removeDuplicates(li):
    seen = {}
    uniqLi = []

    for item in li:
        if item not in seen:
            uniqLi.append(item) 
            seen[item] = 1
            
    return uniqLi

def errorLog(listMsgs):
    f=open(filenameErrorLog, 'a')
    f.write(time.asctime() +'\t') 
    f.write('FileName='+ inspect.stack()[2][1] +'\t')
    f.write('FuncName='+ inspect.stack()[2][3] +'\t')
    f.write(joinLL(listMsgs, '=', '\t') + '\n')
    f.close()


def printSBS_line(str1, str2):
    # Replace \t with 4 spaces, otherwise creates indentation issues
    str1, str2 = str1.replace('\t', '    '), str2.replace('\t', '    ')

    maxLine = 80
    maxLen = max([len(str1), len(str2)])
    numIter = maxLen / maxLine + 1
    separator = '='
    if str1 != str2: separator = '!'

    for index in range(numIter):
        minRange = index * maxLine; maxRange = minRange + maxLine
        formatStr = ('{:' + str(maxLine) +'}')
        line1 = formatStr.format(str1[minRange : maxRange])
        line2 = formatStr.format(str2[minRange : maxRange])

        print line1, colored(separator, 'red'), line2
        separator += '_'

def printSBS_list(li1, li2):
    '''Print 2 lists side by side'''
    maxLen = max([len(li1), len(li2)])
    for index in range(maxLen):
        str1 = ''.join(li1[index : index+1])
        str2 = ''.join(li2[index : index+1])
        printSBS_line(str1, str2)

    print colored('-'*100, 'blue')

def printSBS_str(str1, str2):
    printSBS_list(str1.splitlines(), str2.splitlines())


def checkEven(roll):
    try:
        return int(roll) % 2 == 0
    except Exception, e:
        return False

def checkOdd(roll):
    try:
        return int(roll) % 2 != 0
    except Exception, e:
        return False

class MaxTimeBreak:
    def __init__(self, maxTime):
        self.maxTime = maxTime
        self.startTime = time.time()
        self.endTime = self.startTime + self.maxTime

    def isTimeUp(self):
        currTime = time.time()
        if currTime >= self.endTime:
            return True
        return False

def avg(li):
    return round(np.average(li), 2)

def div(a, b):
    '''Return a/b (float div). Returns None if denominator is 0'''    
    if a == None: return None
    if b == 0: return None
    return float(a) / b

def fetchExists(dicti, key):
    if key in dicti:
        return dicti[key]
    return None

def fetchExists_list(dicti, listK):
    return [fetchExists(dicti, k) for k in listK]
    