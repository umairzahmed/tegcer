import base64, clang.cindex, collections, traceback
from subprocess import Popen, PIPE, STDOUT

from clang.cindex import Config
from clang.cindex import Diagnostic

from src.Base import ConfigFile as CF, Helper as H
import CompError

ClangArgs = ['-static', '-Wall', '-funsigned-char', '-Wno-unused-result', '-O', '-Wextra', '-std=c99', "-I/"+CF.pathClangLib]

class Code:
    '''Given source-code (plain text), maintains the Clang parse and the compilation-errors associated with it.'''
    index = clang.cindex.Index.create()
        
    def __init__(self, codeText, row=None, indent=False, codeID=None):
        self.row = row
        self.tu = None
        self.ces = []
        self.notIndented = indent # By default False =>  then it won't trigger self.codeIndent(). If indent=True, then indent on calling clangParse
                
        if row == None:
            self.id = 0
            self.assignID = 0
            self.codeID = codeID
            self.compiled = 0
            self.codeText = codeText
            #self.raw_output = ""
            #self.compile_time = ""
            #self.contents = ""
        else:
            self.id = row[0]
            self.assignID = row[1]
            self.codeID = row[2]
            self.compiled = row[3]
            #self.raw_output = row[4]
            #self.compile_time = row[5]
            #self.contents = row[6]
            self.codeText = base64.b64decode(row[6])
        #if autoAddCEs: self.addCEsTU()
    
    def codeIndent(self):
        '''Pretty format the student's codeText (multi statements into multi lines)'''
        p = Popen(['indent', '-linux'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        self.codeText,err = p.communicate(input=self.codeText.replace('\r','\n'))
        
        if err!='' and err!=None:
            pass 
            #print err
    
    def clangParse(self):
        if self.notIndented: 
            self.codeIndent()
            self.notIndented = False

        try:
            filename = str(self.codeID) + '.c'
            self.tu = Code.index.parse(filename, 
            args=ClangArgs, unsaved_files=[(filename, self.codeText)])
        except Exception,e:             
            traceback.print_exc()
            print self.codeText
    
    def getTU(self):
        if self.tu == None:
            self.clangParse()
            self.addCEsTU(self.row)
            
        return self.tu
    
    def delTU_CEs(self):
        self.tu = None
        self.ces = [] 


    def getTokens(self):
        tu = self.getTU()
        if tu != None:
            return list(self.getTU().cursor.get_tokens())
        else: return None

    def getTokenLines(self):
        allTokens = self.getTokens()
        lineNum, index = 1, 0
        tempLine, tokenLines = [], []

        while index < len(allTokens):
            token = allTokens[index]
            if token.location.line == lineNum:
                tempLine.append(token)
                index += 1 
            else:
                tokenLines.append(tempLine)
                tempLine = []
                lineNum += 1

        if len(tempLine)!=0: # Add the leftover tokens
            tokenLines.append(tempLine)

        return tokenLines

    def getTokenSpellLines(self):
        tokenLines = self.getTokenLines()
        return [[token.spelling for token in line] for line in tokenLines]

    def getTokensAtLine(self, lineNum):
        fileTokens = self.getTokens()
        if fileTokens != None:
            return [t for t in fileTokens if t.location.line==lineNum]
        return None

    def getCEs(self):
        if len(self.ces) == 0:
            tu = self.getTU()
            if tu == None: return None
        
        return self.ces

    def getCEsAtLine(self, lineNum):
        if self.getCEs() != None:
            return [ce for ce in self.ces if ce.line==lineNum]
        return None

    def addCEsTU(self, row):
        if len(self.ces) == 0 and self.tu != None:
            for diag in self.tu.diagnostics:
                ce = CompError.CompError(row)
                ce.initDiagnostics(diag)
                ce.findTokens(self.tu)
                
                self.ces.append(ce)
                #print ce
    
    def getSevereErrors(self):
        return [ ce for ce in self.getCEs()
                    if ce.severity == Diagnostic.Error or ce.severity == Diagnostic.Fatal
               ]
    
    def getWarnings(self):
        return [ce for ce in self.getCEs() if ce.severity == Diagnostic.Warning]

    def getNumErrors(self):
        return len(self.getSevereErrors())

    def getNumWarnings(self):
        #print [ce.msg for ce in self.getSevereErrors()]
        #print [ce.msg for ce in self.getWarnings()]
        #print '\n'

        return len(self.getWarnings())

    def cesToString(self):
        return '\n'.join([str(ce) for ce in self.getCEs()])            
        
    def checkErrLineExists(self, givenCE):
        '''Check if the givenCE exists in this particular, at the level of line (ignore pos, just check for line and msg equivalence)'''
        for ce in self.getCEs():
            if ce.compareErrLine(givenCE):
                return True
        return False
    
