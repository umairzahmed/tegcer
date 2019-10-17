import re
from src.Base import ConfigFile as CF, Helper as H

class CompError():
    '''Compilation errors associated with a code (obtained from diagnostic module of Clang)'''
    def __init__(self, row):
        self.row = row
        self.ids = []
        self.tokens = []
        self.tokenSpellings = []
                
        self.id = 0
        self.compID = 0
        self.codeID = 0
        self.severity = -1
        self.contents = None
    
    def initDiagnostics(self, diag):
        self.line = diag.location.line
        self.pos = diag.location.column
        self.typeErr = diag.category_name
        self.msg = diag.spelling
        self.severity = diag.severity
        #self.findIDs()
        
    def __str__(self):
        return str(self.line) + "-" + str(self.pos) + " " + self.typeErr + ": " + self.msg 
    #+ "\ne-" + str(self.id) + "->cc-" + str(self.compID) + "->co-" + str(self.codeID) 
    
    def findIDs(self):
        patternStr = r'([^\']*?)\'(.*?)\'([^\']*?)'    
        matches = re.findall(patternStr, self.msg)
        for match in matches:
            self.ids.append(match[1]) # Pick the string between quotes (2nd group), for all matches
            
    
    def findTokens(self, tu):   
        for token in tu.cursor.get_tokens():
            if token.location.line == self.line:
                self.tokens.append(token)
                
        self.tokenSpellings = [t.spelling for t in self.tokens]
        
         
    def getTokenStr(self):
        return ' '.join(self.tokenSpellings)                

    def compareErr(self, ce):
        return self.line == ce.line and self.pos == ce.pos and self.msg == ce.msg 
    
    def compareErrLine(self, ce): # Don't check position'
        return self.line == ce.line and self.msg == ce.msg
    