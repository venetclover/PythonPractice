'''
Created on Nov 17, 2012

@author: aurora
'''
import re

if __name__ == '__main__':
    
    relation = "@RELATION myData"
    attr = []
    for n in range(1,56):
        attr.append("@ATTRIBUTE F" + str(n) + " REAL")
    
    classAttr = "@ATTRIBUTE class    {spam,notspam}"
    dataHead = "@DATA"
    
    classAttrStr = ['spam', 'notspam']
    
    newLines = []
    newLines.append(relation + '\n')
    newLines.append('\n'.join(attr) +'\n')
    newLines.append(classAttr +'\n')
    newLines.append(dataHead + '\n')
    
    f = open("spambase_1/spambase.data",'r')
    lines = f.readlines();
    
    for s in lines:
        a = re.split(',',s)
        if a[len(a)-1] == '0\n' :
            a[len(a)-1] = 'notspam\n'
        elif a[len(a)-1] == '1\n' :
            a[len(a)-1] = 'spam\n'
            
        newLine = ",".join(a)
        
        newLines.append(newLine)
        
        
    f.close()
    
    newf = open("spambase_1/spambase.data.weka",'w')
    newf.writelines(newLines)
    newf.close()