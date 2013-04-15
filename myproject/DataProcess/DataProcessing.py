'''
Created on Nov 17, 2012

@author: aurora
'''
import re

if __name__ == '__main__':
    f = open("spambase_1\spambase.data",'r')
    
    
    lines = f.readlines()
    
    newLines = []

    for s in lines:

        a = re.split(',', s)
#        print a
        
        lineP = []
        lineP.append(a[len(a)-1].strip('\n'))
        del a[len(a)-1]
        
        for n in range(len(a)):
            if a[n] != '0':
                stringItem = str(n) + ':' + str(a[n])
                lineP.append(stringItem)
        
        aline = ' '.join(lineP)
        print aline
        
        aline = aline+'\n'
        newLines.append(aline)
    
    f.close();
    
    '\n'.join(newLines);
    newf = open("spambase_1\spambase.data.new",'w')
    newf.writelines(newLines);
    newf.close()
    
    