import numpy as np
import pandas as pd

data = open('data\data.txt', 'r')
out = data.read()
data.close()
out0=out.split("\n\n")
out0=[l.split("\n") for l in out0]
out03=out0
j=0  
for a in out0:
    out03[j]=[]
    a=[x.split("\t") for x in a]
    for b in a:
        if (len(b)<2):
            a.remove(b)
        else:
            c=[float(y) for y in b]
            #c = [x - np.mean(b) for x in b]
            out03[j].append(c)
    j+=1
    
    
out04=out03

j=0
for a in out03:
    out04[j]=[]
    for line in a:
        zeros_num=0 
        dataline=[]
        for num in line:
            if (num==0):
                #print(line)
                zeros_num+=1
        av=sum(line)/(len(line)-zeros_num)
        for num in line:
            num1=num-av
            if (num==0):
                num1=num        
            dataline.append(num1)
        #dataline = line - np.average(np.array(line))
        #dataline = [num - np.average(line) if num!=0 elif 0 for num in line]
        out04[j].append(dataline)
    j+=1
        
#print(out04)
data=[]
for pack in out04:
    for line in pack:
        dataline = line #- np.mean(line)
        data.append(dataline)
X = np.corrcoef(data, rowvar=False)
X2 = pd.DataFrame(data).corr()
#print(X)
X1 = np.array([[1,0.685,0.613,0.733,0.766,0.607,0.657],
               [0.685,1,0.603,0.739,0.725,0.670,0.596],
               [0.613,0.603,1,0.683,0.835,0.764,0.706],
               [0.733,0.739,0.683,1,0.798,0.770,0.655],
               [0.766,0.725,0.835,0.798,1,0.807,0.731],
               [0.607,0.670,0.764,0.770,0.807,1,0.746],
               [0.657,0.596,0.706,0.655,0.731,0.746,1]])
