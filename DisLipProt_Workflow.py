#!/home/pyenv/versions/py3.7/bin/python

import os

#os.chdir("/home/biomine/webserver")
#path="/usr/lib/x86_64-linux-gnu/libjemalloc.so.1"
#command="export LD_PRELOAD="+path
#os.system(command)
#os.chdir("/home/biomine/webserver/biomine/DisLipProt")

#path="/usr/lib/x86_64-linux-gnu/libjemalloc.so.1"
#command="export LD_PRELOAD="+path
#os.system("export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.1")


#os.system("export PATH=/usr/lib/x86_64-linux-gnu${PATH:+:${PATH}}")
#os.system("export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}")



import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import io
from base64 import b64encode
import csv
import sys
from tensorflow import keras
from keras.models import Sequential
import tensorflow as tf
from keras import metrics
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten,Reshape,Dense, LSTM, RepeatVector, TimeDistributed,BatchNormalization,Embedding,Conv1D, GlobalAveragePooling1D, MaxPooling1D,LSTM,TimeDistributed,Bidirectional




# In[ ]:

inFile1=sys.argv[1]
result_dir = os.path.dirname(inFile1)
Lines= []
f = open(inFile1,'r')
for line in f:
    data_line = line.rstrip().split('\t')
    Lines.append(data_line)


# In[2]:


ID=[]
Sequence=[]

J = len(Lines)-1
b =0
for b in range(0,J,2):
    ID.append(Lines[b][0])
    Sequence.append(list(Lines[b+1][0]))


# In[3]:

print('Read')

#breaking down the big file into single sequence files
from itertools import zip_longest

def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

n = 2

with open(inFile1,'r') as f:
    for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
        with open(result_dir+'/Single_{0}.fasta'.format(i), 'w') as fout:
            fout.writelines(g)


Original_Indexes=[]
b=0
for b in range(0,len(ID),1):
        Original_Indexes.append(ID[b][1:])

os.chdir(result_dir)
b=0
for b in range(1,len(Sequence)+1,1):
         filename = result_dir+"/Single_"+str(b)+".fasta"
         command="/home/biomine/programs/ASAquick/bin/ASAquick "+filename
         os.system(command)


#os.chdir("/home/biomine/webserver/biomine/DisLipProt")


outfile = open(result_dir+'/ASA.txt','w')
for b in range(1,len(Sequence)+1,1):
    fileaddress =  result_dir+"/asaq.Single_"+str(b)+".fasta/rasaq.pred"
    f = open(fileaddress)
    res_list = list()
    line = f.readline()
    while line:
        line = line.rstrip('\n')
        line_cols = line.split()
        res_list.append(str(round((float(line_cols[2])/2)+0.5,2)))
        line = f.readline()
    f.close()
    #res_list = res_list[:-1]
    entry_res_string = ",".join(res_list)
    outfile.writelines(entry_res_string+'\n')
outfile.close()


# In[82]:


os.chdir(result_dir)
for b in range(1,len(Sequence)+1,1):
    os.system("runpsipred_single "+result_dir+"/Single_"+str(b)+".fasta")

#os.chdir("/home/biomine/webserver/biomine/DisLipProt")
# In[93]:


#Merging all secondary structure predicion files to one
with open(result_dir+'/Secondary.txt','w') as outfile:
    b=0
    for b in range(1,len(Sequence)+1,1):
        with open(result_dir+'/Single_'+str(b)+'.ss2','r') as infile:
            for line in infile:
                outfile.write(line)


Lines= []
f = open(result_dir+'/ASA.txt','r')
for line in f:
    data_line = line.rstrip().split('\t')
    Lines.append(data_line)

ASA=[]
for b in range(0,len(Lines),1):
     ASA.append(Lines[b][0].split(','))


# In[6]:


Psi = pd.read_csv(result_dir+"/Secondary.txt", comment="#", delim_whitespace= True, names =['ResidueNo', 'AA', 'SS', 'Coils', 'Helix', 'Strands'], dtype={'ResidueNo':int, 'AA':str, 'SS':str, 'Coils':str, 'Helix':str, 'Strands':str }, skiprows=1, header=None)
Helix_Score = Psi.loc[:,('Helix')].values
Coil_Score = Psi.loc[:,('Coils')].values
Strand_Score = Psi.loc[:,('Strands')].values


# In[7]:


len(ASA)


# In[8]:
#os.system("nohup nice SPOT-disorder.py "+inFile1+" "+result_dir+"/SpotResult.txt")

Spot = pd.read_csv(result_dir+"/SpotResult.txt", 
                   comment=">", 
                   delim_whitespace= True, 
                   names =['ResidueNo', 'AA', 'Prediction_Score', 'Binary_Prediction'], 
                   dtype={'ResidueNo':str, 'AA':str, 'Prediction_Score': str, 'Binary_Prediction':str}, 
                   skiprows=1, 
                   header=None)
Spot.head()

Spot_Score = Spot.loc[:,('Prediction_Score')].values





# In[13]:


EspD_PredictionScore=Spot_Score


# In[14]:
b=0
for b in range(1,len(Sequence)+1,1):
         filename = result_dir+'/Single_'+str(b)+'.fasta'
         command='perl /home/biomine/programs/ANCHOR/anchor.pl '+filename+' > '+filename+'.anchorout'
         os.system(command)

Lines= []
b=0
for b in range(1,len(Sequence)+1,1):
              f = open(result_dir+'/Single_'+str(b)+'.fasta.anchorout','r')
              for line in f:
                  data_line = line.rstrip()
                  Lines.append(data_line)
				  
with open(result_dir+'/ANCHORResult.txt', 'w') as f:
    for item in Lines:
        f.write("%s\n" % item)
		
ANCHOR = pd.read_csv(result_dir+"/ANCHORResult.txt", 
                   comment="#", 
                   delim_whitespace= True, 
                   names =['ResidueNo', 'AA', 'IUPRED2','ANCHOR'], 
                   dtype={'ResidueNo':int, 'AA':str, 'IUPRED2': np.float64, 'ANCHOR': np.float64}, 
                   skiprows=0, 
                   header=None)
ANCHOR.head()


# In[15]:


ANCHOR_PredictionScore = ANCHOR["ANCHOR"].values


# In[16]:


len(ANCHOR_PredictionScore)


# In[17]:

os.chdir("/home/biomine/webserver/biomine/DisoRDPbind")
os.system("./DisoRDPbind "+inFile1)
os.chdir("/home/biomine/webserver/biomine/DisLipProt")
diordp_out=result_dir+'/seqs.fasta.pred'
Lines= []
f = open(diordp_out,'r')
for line in f:
    data_line = line.rstrip().split('\t')
    Lines.append(data_line)


# In[18]:


DiordProt_Score_byProteins =[]
DiordDNA_Score_byProteins =[]
DiordRNA_Score_byProteins =[]
J = len(Lines)-7
b =0
for b in range(0,J,8):
               rna=Lines[b+3][0].strip().split(':')
               DiordRNA_Score_byProteins.append((rna[1]).strip().split(','))
               dna=Lines[b+5][0].strip().split(':')
               DiordDNA_Score_byProteins.append((dna[1]).strip().split(','))
               prot=Lines[b+7][0].strip().split(':')
               DiordProt_Score_byProteins.append((prot[1]).strip().split(','))


# In[19]:


b=0
for b in range(0,len(DiordProt_Score_byProteins),1):
                        unit= len(DiordProt_Score_byProteins[b])-1
                        DiordProt_Score_byProteins[b].pop(unit)
                        DiordDNA_Score_byProteins[b].pop(unit)
                        DiordRNA_Score_byProteins[b].pop(unit)


# In[20]:

os.chdir("/home/biomine/webserver/biomine/DFLpred")

os.system("java -jar DFLpred.jar "+inFile1+" "+result_dir+"/DFL.txt")
os.chdir("/home/biomine/webserver/biomine/DisLipProt")
Lines= []
f = open(result_dir+'/DFL.txt','r')
for line in f:
    data_line = line.rstrip().split('\t')
    Lines.append(data_line)


# In[21]:


DFL_Score_byProteins =[]
J = len(Lines)-2
b =0
for b in range(0,J,3):
               DFL_Score_byProteins.append(Lines[b+2][0].rstrip().split(','))


# In[22]:


#Getting length of each protein to a list
Length_EachProtein = []
b=0
for b in range(0,len(Sequence),1):
                   c = len(Sequence[b])
                   Length_EachProtein.append(c)

#Getting continous sum off All proteins to an array
d=np.cumsum(Length_EachProtein)


# In[23]:


EspD_PredictionScore_byProteins= np.split(EspD_PredictionScore,d)
ANCHOR_PredictionScore_byProteins= np.split(ANCHOR_PredictionScore,d)
Helix_Score_byProteins= np.split(Helix_Score,d) 
Coil_Score_byProteins= np.split(Coil_Score,d)
Strand_Score_byProteins= np.split(Strand_Score,d)


# In[24]:


EspD_PredictionScore=np.concatenate(EspD_PredictionScore_byProteins)
Helix_Score=np.concatenate(Helix_Score_byProteins)
Coil_Score=np.concatenate(Coil_Score_byProteins)
Strand_Score=np.concatenate(Strand_Score_byProteins)
AA =np.concatenate(Sequence)



DiordProt_Score=np.concatenate(DiordProt_Score_byProteins)
DiordDNA_Score=np.concatenate(DiordDNA_Score_byProteins)
DiordRNA_Score=np.concatenate(DiordRNA_Score_byProteins)
DFL_Score=np.concatenate(DFL_Score_byProteins)
ANCHOR_Score=np.concatenate(ANCHOR_PredictionScore_byProteins)


EspD_PredictionScore=list(map(float, EspD_PredictionScore))
Helix_Score=list(map(float,Helix_Score ))
Coil_Score=list(map(float,Coil_Score ))
Strand_Score=list(map(float,Strand_Score ))


DiordProt_Score=list(map(float,DiordProt_Score))
DiordDNA_Score=list(map(float,DiordDNA_Score))
DiordRNA_Score=list(map(float,DiordRNA_Score))
DFL_Score=list(map(float,DFL_Score))
ANCHOR_Score=list(map(float,ANCHOR_Score))


# In[25]:


ASA=np.concatenate(ASA)


# In[26]:


len(ASA)


# In[27]:


ASA=list(map(float, ASA))


# In[28]:


ASA= np.split(ASA,d)


# In[29]:


NewScore10_byProteins= ASA
ASA_ScoreWindow= [] 
newscore10window=0
b = 0
for NewScore10_byProtein in NewScore10_byProteins:
                                        for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10window= (NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/6
                                                      elif b == 1 : newscore10window= (NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/7
                                                      elif b == 2 : newscore10window= (NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/8
                                                      elif b == 3 : newscore10window= (NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/9
                                                      elif b == 4 : newscore10window= (NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4])/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3])/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2])/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1])/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b])/6  
                                                      else: newscore10window= (NewScore10_byProtein[b-5]+NewScore10_byProtein[b-4]+NewScore10_byProtein[b-3]+NewScore10_byProtein[b-2]+NewScore10_byProtein[b-1]+NewScore10_byProtein[b]+NewScore10_byProtein[b+1]+NewScore10_byProtein[b+2]+NewScore10_byProtein[b+3]+NewScore10_byProtein[b+4]+NewScore10_byProtein[b+5])/11
                                                      ASA_ScoreWindow.append(newscore10window)

ASA_ScoreWindowWeighted= [] 
newscore10windowweighted=0
b = 0
for NewScore10_byProtein in NewScore10_byProteins:
                                        for b in range(0, len(NewScore10_byProtein), 1):
                                                      if b == 0 : newscore10windowweighted= (NewScore10_byProtein[b]*0.65+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/6
                                                      elif b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/7
                                                      elif b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/8
                                                      elif b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/9
                                                      elif b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/10
                                                      elif len(NewScore10_byProtein)-b == 5 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.35+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06)/10
                                                      elif len(NewScore10_byProtein)-b == 4 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.41+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07)/9
                                                      elif len(NewScore10_byProtein)-b == 3 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.48+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08)/8
                                                      elif len(NewScore10_byProtein)-b == 2 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.56+NewScore10_byProtein[b+1]*0.09)/7
                                                      elif len(NewScore10_byProtein)-b == 1 : newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.65)/6  
                                                      else: newscore10windowweighted= (NewScore10_byProtein[b-5]*0.05+NewScore10_byProtein[b-4]*0.06+NewScore10_byProtein[b-3]*0.07+NewScore10_byProtein[b-2]*0.08+NewScore10_byProtein[b-1]*0.09+NewScore10_byProtein[b]*0.3+NewScore10_byProtein[b+1]*0.09+NewScore10_byProtein[b+2]*0.08+NewScore10_byProtein[b+3]*0.07+NewScore10_byProtein[b+4]*0.06+NewScore10_byProtein[b+5]*0.05)/11
                                                      ASA_ScoreWindowWeighted.append(newscore10windowweighted)


# In[30]:


ASA=np.concatenate(ASA)


# In[31]:


len(EspD_PredictionScore)==len(Helix_Score)==len(AA)==len(ASA)


# In[32]:


# KYTJ820101
#Hydropathy index (Kyte-Doolittle, 1982)
Alloc= [1.8,    -4.5,    -3.5,    -3.5,     2.5,    -3.5,    -3.5,    -0.4,    -3.2,     4.5,
     3.8,    -3.9,     1.9,     2.8,    -1.6,    -0.8,    -0.7,    -0.9,    -1.3,     4.2] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Hydropathy=VALUE                      


# In[33]:


# KLEP840101
# Net charge (Klein et al., 1984)
Alloc= [0.0,      1.0,      0.0,     -1.0,      0.0,      0.0,     -1.0,      0.0,      0.0,      0.0,
      0.0,      1.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Net_Charge=VALUE    


# In[34]:


# GRAR740102
# Polarity (Grantham, 1974)
Alloc= [8.1,    10.5,    11.6,    13.0,     5.5,    10.5,    12.3,     9.0,    10.4,     5.2,
     4.9,    11.3,     5.7,     5.2,     8.0,     9.2,     8.6,     5.4,     6.2,     5.9,] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Polarity=VALUE    


# In[35]:


# YUTK870101
#  Unfolding Gibbs energy in water, pH7.0 (Yutani et al., 1987)
Alloc= [8.5,      0.0,     8.2,     8.5,    11.0,     6.3,     8.8,     7.1,    10.1,    16.8,
    15.0,     7.9,    13.3,    11.2,     8.2,     7.4,     8.8,     9.9,     8.8,    12.0] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Gibbs_Energy=VALUE   


# In[36]:


# OOBM850103
#Optimized transfer energy parameter (Oobatake et al., 1985)
Alloc= [0.46,   -1.54,    1.31,   -0.33,    0.20,   -1.12,    0.48,    0.64,   -1.31,    3.28,
    0.43,   -1.71,    0.15,    0.52,   -0.58,   -0.83,   -1.52,    1.25,   -2.21,    0.54] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Transfer_Energy=VALUE   


# In[37]:


# EISD860101
#Solvation free energy (Eisenberg-McLachlan, 1986)
Alloc= [0.67,    -2.1,    -0.6,    -1.2,    0.38,   -0.22,   -0.76,      0.0,    0.64,     1.9,
     1.9,   -0.57,     2.4,     2.3,     1.2,    0.01,    0.52,     2.6,     1.6,     1.5] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Solvation_Energy=VALUE   


# In[38]:


# HUTJ700102
#Absolute entropy (Hutchens, 1970)
Alloc= [30.88,   68.43,   41.70,   40.66,   53.83,   46.62,   44.98,   24.74,   65.99,   49.71,
   50.62,   63.21,   55.32,   51.06,   39.21,   35.65,   36.50,   60.00,   51.15,   42.75] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Absolute_Entropy=VALUE   


# In[39]:


# ZIMJ680104
#Isoelectric point (Zimmerman et al., 1968)
Alloc= [6.00,   10.76,    5.41,    2.77,    5.05,    5.65,    3.22,    5.97,    7.59,    6.02,
    5.98,    9.74,    5.74,    5.48,    6.30,    5.68,    5.66,    5.89,    5.66,    5.96] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Isoelectric_Point=VALUE   


# In[40]:


# CHAM830107
#A parameter of charge transfer capability (Charton-Charton, 1983)
Alloc= [0.0,      0.0,      1.0,      1.0,      0.0,      0.0,      1.0,      1.0,      0.0,      0.0,
      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Charge_Transfer=VALUE   


# In[41]:


# CHAM830108
#A parameter of charge transfer donor capability (Charton-Charton, 1983)
Alloc= [0.0,      1.0,      1.0,      0.0,      1.0,      1.0,      0.0,      0.0,      1.0,      0.0,
      0.0,      1.0,      1.0,      1.0,      0.0,      0.0,      0.0,      1.0,      1.0,      0.0] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Charge_Donor=VALUE   


# In[42]:


# FAUJ880111
#Positive charge (Fauchere et al., 1988)
Alloc= [0.0,      1.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      1.0,      0.0,
      0.0,      1.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Positive_Charge=VALUE   


# In[43]:


# FAUJ880112
#Negative charge (Fauchere et al., 1988)
Alloc= [ 0.0,      0.0,      0.0,      1.0,      0.0,      0.0,      1.0,      0.0,      0.0,      0.0,
      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Negative_Charge=VALUE   


# In[44]:


# ARGP820101
#Hydrophobicity index (Argos et al., 1982)
Alloc= [0.61,    0.60,    0.06,    0.46,    1.07,      0.0,    0.47,    0.07,    0.61,    2.22,
    1.53,    1.15,    1.18,    2.02,    1.95,    0.05,    0.05,    2.65,    1.88,    1.32] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Hydrophobicity_Index=VALUE   


# In[45]:


# EISD840101
#Consensus normalized hydrophobicity scale (Eisenberg, 1984)
Alloc= [0.25,   -1.76,   -0.64,   -0.72,    0.04,   -0.69,   -0.62,    0.16,   -0.40,    0.73,
    0.53,   -1.10,    0.26,    0.61,   -0.07,   -0.26,   -0.18,    0.37,    0.02,    0.54] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Consensus_Hydrophobicity=VALUE   


# In[46]:


# MANP780101
#Average surrounding hydrophobicity (Manavalan-Ponnuswamy, 1978)
Alloc= [12.97,   11.72,   11.42,   10.85,   14.63,   11.76,   11.89,   12.43,   12.16,   15.67,
   14.90,   11.36,   14.39,   14.00,   11.37,   11.23,   11.69,   13.93,   13.42,   15.71] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Surrounding_Hydrophobicity=VALUE   


# In[47]:


# COWR900101
#Hydrophobicity index, 3.0 pH (Cowan-Whittaker, 1990)
Alloc= [0.42,   -1.56,   -1.03,   -0.51,    0.84,   -0.96,   -0.37,    0.00,   -2.28,    1.81,
    1.80,   -2.03,    1.18,    1.74,    0.86,   -0.64,   -0.26,    1.46,    0.51,    1.34] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
PH3_Hydrophobicity=VALUE   


# In[48]:


# CASG920101
#Hydrophobicity scale from native protein structures (Casari-Sippl, 1992)
Alloc= [0.2,    -0.7,    -0.5,    -1.4,     1.9,    -1.1,    -1.3,    -0.1,     0.4,     1.4,
     0.5,    -1.6,     0.5,     1.0,    -1.0,    -0.7,    -0.4,     1.6,     0.5,     0.7] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Native_Hydrophobicity=VALUE   


# In[49]:


# ANDN920101
#alpha-CH chemical shifts (Andersen et al., 1992)
Alloc= [4.35,    4.38,    4.75,    4.76,    4.65,    4.37,    4.29,    3.97,    4.63,    3.95,
    4.17,    4.36,    4.52,    4.66,    4.44,    4.50,    4.35,    4.70,    4.60,    3.95] 
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Alpha_CH=VALUE   


# In[50]:


# BUNA790103
# Spin-spin coupling constants 3JHalpha-NH (Bundi-Wuthrich, 1979)
Alloc= [ 6.5,     6.9,     7.5,     7.0,     7.7,     6.0,     7.0,     5.6,     8.0,     7.0,
     6.5,     6.5,      0.0,     9.4,      0.0,     6.5,     6.9,      0.0,     6.8,     7.0 ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Spin_Coupling=VALUE   


# In[51]:


# DESM900101
# Membrane preference for cytochrome b: MPH89 (Degli Esposti et al., 1990)
Alloc= [  1.56,    0.59,    0.51,    0.23,    1.80,    0.39,    0.19,    1.03,      1.0,    1.27,
    1.38,    0.15,    1.93,    1.42,    0.27,    0.96,    1.11,    0.91,    1.10,    1.58 ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Membrane_Preference=VALUE   


# In[52]:


# EISD860102
# Atom-based hydrophobic moment (Eisenberg-McLachlan, 1986)
Alloc= [ 0.0,     10.0,     1.3,     1.9,    0.17,     1.9,      3.0,      0.0,    0.99,     1.2,
     1.0,     5.7,     1.9,     1.1,    0.18,    0.73,     1.5,     1.6,     1.8,    0.48  ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Hydrophobic_Moment=VALUE   


# In[53]:


# EISD860103
# Direction of hydrophobic moment (Eisenberg-McLachlan, 1986)
Alloc= [0.0,   -0.96,   -0.86,   -0.98,    0.76,    -1.0,   -0.89,      0.0,   -0.75,    0.99,
    0.89,   -0.99,    0.94,    0.92,    0.22,   -0.67,    0.09,    0.67,   -0.93,    0.84  ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Moment_Direction=VALUE   


# In[54]:


# PARS000101
# p-Values of mesophilic proteins based on the distributions of B values (Parthasarathy-Murthy, 2000)
Alloc= [0.343,   0.353,   0.409,   0.429,   0.319,   0.395,   0.405,   0.389,   0.307,   0.296,
   0.287,   0.429,   0.293,   0.292,   0.432,   0.416,   0.362,   0.268,    0.22,   0.307  ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Mesophilic_Value=VALUE   


# In[55]:


# KUMS000101
# Distribution of amino acid residues in the 18 non-redundant families of thermophilic proteins (Kumar et al., 2000)
Alloc= [8.9,     4.6,     4.4,     6.3,     0.6,     2.8,     6.9,     9.4,     2.2,     7.0,
     7.4,     6.1,     2.3,     3.3,     4.2,     4.0,     5.7,     1.3,     4.5,     8.2  ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Thermophilic_Value=VALUE   


# In[56]:


# VINM940103
# Normalized flexibility parameters (B-values) for each residue surrounded by one rigid neighbours (Vihinen et al., 1994)
Alloc= [ 0.994,   1.026,   1.022,   1.022,   0.939,   1.041,   1.052,   1.018,   0.967,   0.977,
   0.982,   1.029,   0.963,   0.934,   1.050,   1.025,   0.998,   0.938,   0.981,   0.968  ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Flexibility_Value=VALUE   


# In[57]:


# NISK860101
# 14 A contact number (Nishikawa-Ooi, 1986)
Alloc= [-0.22,   -0.93,   -2.65,   -4.12,    4.66,   -2.76,   -3.64,   -1.62,    1.28,    5.58,
    5.01,   -4.18,    3.51,    5.27,   -3.03,   -2.84,   -1.20,    5.20,    2.15,    4.45 ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Contact_14A=VALUE   


# In[58]:


# WIMW960101
# Free energies of transfer of AcWl-X-LL peptides from bilayer interface to water (Wimley-White, 1996)
Alloc= [4.08,    3.91,    3.83,    3.02,    4.49,    3.67,    2.23,    4.24,    4.08,    4.52,
    4.81,    3.77,    4.48,    5.38,    3.80,    4.12,    4.11,    6.10,    5.19,    4.18 ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Bilayer_Energy=VALUE   


# In[59]:


# OOBM850105
# Optimized side chain interaction parameter (Oobatake et al., 1985)
Alloc= [ 4.55,    5.97,    5.56,    2.85,   -0.78,    4.15,    5.16,    9.14,    4.48,    2.10,
    3.24,   10.68,    2.18,    4.37,    5.14,    6.78,    8.60,    1.97,    2.40,    3.81 ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
SideChain_Interaction=VALUE   


# In[60]:


# KRIW790102
# Fraction of site occupied by water (Krigbaum-Komoriya, 1979)
Alloc= [  0.28,    0.34,    0.31,    0.33,    0.11,    0.39,    0.37,    0.28,    0.23,    0.12,
    0.16,    0.59,    0.08,    0.10,    0.46,    0.27,    0.26,    0.15,    0.25,    0.22 ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Water_Occupancy=VALUE   


# In[61]:


# ZASB820101
# Dependence of partition coefficient on ionic strength (Zaslavsky et al., 1982)
Alloc= [-0.152,  -0.089,  -0.203,  -0.355, 0.0,  -0.181,  -0.411,  -0.190,  0.0,  -0.086,
  -0.102,  -0.062,  -0.107,   0.001,  -0.181,  -0.203,  -0.170,   0.275,    0.0,  -0.125 ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Ionic_Strength=VALUE   


# In[62]:


# ROSM880102
# Side chain hydropathy, corrected for solvation (Roseman, 1988)
Alloc= [-0.67,    3.89,    2.27,    1.57,   -2.00,    2.12,    1.78,    0.00,    1.09,   -3.02,
   -3.02,    2.46,   -1.67,   -3.24,   -1.75,    0.10,   -0.42,   -2.86,    0.98,   -2.18 ]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
SideChain_Hydropathy=VALUE   


# In[63]:


# NAKH900112
# Transmembrane regions of mt-proteins (Nakashima et al., 1990)
Alloc= [6.61,    0.41,    1.84,   0.59,    0.83,    1.20,    1.63,    4.88,    1.14,   12.91,
   21.66,    1.15,    7.17,    7.76,    3.51,    6.84,    8.89,    2.11,    2.57,    6.30]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Transmem_Affinity=VALUE   


# In[64]:


# JURD980101
# Modified Kyte-Doolittle hydrophobicity scale (Juretic et al., 1998)
Alloc= [1.10,   -5.10,   -3.50,   -3.60,    2.50,   -3.68,   -3.20,   -0.64,   -3.20,    4.50,
    3.80,   -4.11,    1.90,    2.80,   -1.90,   -0.50,   -0.70,   -0.46,    -1.3,     4.2]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Modified_Hydrophobicity=VALUE   


# In[65]:


# YUTK870104
# Activation Gibbs energy of unfolding, pH9.0 (Yutani et al., 1987)
Alloc= [18.56,  0.0,   18.24,   17.94,   17.84,   18.51,   17.97,   18.57,   18.64,   19.21,
   19.01,   18.36,   18.49,   17.95,   18.77,   18.06,   17.71,   16.87,   18.23,   18.98]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
PH9_Gibbs=VALUE   


# In[66]:


# RICJ880105
# Relative preference value at N2 (Richardson-Richardson, 1988)
Alloc= [ 1.6,   0.9,     0.7,     2.6,     1.2,     0.8,      2.0,     0.9,     0.7,     0.7,
     0.3,      1.0,      1.0,     0.9,     0.5,     0.8,     0.7,     1.7,     0.4,     0.6]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
N2_Preference=VALUE   


# In[67]:


# FAUJ880104
# STERIMOL length of the side chain (Fauchere et al., 1988)
Alloc= [2.87,    7.82,    4.58,    4.74,    4.47,    6.11,    5.97,    2.06,    5.23,    4.92,
    4.92,    6.89,    6.36,    4.62,    4.11,    3.97,    4.11,    7.68,    4.73,    4.11]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Sterimol_Length=VALUE   


# In[68]:


# RADA880104
# Transfer free energy from chx to oct (Radzicka-Wolfenden, 1988)
Alloc= [ 1.29,  -13.60,   -6.63,    0.0,    0.0,   -5.47,   -6.02,    0.94,   -5.61,    2.88,
    3.16,   -5.63,    1.03,    0.89,    0.0,   -3.44,   -2.84,   -0.18,   -1.77,    2.86]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Chx_Energy=VALUE   


# In[69]:


# ROBB760109
# Information measure for N-terminal turn (Robson-Suzuki, 1976)
Alloc= [ -3.3,     0.0,     5.4,     3.9,    -0.3,    -0.4,    -1.8,    -1.2,     3.0,    -0.5,
    -2.3,    -1.2,    -4.3,     0.8,     6.5,     1.8,    -0.7,    -0.8,     3.1,    -3.5]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Ntermi_Info=VALUE   


# In[70]:


# LEVM760104
# Side chain torsion angle phi(AAAR) (Levitt, 1976)
Alloc= [ 243.2,   206.6,   207.1,   215.0,   209.4,   205.4,   213.6,   300.0,   219.9,   217.9,
   205.6,   210.9,   204.0,   203.7,   237.4,   232.0,   226.7,   203.7,   195.6,   220.3]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
SideChain_Torsion=VALUE   


# In[71]:


# NAKH900113
#  Ratio of average and computed composition (Nakashima et al., 1990)
Alloc= [1.61,    0.40,    0.73,    0.75,    0.37,    0.61,    1.50,    3.12,    0.46,    1.61,
    1.37,    0.62,    1.59,    1.24,    0.67,    0.68,    0.92,    1.63,    0.67,    1.30]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Computed_Composition=VALUE   


# In[72]:


# FINA910101
# Helix initiation parameter at posision i-1 (Finkelstein et al., 1991)
Alloc= [1.0,    0.70,    1.70,    3.20,     1.0,    1.0,    1.70,      1.0,      1.0,    0.60,
      1.0,    0.70,      1.0,      1.0,      1.0,    1.70,    1.70,      1.0,      1.0,    0.60]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Helix_Initiation=VALUE   


# In[73]:


# ROBB760106
# Information measure for pleated-sheet (Robson-Suzuki, 1976)
Alloc= [-2.7,     0.4,    -4.2,    -4.4,     3.7,     0.8,    -8.1,    -3.9,    -3.0,     7.7,
     3.7,    -2.9,     3.7,     3.0,    -6.6,    -2.4,     1.7,     0.3,     3.3,     7.1]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Pleated_Sheets=VALUE   


# In[74]:


# NAKH900107
# AA composition of mt-proteins from fungi and plant (Nakashima et al., 1990)
Alloc= [ 5.39,    2.81,    7.31,    3.07,    0.86,    2.31,    2.70,    6.52,    2.23,    9.94,
   12.64,    4.67,    3.68,    6.34,    3.62,    7.24,    5.44,    1.64,    5.42,    6.18]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Fungi_Membrane=VALUE   


# In[75]:


# KOEP990101
# Alpha-helix propensity derived from designed sequences (Koehl-Levitt, 1999)
Alloc= [ -0.04,   -0.30,    0.25,    0.27,    0.57,   -0.02,   -0.33,    1.24,   -0.11,   -0.26,
   -0.38,   -0.18,   -0.09,   -0.01,    0.0,    0.15,    0.39,    0.21,    0.05,   -0.06]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Helix_Propensity=VALUE   


# In[76]:


# GEIM800104
#  Alpha-helix indices for alpha/beta-proteins (Geisow-Roberts, 1980)
Alloc= [ 1.19,  1.0,    0.94,    1.07,    0.95,    1.32,    1.64,    0.60,    1.03,    1.12,
    1.18,    1.27,    1.49,    1.02,    0.68,    0.81,    0.85,    1.18,    0.77,    0.74]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Helix_Beta=VALUE   


# In[77]:


# MAXF760101
# Normalized frequency of alpha-helix (Maxfield-Scheraga, 1976)
Alloc= [ 1.43,    1.18,    0.64,    0.92,    0.94,    1.22,    1.67,    0.46,    0.98,    1.04,
    1.36,    1.27,    1.53,    1.19,    0.49,    0.70,    0.78,    1.01,    0.69,    0.98]
VALUE = []
b= 0
for b in range(0,len(AA),1):
            if AA[b] == 'A': value = Alloc[0]                            
            elif AA[b] == 'L': value = Alloc[10]                         
            elif AA[b] == 'R': value = Alloc[1]                          
            elif AA[b] == 'K': value = Alloc[11]                         
            elif AA[b] == 'N': value = Alloc[2]                          
            elif AA[b] == 'M': value = Alloc[12]                         
            elif AA[b] == 'D': value = Alloc[3]                          
            elif AA[b] == 'F': value = Alloc[13]                         
            elif AA[b] == 'C': value = Alloc[4]                          
            elif AA[b] == 'P': value = Alloc[14]                         
            elif AA[b] == 'Q': value = Alloc[5]                          
            elif AA[b] == 'S': value = Alloc[15]                         
            elif AA[b] == 'E': value = Alloc[6]                          
            elif AA[b] == 'T': value = Alloc[16]                         
            elif AA[b] == 'G': value = Alloc[7]                          
            elif AA[b] == 'W': value = Alloc[17]                         
            elif AA[b] == 'H': value = Alloc[8]                          
            elif AA[b] == 'Y': value = Alloc[18]                         
            elif AA[b] == 'I': value = Alloc[9]                          
            elif AA[b] == 'V': value = Alloc[19]                         
            VALUE.append(value)
Normalized_Alpha=VALUE   


# In[ ]:


Scale=[]
b=0
for b in range(0,len(AA),1):
    if   AA[b] == 'A':Scale.append(-0.2038088104698756)
    elif AA[b] == 'C':Scale.append(0.2782851748545145)
    elif AA[b] == 'D':Scale.append(0.24254515724504216)
    elif AA[b] == 'E':Scale.append(0.03489736062782545)
    elif AA[b] == 'F':Scale.append(0.16091732324562574)
    elif AA[b] == 'G':Scale.append(-0.020741793233957202)
    elif AA[b] == 'H':Scale.append(0.7259885226943301)
    elif AA[b] == 'I':Scale.append(-0.14997858203464914)
    elif AA[b] == 'K':Scale.append(0.1789539089442147)
    elif AA[b] == 'L':Scale.append(-0.09763189910003832)
    elif AA[b] == 'M':Scale.append(-0.270795174838208)
    elif AA[b] == 'N':Scale.append(-0.24187247951090707)  
    elif AA[b] == 'P':Scale.append(0.36321907557258487)
    elif AA[b] == 'Q':Scale.append(0.08370646616693535)
    elif AA[b] == 'R':Scale.append(-0.09465484033929307) 
    elif AA[b] == 'S':Scale.append(0.07904900277374191)
    elif AA[b] == 'T':Scale.append(-0.20811116384243883)  
    elif AA[b] == 'V':Scale.append(-0.1916646549489514)
    elif AA[b] == 'W':Scale.append(-0.003462790780651257)
    elif AA[b] == 'Y':Scale.append(-0.18466321666615634)
    else:Scale.append(1)


# In[78]:

print(len(EspD_PredictionScore))
print(len(ASA))
print(len(DiordProt_Score))
print(len(ANCHOR_Score))
print(len(DFL_Score))
print(len(Helix_Score))
Source_Featuers=pd.DataFrame({'EspD_Score1':EspD_PredictionScore,
'ASA1':ASA,
'ASA_ScoreWindowWeighted1':ASA_ScoreWindowWeighted,
'ASA_ScoreWindow1':ASA_ScoreWindow,
'DiordProt_Score1':DiordProt_Score,
'ANCHOR_Score1':ANCHOR_Score,
'DiordDNA_Score1':DiordDNA_Score,
'DiordRNA_Score1':DiordRNA_Score,
'DFL_Score1':DFL_Score,
'Helix_Score1':Helix_Score,
'Coil_Score1':Coil_Score,
'Strand_Score1':Strand_Score})   
Source_Featuers.head()


# In[81]:


Target_Featuers=pd.DataFrame({'Water_Occupancy': Water_Occupancy,           
'SideChain_Interaction':SideChain_Interaction,
'Bilayer_Energy':Bilayer_Energy,
'Contact_14A':Contact_14A,
'Flexibility_Value':Flexibility_Value,
'Thermophilic_Value':Thermophilic_Value,
'Mesophilic_Value':Mesophilic_Value,
'Moment_Direction':Moment_Direction,
'Hydrophobic_Moment':Hydrophobic_Moment,
'Membrane_Preference':Membrane_Preference,
'Spin_Coupling':Spin_Coupling,
'Alpha_CH':Alpha_CH,
'Native_Hydrophobicity':Native_Hydrophobicity,
'PH3_Hydrophobicity':PH3_Hydrophobicity,
'Surrounding_Hydrophobicity':Surrounding_Hydrophobicity,
'Consensus_Hydrophobicity':Consensus_Hydrophobicity,
'Hydrophobicity_Index':Hydrophobicity_Index,
'Negative_Charge':Negative_Charge,
'Positive_Charge':Positive_Charge,
'Charge_Donor':Charge_Donor,
'Charge_Transfer':Charge_Transfer,
'Isoelectric_Point':Isoelectric_Point,
'Absolute_Entropy':Absolute_Entropy,
'Solvation_Energy':Solvation_Energy,
'Transfer_Energy':Transfer_Energy,
'Gibbs_Energy':Gibbs_Energy,
'Polarity':Polarity,
'Net_Charge':Net_Charge,
'Hydropathy':Hydropathy,
'Helix_Score':Helix_Score,
'Coil_Score':Coil_Score,
'Strand_Score':Strand_Score,
'EspD_Score':EspD_PredictionScore,
'ASA':ASA,
'Normalized_Alpha':Normalized_Alpha,           
'Helix_Beta':Helix_Beta,
'Helix_Propensity':Helix_Propensity,
'Fungi_Membrane':Fungi_Membrane,
'Pleated_Sheets':Pleated_Sheets,
'Helix_Initiation':Helix_Initiation,
'Computed_Composition':Computed_Composition,
'SideChain_Torsion':SideChain_Torsion,
'Ntermi_Info':Ntermi_Info,
'Chx_Energy':Chx_Energy,
'Sterimol_Length':Sterimol_Length,
'N2_Preference':N2_Preference,
'PH9_Gibbs':PH9_Gibbs,
'Modified_Hydrophobicity':Modified_Hydrophobicity,
'Transmem_Affinity':Transmem_Affinity,
'SideChain_Hydropathy':SideChain_Hydropathy,
'Ionic_Strength':Ionic_Strength,
'ASA_ScoreWindowWeighted':ASA_ScoreWindowWeighted,
'ASA_ScoreWindow':ASA_ScoreWindow,
'Lipid_Scale': Scale})   
Target_Featuers.head()

text_file = open(result_dir+"/command1.txt", "w")
text_file.write(result_dir+"/results2.html.........")
text_file.write(result_dir+"/results.txt...........")
text_file.write(command)
text_file.close()

# In[ ]:


#import time
#time.sleep(10)

model_base = tf.keras.models.load_model('Base_Interaction_DisoOnly_FuncPred_BID.h5')


# In[ ]:


modelT = tf.keras.models.load_model('Top_Lipid_PredDiso_FuncPred_SpotwithBaseinputs_BIDFull_1.h5')


# In[ ]:


result = pd.concat([Target_Featuers, Source_Featuers], axis=1)


# In[8]:


Test_Featuers=result


# In[9]:


Test_Featuers.shape


# In[10]:


FullTest_EspD = Test_Featuers.loc[:,('EspD_Score1')].values


# In[12]:


test_disorder_indexes=[]
test_predorder_score=[]
test_predorder_indexes=[]
b=0
for b in range(0,len(FullTest_EspD),1):
                    if FullTest_EspD[b]>0.3:
                                    test_disorder_indexes.append(b)
                    else:
                        test_predorder_indexes.append(b)
                        test_predorder_score.append(FullTest_EspD[b])


# In[13]:


DisoTest_set=Test_Featuers.iloc[test_disorder_indexes,:]
Test_Featuers=DisoTest_set


# In[14]:


X_test_baseIn= np.array(Test_Featuers.iloc[:,54:66])


# In[15]:


X_test_baseIn = X_test_baseIn.reshape(X_test_baseIn.shape[0], X_test_baseIn.shape[1], 1)


# In[16]:


Test_EspD = Test_Featuers.loc[:,('EspD_Score1')].values
ASA = Test_Featuers.loc[:,('ASA')].values
ASAWindow = Test_Featuers.loc[:,('ASA_ScoreWindow')].values
ASAWindowWeight = Test_Featuers.loc[:,('ASA_ScoreWindowWeighted')].values


# In[17]:


print('Before Base')

with tf.device('/cpu:2'):
                  test_base = model_base.predict(X_test_baseIn, batch_size=100)

test_base=test_base.reshape(test_base.shape[0], )


# In[18]:


Test_Small=pd.DataFrame({'Base_Score':test_base,   
'EspD_Score': Test_EspD,
'ASA': ASA,
'ASA_ScoreWindowWeighted':ASAWindowWeight,
'ASA_ScoreWindow':ASAWindow }) 


# In[19]:


Xsmall_Test= np.array(Test_Small.iloc[:,0:])


# In[20]:


Xsmall_Test = Xsmall_Test.reshape(Xsmall_Test.shape[0], Xsmall_Test.shape[1], 1)
Xsmall_Test.shape


# In[21]:


with tf.device('/cpu:2'):
                  test_score = modelT.predict(Xsmall_Test, batch_size=50)


# In[22]:
print('After Target')



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.1,1.0))
scaler.fit(test_score)
test_score=scaler.transform(test_score)
test_score=np.concatenate(test_score)


# In[23]:


test_predorder_score=np.asarray(test_predorder_score)
test_predorder_score=test_predorder_score.reshape(-1, 1)


# In[24]:



scaler = MinMaxScaler(feature_range=(0.0,0.3))
scaler.fit(test_predorder_score)
test_predorder_score=scaler.transform(test_predorder_score)
test_predorder_score=np.concatenate(test_predorder_score)


# In[26]:


Test_FullPrediction = np.zeros((len(FullTest_EspD),),dtype=float)


# In[27]:


np.put(Test_FullPrediction ,test_disorder_indexes,test_score)
np.put(Test_FullPrediction ,test_predorder_indexes,test_predorder_score)

Binary=[]
b=0
for b in range(0,len(Test_FullPrediction),1):
    if Test_FullPrediction[b]>0.451:Binary.append(1)
    else:Binary.append(0)

Test_FullPrediction=np.around(Test_FullPrediction,decimals=3)
PredictionScore_byProteins= np.split(Test_FullPrediction,d)
Binary_byProteins= np.split(Binary,d)



text_file = open(result_dir+"/command2.txt", "w")
text_file.write(result_dir+"/results2.html....")
text_file.write(result_dir+"/results.txt....")
text_file.write(command)
text_file.close()

Lines=[]
b=0
for b in range(0,len(ID),1):
           Lines.append(ID[b])
           x = str(list(Sequence[b]))
           x = x[1:-1]
           x = x.replace(",", "")
           x = x.replace(" ", "")
           x = x.replace("'", "")
           Lines.append(x)
        
           x = str(list(Binary_byProteins[b]))
           x = x[1:-1]
           x = x.replace(",", "")
           x = x.replace(" ", "")
           x = x.replace("'", "")
           Lines.append(x)
           x = str(list(PredictionScore_byProteins[b]))
           x = x[1:-1]
           x = x.replace(" ", "")
           x = x.replace("'", "")
           Lines.append(x)


with open(r"seqs.fasta.out", 'w') as f:
    for item in Lines:
        f.write("%s\n" % item)
j=0
for j in range(0,len(ID),1):
     Lipid_Residues= np.concatenate(np.argwhere(Binary_byProteins[j]>0))
     Position =np.arange(0,len(PredictionScore_byProteins[j]),1)
     AminoAcids=list(Sequence[j])
     Y_Sequence = np.full(len(Position), 1.5)
     Y_Lipid = np.full(len(Position), 1)
     buffer = io.StringIO()
     fig = make_subplots(rows=4, cols=1,
                         shared_xaxes=True,
                         vertical_spacing=0.01,
                         specs=[ [{'rowspan':3}],
                                [None],
                                 [None],
                                  [{}]    ]
                        )
     fig.add_trace(go.Scatter(x=Position, y=PredictionScore_byProteins[j],name='Disordered Lipid Binding Propensity',
             marker=dict(color='blue'),
             hovertemplate='Score: %{y:.3f}'+'<br>Position: %{x}'),row=1, col=1)
     
     fig.add_shape(
             type='line',
             x0=0,
             y0=0.451,
             x1=len(Position)-1,
             y1=0.451,
             line=dict(
                 color='blue',
                 dash="dot",
                 width=3),
                   row=1, col=1)
     
     fig.add_trace(go.Scatter(
         x=Position,
         y=Y_Sequence,
         mode="text",
         name="Sequence",
         text=AminoAcids,
         textposition="bottom center",
         marker=dict(size=2,color='black'),
         meta=AminoAcids,
         hovertemplate='Residue: %{meta}'+'<br>Position: %{x}'),
          row=4, col=1 )
     
     fig.add_trace(go.Scatter(x=Lipid_Residues, y=Y_Lipid,mode='markers',name='Disordered Lipid Binding Regions',
         marker=dict(size=16,color='blue', symbol='square')),
                   row=4, col=1)
     tickvals = [1,1.5]
     ticktext = ['Lipid Binding','Sequence']
     
     fig.update_layout( {'height':400, 
                         'width':1000,
                         'title_text':ID[j][1:],
                         'xaxis':{'range': [0, len(Position)-1],'mirror':True},
                         'xaxis2':{'range': [0, len(Position)-1],'mirror':True},
                         'yaxis':{'range': [0, 1], 'dtick': 0.2,'mirror':True,'title':' DisLipProt Propensity'},
                         'yaxis2':{'tickvals': tickvals, 'ticktext': ticktext,'mirror':True},
                         'paper_bgcolor':'rgba(255,255,255)',
                         'plot_bgcolor':'rgba(0, 0, 0, 0)',
                         'legend_title_text':'Click on legend to show/hide',
                         'showlegend':True,
                          }
                         )
     
     fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black')
     fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black')
     fig.write_html(result_dir+"/"+ID[j][1:]+'.html')
	 #fig.write_html(ID[j][1:]+'.html')
     print('draw graphic')
     fig.show()

def write_to_html(ID_List,result_dir):
         spl = result_dir.split("/")
         output_dir_url_suffix = "/".join(spl[3:-1])
         output_folder_url  = "http://biomine.cs.vcu.edu/"+output_dir_url_suffix
         txt_url_suffix = output_folder_url+"/results.txt"
         html_begining = r'''
         <!DOCTYPE html>
	     <html lang="en">
	     <head>
	     <meta charset="utf-8">
	     <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
	     <meta http-equiv="X-UA-Compatible" content="IE=Edge">
	     <title>DisLipProt Server - Results Page</title>
	     <link rel="stylesheet" href="../../../css/bootstrap.css">
	     <link rel="stylesheet" href="../../../assets/css/custom.min.css">
	     <link rel="stylesheet" href="../../../servers/biomine.css">
	     <link rel="stylesheet" href="../../stylesSCRIBER.css">
	     </head>
	     <body>
	     <div class="container">
	     <div class="row">
	     <div class="col-lg-8 col-lg-offset-2">
	     <h1>DisLipProt results page</h1>
	     <p>Results for <a target="_blank" href="http://biomine.cs.vcu.edu/servers/DisLipProt/">DisLipProt</a> webserver.</p>
	     <p>Use this link to download the results as a text file:
	     <a target="_blank" href="'''+txt_url_suffix+r'''">results.txt</a></p>
	     </div>
	     </div>
	     <div class="row">
	     <div class="col-lg-8 col-lg-offset-2">
	     <div class="Predictions">
	     <h2>Results</h2>
	     <div class="table-responsive">
         '''  
         html_end  = r'''
	     </div>
	     </div>
	     </div>
	     <div class="row">
	     <div class="col-lg-8 col-lg-offset-2">
	     <footer>
	     <h2>Visit biomine lab web page</h2>
	     <a target="_blank" href="http://biomine.cs.vcu.edu">http://biomine.cs.vcu.edu</a>
	     </footer>
	     </div>
	     </div>
	     </div>
	     </body>
	     </html>
	     '''
         html_middle =""
         for i in range(len(ID_List)):
                        html_middle += '''<object width="1000px" height = "400px" data="'''+output_folder_url+"/"+ID_List[i][1:]+".html"+r'''"></object>'''
         with open(result_dir,"w") as outf:
                outf.writelines(html_begining)
                outf.writelines(html_middle)
                outf.writelines(html_end)
	 	
write_to_html(ID,result_dir+"/results2.html")
#write_to_html(ID,"results2.html")

text_file = open("command3.txt", "w")
text_file.write(result_dir+"/results2.html....")
text_file.write(result_dir+"/results.txt....")
text_file.write(command)
text_file.close()
