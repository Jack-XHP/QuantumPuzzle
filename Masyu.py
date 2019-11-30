from dwave_qbsolv import QBSolv
import matplotlib.pyplot as plt
import numpy as np
import minorminer
import networkx as nx
from dwave.system.composites import FixedEmbeddingComposite
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
# Functions    
def mapseq(F):
    S=np.zeros(F.shape[0])
    for i in range(F.shape[0]):
        S[i]=np.asscalar(np.where(F[i,:]==1)[0])
    return S
# Input
r=16
In=x=np.array([[ 0, 0, 0, 0],
               [ 1, 0, 0, 1],
               [ 0, 0,-1, 0],
               [ 0, 0, 1, 0]])
# Exrtraxt Dimenssions
(m,n)=In.shape
# Declare Coefficients
Coefficients=np.zeros((r*m*n,r*m*n),dtype='float32')
# Valid Map (squence of vertices) Condition
for i in range(r):
    for j in range(m*n):
        Coefficients[i*m*n+j,i*m*n+j]-=2
        for k in range(m*n):
            Coefficients[i*m*n+j,i*m*n+k]+=1      
# Injective Map (Vertix appears only once) Condition
for j in range(m*n):
    for i in range(r):
        Coefficients[i*m*n+j,i*m*n+j]-=1
        for k in range(r):
            Coefficients[i*m*n+j,k*m*n+j]+=1
# Forming a cycle Condition
Ind=set({})
for j1 in range(m*n):
    for j2 in range(m*n):
        Ind.update({(j1,j2)})
for i in range(m):
    for j in range(n-1):
           Ind-={(j+i*n,j+i*n+1)}     
for i in range(m-1):
    for j in range(n):
           Ind-={(j+i*n,j+i*n+n)}  
for i in range(m):
    for j in range(1,n):
           Ind-={(j+i*n,j+i*n-1)}     
for i in range(1,m):
    for j in range(n):
           Ind-={(j+i*n,j+i*n-n)}  
for i in range(r-1):
    for (j1,j2) in Ind:
        Coefficients[i*m*n+j1,(i+1)*m*n+j2]+=1
for (j1,j2) in Ind:
    Coefficients[(r-1)*m*n+j1,(0)*m*n+j2]+=1           
# White circles Condition
for i in range(1,r-1):
    for j in range(m*n):
        if (In[j//n,j%n]==1)&(0<j%n<n-1):
            Coefficients[(i-1)*m*n+j-1,(i+1)*m*n+j+1]-=0.5
            Coefficients[(i-1)*m*n+j+1,(i+1)*m*n+j-1]-=0.5
        if (In[j//n,j%n]==1)&(0<j//n<m-1):   
            Coefficients[(i-1)*m*n+j-n,(i+1)*m*n+j+n]-=0.5
            Coefficients[(i-1)*m*n+j+n,(i+1)*m*n+j-n]-=0.5
for j in range(m*n):
        if (In[j//n,j%n]==1)&(0<j%n<n-1):
            Coefficients[(r-2)*m*n+j-1,(0)*m*n+j+1]-=0.5
            Coefficients[(r-2)*m*n+j+1,(0)*m*n+j-1]-=0.5
        if (In[j//n,j%n]==1)&(0<j//n<m-1):       
            Coefficients[(r-2)*m*n+j-n,(0)*m*n+j+n]-=0.5
            Coefficients[(r-2)*m*n+j+n,(0)*m*n+j-n]-=0.5    
for j in range(m*n):
        if (In[j//n,j%n]==1)&(0<j%n<n-1):
            Coefficients[(r-1)*m*n+j-1,(1)*m*n+j+1]-=0.5
            Coefficients[(r-1)*m*n+j+1,(1)*m*n+j-1]-=0.5
        if (In[j//n,j%n]==1)&(0<j//n<m-1):       
            Coefficients[(r-1)*m*n+j-n,(1)*m*n+j+n]-=0.5
            Coefficients[(r-1)*m*n+j+n,(1)*m*n+j-n]-=0.5        
# Black circles Condition
for i in range(1,r-1):
    for j in range(m*n):
        if (In[j//n,j%n]==-1)&(0<j%n)&(0<j//n):
            Coefficients[(i-1)*m*n+j-1,(i+1)*m*n+j-n]-=0.5
            Coefficients[(i-1)*m*n+j-n,(i+1)*m*n+j-1]-=0.5
        if (In[j//n,j%n]==-1)&(j%n<n-1)&(0<j//n):   
            Coefficients[(i-1)*m*n+j-n,(i+1)*m*n+j+1]-=0.5
            Coefficients[(i-1)*m*n+j+1,(i+1)*m*n+j-n]-=0.5
        if (In[j//n,j%n]==-1)&(j%n<n-1)&(j//n<m-1):   
            Coefficients[(i-1)*m*n+j+n,(i+1)*m*n+j+1]-=0.5
            Coefficients[(i-1)*m*n+j+1,(i+1)*m*n+j+n]-=0.5
        if (In[j//n,j%n]==-1)&(0<j%n)&(j//n<m-1):   
            Coefficients[(i-1)*m*n+j+n,(i+1)*m*n+j-1]-=0.5
            Coefficients[(i-1)*m*n+j-1,(i+1)*m*n+j+n]-=0.5
for j in range(m*n):
        if (In[j//n,j%n]==-1)&(0<j%n)&(0<j//n):
            Coefficients[(r-2)*m*n+j-1,(0)*m*n+j-n]-=0.5
            Coefficients[(r-2)*m*n+j-n,(0)*m*n+j-1]-=0.5
        if (In[j//n,j%n]==-1)&(j%n<n-1)&(0<j//n):   
            Coefficients[(r-2)*m*n+j-n,(0)*m*n+j+1]-=0.5
            Coefficients[(r-2)*m*n+j+1,(0)*m*n+j-n]-=0.5
        if (In[j//n,j%n]==-1)&(j%n<n-1)&(j//n<m-1):   
            Coefficients[(r-2)*m*n+j+n,(0)*m*n+j+1]-=0.5
            Coefficients[(r-2)*m*n+j+1,(0)*m*n+j+n]-=0.5
        if (In[j//n,j%n]==-1)&(0<j%n)&(j//n<m-1):   
            Coefficients[(r-2)*m*n+j+n,(0)*m*n+j-1]-=0.5
            Coefficients[(r-2)*m*n+j-1,(0)*m*n+j+n]-=0.5
for j in range(m*n):
        if (In[j//n,j%n]==-1)&(0<j%n)&(0<j//n):
            Coefficients[(r-1)*m*n+j-1,(1)*m*n+j-n]-=0.5
            Coefficients[(r-1)*m*n+j-n,(1)*m*n+j-1]-=0.5
        if (In[j//n,j%n]==-1)&(j%n<n-1)&(0<j//n):   
            Coefficients[(r-1)*m*n+j-n,(1)*m*n+j+1]-=0.5
            Coefficients[(r-1)*m*n+j+1,(1)*m*n+j-n]-=0.5
        if (In[j//n,j%n]==-1)&(j%n<n-1)&(j//n<m-1):   
            Coefficients[(r-1)*m*n+j+n,(1)*m*n+j+1]-=0.5
            Coefficients[(r-1)*m*n+j+1,(1)*m*n+j+n]-=0.5
        if (In[j//n,j%n]==-1)&(0<j%n)&(j//n<m-1):   
            Coefficients[(r-1)*m*n+j+n,(1)*m*n+j-1]-=0.5
            Coefficients[(r-1)*m*n+j-1,(1)*m*n+j+n]-=0.5
for j in range(m*n):
    if In[j//n,j%n]==-1:
        for i in range(r):
            Coefficients[i*m*n+j,i*m*n+j]-=2
            for k in range(r):
                Coefficients[i*m*n+j,k*m*n+j]+=1
# Filling The Coefficients
D=dict() 
for i in range(r*m*n):
    for j in range(i+1):
        if i!=j:
            D[(i,j)]= Coefficients[i,j]+Coefficients[j,i]
        elif i==j:
            D[(i,j)]= Coefficients[i,j]
# Solve the Puzzle
use_qpu=True
if use_qpu:
    solver_limit = 256
    G = nx.complete_graph(solver_limit)
    system = DWaveSampler(token='DEV-6189564036d19f88b3a555b4175a353d6d2c0218')
    embedding = minorminer.find_embedding(D.keys(), system.edgelist)
    print(embedding)
    res = QBSolv().sample_qubo(D, solver=FixedEmbeddingComposite(system, embedding), solver_limit=solver_limit,token='DEV-6189564036d19f88b3a555b4175a353d6d2c0218', num_reads=20)
    #Emb = EmbeddingComposite(DWaveSampler(token='DEV-6189564036d19f88b3a555b4175a353d6d2c0218'))
    #res = Emb.sample_qubo(D, num_reads=10000)
else:
    res = QBSolv().sample_qubo(D,num_repeats=20)
samples = list(res.samples())
energy = list(res.data_vectors['energy'])
print(samples)
print(energy)
# Represent the Results
for i in range(len(samples)):
    result = samples[i]
    output = []
    # ignore ancillary variables, which are all negative, only get positive bits
    for x in range(r*m*n):
        output.append(result[x])
    output = np.array(output)#DisplayOut(In,np.array(output),r)
    F=output.reshape(r,m*n)
    output=mapseq(F)
    print("energy: {}_____________________________".format(energy[i]))
    print(output)
