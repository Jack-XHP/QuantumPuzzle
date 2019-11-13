from dwave_qbsolv import QBSolv
import matplotlib.pyplot as plt
import numpy as np
# %% Functions    
def DisplayIn(In,m,n):
    II=In[0:m*n]
    HI=In[m*n:(2*m*n-m)]
    VI=In[(2*m*n-m):(3*m*n-m-n)]
    D=np.zeros((2*m-1,2*n-1))
    for i in range(0,2*m-1):
        for j in range(0,2*n-1):
            if (i%2==0)&(j%2==0):
                D[i,j]=II[i//2*n+j//2]
            elif (i%2==1)&(j%2==0):
                D[i,j]=VI[i//2*n+j//2]
            elif (i%2==0)&(j%2==1):
                D[i,j]=HI[i//2*(n-1)+j//2]
    return D
    
def ExpandGrid(In):
    (yl,xl)=In.shape
    Out=np.zeros((2*yl-1,2*xl-1),dtype='int32')
    Out[:,1:-1:2]-=2
    Out[1:-1:2,:]-=3
    for i in range(0,yl):
        for j in range(0,xl):
            Out[2*i,2*j]=In[i,j]
    return Out    
# %% Test
    
# Input
In=x=np.array([[0,0,0,-1,0,0,0,-1,0,-1],[0,0,0,0,0,0,0,0,0,0],[1,0,0,1,1,0,0,0,0,0],[0,-1,0,0,1,-1,0,0,0,0],[0,0,0,0,0,0,1,0,0,1],[1,-1,0,0,0,0,0,0,0,1],[0,0,0,1,1,0,0,1,0,0],[0,0,0,1,0,0,-1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[-1,0,0,1,0,0,0,0,0,-1]])
# Exrtraxt Dimenssions
(m,n)=In.shape
H=m*n
V=2*m*n-m
# Declare Coefficients
Coefficients=np.zeros((3*m*n-m-n,3*m*n-m-n),dtype='float32')
# White/Black Crossing Condition
for i in range(m):
        for j in range(n):
            if In[i,j]==1:
                if j%n!=0: #aa
                    Coefficients[H+i*(n-1)+j-1,H+i*(n-1)+j-1]-=3 
                if i!=0: #bb
                    Coefficients[V+i*n+j-n,V+i*n+j-n]-=3
                if j%n!=n-1: #cc
                    Coefficients[H+i*(n-1)+j,H+i*(n-1)+j]-=3
                if i!=m-1: #dd
                    Coefficients[V+i*n+j,V+i*n+j]-=3
                    
                if (j%n!=0)&(i!=0): #ab
                    Coefficients[H+i*(n-1)+j-1,V+i*n+j-n]+=3
                if (j%n!=0)&(j%n!=n-1): #ac
                    Coefficients[H+i*(n-1)+j-1,H+i*(n-1)+j]+=2
                if (j%n!=0)&(i!=m-1): #ad
                    Coefficients[H+i*(n-1)+j-1,V+i*n+j]+=3 
                if (i!=0)&(j%n!=n-1): #bc
                    Coefficients[V+i*n+j-n,H+i*(n-1)+j]+=3
                if (i!=0)&(i!=m-1): #bd
                    Coefficients[V+i*n+j-n,V+i*n+j]+=2
                if (j%n!=n-1)&(i!=m-1): #cd
                    Coefficients[H+i*(n-1)+j,V+i*n+j]+=3
                    
            elif In[i,j]==-1:
                if j%n!=0: #aa
                    Coefficients[H+i*(n-1)+j-1,H+i*(n-1)+j-1]-=5 
                if i!=0: #bb
                    Coefficients[V+i*n+j-n,V+i*n+j-n]-=8 
                if j%n!=n-1: #cc
                    Coefficients[H+i*(n-1)+j,H+i*(n-1)+j]-=5 
                if i!=m-1: #dd
                    Coefficients[V+i*n+j,V+i*n+j]-=8
                
                if (j%n!=0)&(i!=0): #ab
                    Coefficients[H+i*(n-1)+j-1,V+i*n+j-n]+=4 
                if (j%n!=0)&(j%n!=n-1): #ac
                    Coefficients[H+i*(n-1)+j-1,H+i*(n-1)+j]+=2
                if (j%n!=0)&(i!=m-1): #ad
                    Coefficients[H+i*(n-1)+j-1,V+i*n+j]+=4
                if (i!=0)&(j%n!=n-1): #bc
                    Coefficients[V+i*n+j-n,H+i*(n-1)+j]+=4
                if (i!=0)&(i!=m-1): #bd
                    Coefficients[V+i*n+j-n,V+i*n+j]+=8
                if (j%n!=n-1)&(i!=m-1): #cd
                    Coefficients[H+i*(n-1)+j,V+i*n+j]+=4

            Coefficients[i*n+j,i*n+j]+=4 #II         
            if j%n!=0: 
                Coefficients[H+i*(n-1)+j-1,H+i*(n-1)+j-1]+=1 #aa
                Coefficients[H+i*(n-1)+j-1,i*n+j]-=4 #aI
            if i!=0: 
                Coefficients[V+i*n+j-n,V+i*n+j-n]+=1 #bb
                Coefficients[V+i*n+j-n,i*n+j]-=4 #bI
            if j%n!=n-1: 
                Coefficients[H+i*(n-1)+j,H+i*(n-1)+j]+=1  #cc
                Coefficients[H+i*(n-1)+j,i*n+j]-=4  #cI
            if i!=m-1: 
                Coefficients[V+i*n+j,V+i*n+j]+=1 #dd
                Coefficients[V+i*n+j,i*n+j]-=4 #dI
            
            if (j%n!=0)&(i!=0): #ab
                Coefficients[H+i*(n-1)+j-1,V+i*n+j-n]+=2 
            if (j%n!=0)&(j%n!=n-1): #ac
                Coefficients[H+i*(n-1)+j-1,H+i*(n-1)+j]+=2 
            if (j%n!=0)&(i!=m-1): #ad
                Coefficients[H+i*(n-1)+j-1,V+i*n+j]+=2 
            if (i!=0)&(j%n!=n-1): #bc
                Coefficients[V+i*n+j-n,H+i*(n-1)+j]+=2 
            if (i!=0)&(i!=m-1): #bd
                Coefficients[V+i*n+j-n,V+i*n+j]+=2 
            if (j%n!=n-1)&(i!=m-1): #cd
                Coefficients[H+i*(n-1)+j,V+i*n+j]+=2 
                print(V+i*n+j)
  
                
            
# Filling The Coefficients
D=dict() 
for i in range(3*m*n-m-n):
    for j in range(i+1):
        if i!=j:
            D[(i,j)]= Coefficients[i,j]+Coefficients[j,i]
        elif i==j:
            D[(i,j)]= Coefficients[i,j]
  
# Solve the Puzzle
res = QBSolv().sample_qubo(D,num_repeats=1000)
samples = list(res.samples())
energy = list(res.data_vectors['energy'])
print(samples)
print(energy)
# Represent the Results
for i in range(len(samples)):
    result = samples[i]
    output = []
    # ignore ancillary variables, which are all negative, only get positive bits
    for x in range(3*m*n-m-n):
        output.append(result[x])
    output = DisplayIn(output,m,n)
    print("energy: {}_____________________________".format(energy[i]))
    print(output)
    