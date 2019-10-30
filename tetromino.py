
# coding: utf-8

import numpy as np
from blueqat import opt
import datetime
import time
from dwave_qbsolv import QBSolv

#five pattern of tiles
p = [[[1,1,1,1]],
      [[1,1],[1,1]],
      [[1,1,1],[1,0,0]],
      [[1,1,1],[0,1,0]],
      [[1,1,0],[0,1,1]]]

def get_rotate_and_flip(piece):
    tmp_array=[]
    piece_f = np.flip(piece, axis=1).tolist()
    target=piece
    for i in range(2):
        if i==1: target=piece_f
        for j in range(4):
            tmp= np.rot90(target, k=j).tolist()
            f=1
            for item in tmp_array:
                if np.array_equal(tmp, item):
                    f=0
                    break
            if f==1:
                tmp_array.append(tmp)
    return tmp_array

#res = get_rotate_and_flip([[1,1,1],[1,0,0]])
#for item in res:
#    print(item)

def get_piece_board(px):
    b = [i+1 for i in range(40)]
    b = np.array(b).reshape((5,8))
    (sx, sy) = np.array(px).shape
    tmp_array = []
    for i in range(b.shape[0]-(sx-1)):
        for j in range(b.shape[1]-(sy-1)):
            tmp = (b[i:i+sx,j:j+sy]*px).reshape(-1)
            tmp2 = tmp[np.where(tmp>0)[0]]
            tmp_array.append(tmp2.tolist())
    return tmp_array


#px=[[1, 1, 1], [1, 0, 0]]
#res =  get_piece_board(px)
#print(np.array(res))
#print(len(res))

def check_answer(answer):
    check=True
    tmp1 = []
    tmp2 = []
    for i in range(len(answer)):
        tmp1.append(qubo_prb[answer[i]][0])
        tmp2.append(prb[qubo_prb[answer[i]][0]][qubo_prb[answer[i]][1]][qubo_prb[answer[i]][2]])
    for i in range(4):
        if len(np.where(np.array(tmp1)==i)[0])!=2:
            check= False
    tmp3=np.array(tmp2).reshape(-1)
    #重なりやギャップが無いか
    for i in range(40):
        if len(np.where(np.array(tmp3)==(i+1))[0])!=1:
            check= False
    return check

import matplotlib.pyplot as plt
import matplotlib.patches as patches
def cplot(answer, prb, filename):
    cmap=['#FFFF99','#FF99FF','#99FFFF','#FF9999','#99FF99','#9999FF','#999933','#993399','#339999','#993333','#339933','#333399']
    fig = plt.figure()
    ax = plt.axes()
    tmp=[]
    for i in range(0, 40):
        x = i%8
        y = 3-i//8
        r = patches.Rectangle(xy=(x*5, y*5), width=5, height=5, ec='#000000', fill=False)
        ax.add_patch(r)
    for i in range(len(answer)):
        item=prb[qubo_prb[answer[i]][0]][qubo_prb[answer[i]][1]][qubo_prb[answer[i]][2]]
        for item1 in item:
            if len(np.where(np.array(tmp)==item1)[0])>0: cc='#FF0000'
            #else: cc = cmap[qubo_prb[answer[i]][0]]
            else: cc = cmap[i]
            tmp.append(item1)
            x = (item1-1)%8
            y = 3-(item1-1)//8
            r = patches.Rectangle(xy=(x*5, y*5), width=5, height=5, ec=cc, fc=cc, fill=True)
            ax.add_patch(r)
    for i in range(0, 40):
        x = i%8
        y = 3-i//8
        plt.text(x*5+2, y*5+2, i+1)
    plt.axis('scaled')
    ax.set_aspect('equal')
    plt.savefig(filename)

#make pieces
pr = []
for index, item in enumerate(p):
    pr.append(get_rotate_and_flip(item))
print(pr)

prb = []
for i in range(0, len(pr)):
    prb.append([])
    count = 0
    for j in range(len(pr[i])):
        #print("pr", pr[i][j])
        res =  get_piece_board(pr[i][j])
        count += len(res)
        prb[i].append(res)
    print(count)

#making qubo
qubo_count = 0
qubo_prb=[]
prb_qubo=np.zeros((8, 8, 60))
for i in range(len(prb)):
    for j in range(len(prb[i])):
        for k in range(len(prb[i][j])):
            qubo_prb.append([i,j,k])
            prb_qubo[i][j][k] = qubo_count
            qubo_count +=1
print(qubo_count)

qubo1 = np.zeros((qubo_count,qubo_count))
for i in range(len(prb)):
    tmp=[]
    for j in range(len(prb[i])):
        for k in range(len(prb[i][j])):
            tmp.append(int(prb_qubo[i][j][k]))
    for j in range(len(tmp)):
        for k in range(j, len(tmp)):
            if j==k: qubo1[tmp[j]][tmp[j]]+=-3
            else: qubo1[tmp[j]][tmp[k]]+=2
#print(qubo1)
#print(np.max(qubo1))
#print(np.min(qubo1))

qubo2 = np.zeros((qubo_count,qubo_count))
for l in range(1,41):
    tmp=[]
    for i in range(len(prb)):
        for j in range(len(prb[i])):
            for k in range(len(prb[i][j])):
                if len(np.where(np.array(prb[i][j][k])==l)[0])>=1:
                    tmp.append(int(prb_qubo[i][j][k]))
    print(l, tmp)
    for j in range(len(tmp)):
        for k in range(j, len(tmp)):
            if j==k: qubo2[tmp[j]][tmp[j]]+=-1
            else: qubo2[tmp[j]][tmp[k]]+=2

#print(qubo2)
#print(np.max(qubo2))
#print(np.min(qubo2))

A=0.5
B=0.15
qubo=A*qubo1+B*qubo2

Q={}
for i in range(len(qubo)):
    for j in range(len(qubo)):
        Q.update({(i,j):qubo[i][j]})

import networkx as nx
from dwave_qbsolv import QBSolv
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
import minorminer

solver_limit = 50

G = nx.complete_graph(solver_limit)
#system = DWaveSampler()
#embedding = minorminer.find_embedding(G.edges, system.edgelist)

trueCount=0
timeSum = 0
time1 = 0
for l in range(0, 11):
    print('datetime:',  datetime.datetime.today())
    time1 = time.time()
    #response = QBSolv().sample_qubo(Q, solver=FixedEmbeddingComposite(system, embedding), solver_limit=solver_limit) #When using actual D-wave machine
    response = QBSolv().sample_qubo(Q, solver_limit=solver_limit)
    answers = list(response.samples())
    answer=[]
    for i in range(len(answers[0])):
        if answers[0][i]==1:answer.append(i)
    print(answer)
    for i in range(len(answer)):
        print(qubo_prb[answer[i]][0], prb[qubo_prb[answer[i]][0]][qubo_prb[answer[i]][1]][qubo_prb[answer[i]][2]])
    check_r=check_answer(answer)
    if check_r: trueCount+=1
    print("check result:",check_r)
    cplot(answer, prb, "output_tetro_en/" + str(l) + ".png")
    print('datetime:',  datetime.datetime.today())
    timeSum += (time.time()-time1)

print("timeSum:", timeSum)
print(l+1)
print("timeAve:", timeSum/(l+1))
