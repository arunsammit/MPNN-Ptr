import numpy as np
import pandas as pd
import random
import subprocess as sp
import os
import time
import matplotlib.pyplot as plt
from numba import jit
# total communication cost of the mapping

def comm_cost(mapp):
    T = 0
    prtl = list(mapp)
    for i in list(traff_dict.keys()):
        p1 = prtl.index(traff_dict[i]['src'])
        p2 = prtl.index(traff_dict[i]['dst'])
        x = nw_dict[str(p1)]['cor']
        y = nw_dict[str(p2)]['cor']
        T += traff_dict[i]['val']*(abs(x[0]-y[0])+abs(x[1]-y[1]))      

    return T



# global best finding
# @jit(nopython=True)
def global_best(particle,prtl_fitness):
    gbest  = np.zeros(particle.shape[1])
    min_id = np.where(prtl_fitness == prtl_fitness.min())
    gbest = particle[min_id[0][0]]
    gbest_fit = prtl_fitness[min_id[0][0]]

#     print('global best fitness =%f , g_best_prtl=%s'%(gbest_fit,gbest))
    return gbest,gbest_fit

# funtion for particle intializaton , fitness evaluation, global_best, local_best
# @jit(nopython=True)
def prtl_init(prtl_no,prtl_size):
    particle = np.zeros((prtl_no,prtl_size))
    prtl_fitness = np.zeros(prtl_no)
    prtl_lbest   = np.zeros((prtl_no,prtl_size))
    prtl_lbfit = np.zeros(prtl_no)
    for i in range(prtl_no):
        particle[i] = random.sample(range(16),16)
        prtl_fitness[i] = comm_cost(particle[i])
        prtl_lbest[i]   =  particle[i]
        prtl_lbfit[i]   =  prtl_fitness[i]
    return particle,prtl_fitness,prtl_lbest,prtl_lbfit

@jit(nopython=True)
def swap_seq(prtl,lrgbest):
    ss = []
    for i in range(len(prtl)):
        if(prtl[i] != lrgbest[i]):
            pos = np.where(lrgbest[i] == prtl)[0][0]
            ss.append((i,pos))
            tmp   = prtl[pos]
            prtl[pos] = prtl[i]
            prtl[i]   = tmp
    return ss

# @jit(nopython=True)
def prtl_upd(prtl,SwapSeq):
    for ss in SwapSeq:
        if(random.random() > 0.5): # apply swap operation with prob 0.5
            prtl[ss[0]],prtl[ss[1]] = prtl[ss[1]],prtl[ss[0]]
    return prtl

# @jit(nopython=True)
def pso_algo(prtl_no,no_iter,prtl_size):
    prtl,pfit,plbest,plbest_fit = prtl_init(prtl_no,prtl_size)   #particle intialization
    gbest,gbfit = global_best(prtl,pfit)              # global best finding
    gbest_list = []
    gbest_list.append((gbest,gbfit))
    ss_g = []
    ss_l = []
    check = 0
    for i in range(no_iter):
        for j in range(prtl_no):
            ss_g = swap_seq(prtl[j],gbest)
            ss_l = swap_seq(prtl[j],plbest[j])
            prtl[j] = prtl_upd(prtl[j],np.array(ss_l))  # update prtl w.r.t local best
            prtl[j] = prtl_upd(prtl[j],np.array(ss_l))  # update prtl w.r.t global best
        
        #update local best and global best for current generation
        for k in range(prtl_no):
            pfit[k] = comm_cost(prtl[k])
            if(pfit[k] > pfit[k]):
                plbest[k] = prtl[k]
                plbest_fit[k] = pfit[k]
        gbest_tmp, gbfit_tmp  = global_best(prtl,pfit)
        if(gbfit > gbfit_tmp):
            gbest = gbest_tmp
            gbfit = gbfit_tmp
            
            check = 0
        gbest_list.append((gbest,gbfit))
        if(i%10==0):
            print('gbest = %d after iteration = %d'%(gbfit,i))

        if(gbest_list[-1][1] == gbfit):
            check += 1
            if(check >= 1000):
                print('global best not improved for %d generations \n exit'%check)
                break
        
        
        # print('check=',check)
    return plbest,gbest_list





traffic_file = open('traffic.txt','r+')
network_file = open('network_16.txt','r+')
# # prepare a dictionary of links with their BW,src and dst
# # below dictionay we made it for VOPD application but same code can create any application
# # information about all the links and their corresponding BW, src core and dst core

traff_dict = {}
for idx,line in enumerate(traffic_file):
    for pos,word in enumerate(line.split()):
        if(word.isnumeric() and pos != idx):
            traff_dict['l%st%s'%(str(idx),str(pos))] = {'val':int(word),'src':idx,'dst':pos}


# print(traff_dict)

# prepare a dictionay to store (x,y)coordinates of the each router :: topology of the NoC (MESH)
#*************************************************************************************************
nw_dict = {}
mesh_size = [4,4]
cords = []
for x in range(mesh_size[0]):
    for y in range(mesh_size[1]):
        cords.append((y,x))
for idx,line in enumerate(network_file):
    nw_dict[str(idx)] = {'cor':cords[idx]}
# print(cords)


# DPSO mapping optimization using communication cost
print('******* mapping with comm cost as metric has started ************')

# DPSO algorithm
prtl_no = 500
no_iter  = 100
prtl_size = 16

start_time = time.time()
plbest,gbest_list =   pso_algo(prtl_no,no_iter,prtl_size)  
end_time = time.time()
print('optimization time in secs=%f'%(end_time-start_time))



print('global best particle is',gbest_list[-1][0])
print('global best prtl fitness',gbest_list[-1][1])

print('******* mapping with comm cost as metric is ended ************')

# plot
fig, ax = plt.subplots()
lst = [i[1] for i in gbest_list]
ax.plot(lst, linewidth=2.0)

plt.xlabel('Generations')
plt.ylabel('Communication cost')


plt.show()

