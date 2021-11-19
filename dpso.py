import numpy as np
import torch
import random
import time
import math
from utils.utils import communication_cost
from utils.datagenerate import generate_distance_matrix
from torch_geometric.loader import DataLoader
from numpy.random import default_rng
rng = default_rng()
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
single_graph_data = torch.load('data/data_single_instance_49.pt')
graph_size = single_graph_data.num_nodes
n = math.ceil(math.sqrt(graph_size))
m = math.ceil(graph_size/n)
batch_size = 100
#%%
datalist = [single_graph_data for _ in range(batch_size)]
# just for using communication_cost function
dataloader = DataLoader(datalist, batch_size=batch_size)
#%%datalist = [single_graph_data for _ in range(batch_size)]
# just for using communication_cost function
dataloader = DataLoader(datalist, batch_size=batch_size)
data = next(iter(dataloader))
#%%
distance_matrix = generate_distance_matrix(7,7).to(device)
#%%

def global_best(particle, prtl_fitness):
    gbest = np.zeros(particle.shape[1])
    min_id = np.where(prtl_fitness == prtl_fitness.min())
    gbest = particle[min_id[0][0]]
    gbest_fit = prtl_fitness[min_id[0][0]]

    #     print('global best fitness =%f , g_best_prtl=%s'%(gbest_fit,gbest))
    return gbest, gbest_fit

#%%
# funtion for particle intializaton , fitness evaluation, global_best, local_best
def prtl_init(prtl_no, prtl_size):
    particle = np.zeros((prtl_no, prtl_size), dtype=np.long)
    for i in range(prtl_no):
        particle[i] = rng.permutation(prtl_size)

    prtl_fitness = communication_cost(data.edge_index,data.edge_attr,data.batch, prtl_no, distance_matrix, torch.from_numpy(particle))
    prtl_fitness = prtl_fitness.detach().numpy()
    prtl_lbest = np.copy(particle)

    return particle, prtl_fitness, prtl_lbest
#%%

# finding best and bad particles
def best_bad(prtl, prtl_fit):
    l = prtl.shape[0]
    bstp_no = int(l / 2)
    badp_no = l - bstp_no
    sort = prtl_fit.argsort()
    prtl_sort = prtl[sort]
    pfit_sort = prtl_fit[sort]
    best_prtl = prtl_sort[0:bstp_no, :]
    best_fit = pfit_sort[0:bstp_no]
    bad_prtl = prtl_sort[bstp_no:, :]
    bad_fit = pfit_sort[bstp_no:]
    return best_prtl, best_fit, bad_prtl, bad_fit


# DPSO algorithm
# %%
if __name__ == "__main__":
    prtl_no = batch_size
    no_itera = 1000
    prtl_size = graph_size
    gbest_list = []
    check = 0
    # paricle intialization
    prtl, pfit, plbest = prtl_init(prtl_no, prtl_size)
    gbest, gbfit = global_best(prtl, pfit)
    gbest_list.append((gbest, gbfit))
    print('gbest', gbest)

    for j in range(no_itera):
        best_prtl, best_fit, bad_prtl, bad_fit = best_bad(prtl, pfit)
        rand_loc = random.sample(range(prtl_size), prtl_size)
        swap_limit_good = math.ceil(prtl_size / 4)
        swap_limit_bad = math.ceil(prtl_size / 2)
        lbest_loc = rand_loc[0:swap_limit_good // 2]  # no of locations for
        gbest_loc = rand_loc[swap_limit_good // 2:swap_limit_good]
        bad_loc1 = rand_loc[0:swap_limit_bad // 2]
        bad_loc2 = rand_loc[swap_limit_bad // 2:swap_limit_bad]
        # next generation of good particles
        for i in range(best_prtl.shape[0]):
            for val in lbest_loc:
                min_id = np.where(best_prtl[i] == plbest[i][val])
                if (best_prtl[i][val] != plbest[i][val]):
                    best_prtl[i][min_id[0]] = best_prtl[i][val]
                    best_prtl[i][val] = plbest[i][val]
            for vgl in gbest_loc:
                min_id = np.where(best_prtl[i] == gbest[vgl])
                if (best_prtl[i][vgl] != gbest[vgl]):
                    best_prtl[i][min_id[0]] = best_prtl[i][vgl]
                    best_prtl[i][vgl] = gbest[vgl]

        # next generation of bad particles
        for i in range(bad_prtl.shape[0]):
            temp = bad_prtl[i]
            np.random.shuffle(temp)
            bad_prtl[i] = temp
            for val in bad_loc1:

                min_id = np.where(bad_prtl[i] == plbest[i][val])
                if (bad_prtl[i][val] != plbest[i][val]):
                    bad_prtl[i][min_id[0]] = bad_prtl[i][val]
                    bad_prtl[i][val] = plbest[i][val]
            for vgl in bad_loc2:
                min_id = np.where(bad_prtl[i] == gbest[vgl])
                if (bad_prtl[i][vgl] != gbest[vgl]):
                    bad_prtl[i][min_id[0]] = bad_prtl[i][vgl]
                    bad_prtl[i][vgl] = gbest[vgl]

        # calculate global and local best for this generation
        prtl = np.vstack((best_prtl, bad_prtl))
        pfit = communication_cost(data.edge_index,data.edge_attr,data.batch, prtl_no, distance_matrix, torch.from_numpy(prtl))
        pfit = pfit.detach().numpy()
        for num in range(prtl.shape[0]):
            fit = pfit[num]
            if (fit < pfit[num]):
                plbest[num] = prtl[num]
                pfit[num] = fit

        gbest, gbfit = global_best(prtl, pfit)

        if (j % 10 == 0):
            print('gbest = %d after iteration = %d' % (gbfit, j))

        if (gbest_list[-1][1] == gbfit):
            check += 1
            if (check >= 200):
                print('global best not improved for %d generations \n exit' % check)
                break
        else:
            check = 0

        gbest_list.append((gbest, gbfit))
    end_time = time.time()
    gb_fits = np.array([gbfit for _, gbfit in gbest_list])
    print(f'best cost = {gb_fits.min()}')
