#%%
import numpy as np
from numpy.core.numeric import NaN
import torch
import time
import math

from torch._C import NoneType
from utils.utils import communication_cost
from utils.datagenerate import generate_distance_matrix
from torch_geometric.loader import DataLoader
from numpy.random import default_rng
from numba import njit
rng = default_rng()

#%%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
single_graph_data = torch.load('data/data_single_instance_64.pt').to(device)
graph_size = single_graph_data.num_nodes
print(graph_size)
n = math.ceil(math.sqrt(graph_size))
m = math.ceil(graph_size/n)
batch_size = 128
datalist = [single_graph_data for _ in range(batch_size)]
# just for using communication_cost function
dataloader = DataLoader(datalist, batch_size=batch_size)
# just for using communication_cost function
dataloader = DataLoader(datalist, batch_size=batch_size)
data = next(iter(dataloader))
distance_matrix = generate_distance_matrix(n,m).to(device)
#%%

def global_best(particle, prtl_fitness):
    min_id = np.argmin(prtl_fitness)
    gbest = particle[min_id]
    gbest_fit = prtl_fitness[min_id]
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
    bstp_no = l // 2
    partition_ids = np.argpartition(prtl_fit, -bstp_no)
    # ids of best particles
    bstp_ids = partition_ids[-bstp_no:]
    # ids of bad particles
    badp_ids = partition_ids[:-bstp_no]
    best_partl = prtl[bstp_ids]
    bad_partl = prtl[badp_ids]
    # fitness of best particles
    best_fit = prtl_fit[bstp_ids]
    # fitness of bad particles
    bad_fit = prtl_fit[badp_ids]
    return best_partl, best_fit, bad_partl, bad_fit
#%%
@njit
def transform(idxs, curr_prtl, trns_to_prtl):
    """
    Transforms curr_prtl such that it equal to trns_to_prtl at indices given by idxs
    """
    for val in idxs:
        item = trns_to_prtl[val]
        if (curr_prtl[val] != item):
            index = 0
            for i in range(curr_prtl.shape[0]):
                if curr_prtl[i] == item:
                    index = i
            curr_prtl[index] = curr_prtl[val]
            curr_prtl[val] = item
#%%
@njit
def evolve_particles(lcl_locs, gbest_locs, prtls, lcl_bst_prtl, gbest):
    """
    Evolves particles using lcl_locs for lcl_bst_prtl and gbest_locs for gbest
    """
    for i in  range(prtls.shape[0]):
        curr_prtl = prtls[i]
        curr_lcl_bst_prtl = lcl_bst_prtl[i]
        transform(lcl_locs, curr_prtl, curr_lcl_bst_prtl)
        transform(gbest_locs, curr_prtl, gbest)
#%%
# DPSO algorithm
if __name__ == "__main__":
    prtl_no = batch_size
    no_itera = 5000
    prtl_size = graph_size
    gbest_list = []
    check = 0
    # paricle intialization
    prtl, pfit, lcl_bst_prtl = prtl_init(prtl_no, prtl_size)
    gbest, gbfit = global_best(prtl, pfit)
    gbest_list.append((gbest, gbfit))
    print('gbest', gbest)

    for j in range(no_itera):
        bst_prtl, best_fit, bad_prtl, bad_fit = best_bad(prtl, pfit)
        rand_loc = rng.permutation(prtl_size)
        swap_limit_good = math.ceil(prtl_size / 4)
        swap_limit_bad = math.ceil(prtl_size / 2)
        lbest_loc = rand_loc[0:swap_limit_good // 2]  # no of locations for
        gbest_loc = rand_loc[swap_limit_good // 2:swap_limit_good]
        bad_loc1 = rand_loc[0:swap_limit_bad // 2]
        bad_loc2 = rand_loc[swap_limit_bad // 2:swap_limit_bad]
        # next generation of good particles
        for i in range(bst_prtl.shape[0]):
            curr_prtl = bst_prtl[i]
            curr_lcl_bst_prtl = lcl_bst_prtl[i]
            transform(lbest_loc, curr_prtl, curr_lcl_bst_prtl)
            transform(gbest_loc, curr_prtl, gbest)

        # next generation of bad particles
        for i in range(bad_prtl.shape[0]):
            curr_prtl = bad_prtl[i]
            np.random.shuffle(curr_prtl)
            curr_lcl_bst_prtl = lcl_bst_prtl[i]
            transform(bad_loc1, curr_prtl, curr_lcl_bst_prtl)
            transform(bad_loc2, curr_prtl, gbest)

        # calculate global and local best for this generation
        prtl = np.vstack((bst_prtl, bad_prtl))
        pfit = communication_cost(data.edge_index,data.edge_attr,data.batch, prtl_no, distance_matrix, torch.from_numpy(prtl))
        pfit = pfit.detach().numpy()
        for num in range(prtl.shape[0]):
            fit = pfit[num]
            if (fit < pfit[num]):
                lcl_bst_prtl[num] = prtl[num]
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

# %%
