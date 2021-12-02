#%%
import numpy as np
from numpy.core.numeric import NaN
import torch
import time
import math
from utils.utils import communication_cost
from utils.datagenerate import generate_distance_matrix
from torch_geometric.loader import DataLoader
from numba import njit
import sys
rng = np.random.default_rng()
#%%

def global_best(particle, prtl_fitness):
    min_id = np.argmin(prtl_fitness)
    gbest = particle[min_id]
    gbest_fit = prtl_fitness[min_id]
    return gbest, gbest_fit
#%%
@njit
def rand_permute2D(particle):
    for i in range(particle.shape[0]):
        particle[i] = np.random.permutation(particle.shape[1])
#%%
# funtion for particle intializaton , fitness evaluation, global_best, local_best
def prtl_init(prtl_no, prtl_size):
    particle = np.zeros((prtl_no, prtl_size), dtype=np.long)
    rand_permute2D(particle)
    prtl_fitness = communication_cost(data.edge_index,data.edge_attr,data.batch, prtl_no, distance_matrix, torch.from_numpy(particle))
    prtl_fitness = prtl_fitness.detach().numpy()
    return particle, prtl_fitness
#%%

# finding best and bad particles
def best_bad(prtl, prtl_fit):
    l = prtl.shape[0]
    bstp_no = l // 2
    partition_ids = np.argpartition(prtl_fit, -bstp_no)
    # ids of bad particles
    badp_ids = partition_ids[-bstp_no:]
    # ids of best particles
    bstp_ids = partition_ids[:-bstp_no]
    return bstp_ids, badp_ids
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
    if len(sys.argv) != 3:
        print("Usage: python3 dpso.py <dataset> <no_of_particles>")
        sys.exit(1)
    device = torch.device("cpu")
    single_graph_data = torch.load(sys.argv[1]).to(device)
    prtl_no = int(sys.argv[2])
    graph_size = single_graph_data.num_nodes
    n = math.ceil(math.sqrt(graph_size))
    m = math.ceil(graph_size/n)
    datalist = [single_graph_data for _ in range(prtl_no)]
    # just for using communication_cost function
    dataloader = DataLoader(datalist, batch_size=prtl_no)
    # just for using communication_cost function
    dataloader = DataLoader(datalist, batch_size=prtl_no)
    data = next(iter(dataloader))
    distance_matrix = generate_distance_matrix(n,m).to(device)
    no_itera = 10000
    prtl_size = graph_size
    gbest_list = []
    check = 0
    # paricle intialization
    prtl, prtl_fit = prtl_init(prtl_no, prtl_size)
    lcl_bst_prtl = np.copy(prtl)
    lcl_bst_fit = np.copy(prtl_fit)
    gbest, gbfit = global_best(prtl, prtl_fit)
    gbest_list.append((gbest, gbfit))
    for j in range(no_itera):
        bstp_ids, badp_ids = best_bad(prtl, prtl_fit)
        bst_prtl = prtl[bstp_ids]
        bad_prtl = prtl[badp_ids]
        rand_loc = rng.permutation(prtl_size)
        swap_limit_good = math.ceil(prtl_size / 4)
        swap_limit_bad = math.ceil(prtl_size / 2)
        lbest_loc = rand_loc[0:swap_limit_good // 2]  # no of locations for
        gbest_loc = rand_loc[swap_limit_good // 2:swap_limit_good]
        bad_loc1 = rand_loc[0:swap_limit_bad // 2]
        bad_loc2 = rand_loc[swap_limit_bad // 2:swap_limit_bad]
        # next generation of good particles
        evolve_particles(lbest_loc, gbest_loc, bst_prtl, lcl_bst_prtl, gbest)
        # next generation of bad particles
        rand_permute2D(bad_prtl)
        evolve_particles(bad_loc1, bad_loc2, bad_prtl, lcl_bst_prtl, gbest)
        # calculate global and local best for this generation
        prtl = np.vstack((bst_prtl, bad_prtl))
        prtl_fit = communication_cost(data.edge_index,data.edge_attr,data.batch, prtl_no, distance_matrix, torch.from_numpy(prtl))
        prtl_fit = prtl_fit.detach().numpy()
        # update lcl_bst_prtl and lcl_bst_fit
        all_ids = np.concatenate((bstp_ids, badp_ids))
        update_condition = prtl_fit < lcl_bst_fit[all_ids]
        lcl_bst_prtl = np.where(update_condition.reshape(prtl_no, 1), prtl, lcl_bst_prtl[all_ids, :])
        lcl_bst_fit = np.where(update_condition, prtl_fit, lcl_bst_fit[all_ids])
        # update gbest and gbfit for this generation
        gbest, gbfit = global_best(prtl, prtl_fit)

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
