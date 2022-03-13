#%%
import numpy as np
import torch
import math
from utils.utils import communication_cost_multiple_samples
from utils.datagenerate import generate_distance_matrix, generate_distance_matrix_3D
from torch_geometric.loader import DataLoader
from numba import njit
from models.mpnn_ptr import MpnnPtr
from timeit import default_timer as timer
import random
import argparse
import sys
#%%
@njit
def rand_permute2D(particle):
    for i in range(particle.shape[0]):
        particle[i] = np.random.permutation(particle.shape[1])
        # print(particle[i])
@njit
def crossover(prtls):
    """
    Crossover between two parents to generate two children
    """
    childs = np.zeros((prtls.shape[0], prtls.shape[1]), dtype=np.int64)
    partner_idxs = np.random.permutation(prtls.shape[0]//2) + prtls.shape[0]//2
    for i in range(prtls.shape[0]//2):
        # randomly choose a partner
        partner_idx = partner_idxs[i]
        # randomly choose a cut point
        cut_point = random.randint(1, prtls.shape[1] - 1)
        childs[2*i, :cut_point] = prtls[i, :cut_point]
        childs[2*i, cut_point:] = prtls[partner_idx, cut_point:]
        childs[2*i+1, :cut_point] = prtls[partner_idx, :cut_point]
        childs[2*i+1, cut_point:] = prtls[i, cut_point:]
    # remove the dublicate genes
    for i in range(childs.shape[0]):
        gene_seen = np.zeros((childs.shape[1],), dtype=np.int64)
        for j in range(childs.shape[1]):
            curr_gene = childs[i, j]
            if gene_seen[curr_gene] == 1:
                childs[i, j] = -1
            gene_seen[curr_gene] = 1
    # assign the missing genes until all the values from 0 to n-1 are present in the child
    for i in range(childs.shape[0]):
        gene_present = np.zeros((childs.shape[1],), dtype=np.int64)
        for j in range(childs.shape[1]):
            if childs[i, j] != -1:
                gene_present[childs[i, j]] = 1
        genes_absent = np.flatnonzero(gene_present == 0)
        # randomly shuffle genes_not_seen
        np.random.shuffle(genes_absent)
        k = 0
        for j in range(childs.shape[1]):
            if childs[i, j] == -1:
                childs[i, j] = genes_absent[k]
                k += 1
    return childs
@njit
def mutation(prtls):
    """
    Mutate the particles by swapping the contents of two geners creating new individuals
    """
    childs = np.copy(prtls)
    for i in range(prtls.shape[0]):
        # randomly choose two locations in the particles to perform swapping
        locs = np.random.randint(0, prtls.shape[1], size=2)
        childs[i, locs[0]], childs[i, locs[1]] = childs[i, locs[1]], childs[i, locs[0]]
    return childs
#%%
class GA:
    def __init__(self, dataset_path, num_particles,\
        max_iterations, prtl_init_method="random", model_path=None, use_3d=False):
        device = torch.device("cpu")
        single_graph_data = torch.load(dataset_path, map_location=device)
        graph_size = single_graph_data.num_nodes
        datalist = [single_graph_data]
        # just for using communication_cost function
        dataloader = DataLoader(datalist, batch_size=1)
        self.data = next(iter(dataloader))
        if use_3d:
            print("Using 3D distance matrix")
            n = math.ceil(math.sqrt(graph_size // 2))
            m = math.ceil(graph_size // (2 * n))
            l = 2
            self.distance_matrix = generate_distance_matrix_3D(n, m, l).to(device)
        else: 
            print("Using 2D distance matrix")
            n = math.ceil(math.sqrt(graph_size))
            m = math.ceil(graph_size/n)
            self.distance_matrix = generate_distance_matrix(n,m).to(device)
        self.max_iterations = max_iterations
        self.num_particles = num_particles
        self.particle_size = graph_size
        self.prtl_init_method = prtl_init_method
        self.model_path = model_path
        self.device = device
    def comm_cost(self, particle):
        return communication_cost_multiple_samples(self.data.edge_index,self.data.edge_attr,self.data.batch, self.distance_matrix, torch.from_numpy(particle), particle.shape[0]).detach().numpy()
    def prtl_init(self, num_particles):
        particle = np.zeros((num_particles, self.particle_size), dtype=np.int64)
        rand_permute2D(particle)
        return particle
    def global_best(self, particle, prtl_fitness):
        min_id = np.argmin(prtl_fitness)
        # print(prtl_fitness)
        # print(min_id)
        gbest = particle[min_id]
        gbest_fit = prtl_fitness[min_id]
        return gbest, gbest_fit
    def prtl_init_model(self, num_particles):
        # load model
        # TODO: generate half population from model and half randomly
        num_sampled_particles = int(num_particles * .8)
        num_random_particles = num_particles - num_sampled_particles
        mpnn_ptr = MpnnPtr(input_dim=self.particle_size, embedding_dim=self.particle_size + 10, hidden_dim=self.particle_size + 20, K=3, n_layers=2, p_dropout=0.1, device=self.device, logit_clipping=True, decoding_type="greedy",feature_scale=290)
        if self.model_path is None:
            raise ValueError('model_path is None')
        mpnn_ptr.load_state_dict(torch.load(self.model_path))
        mpnn_ptr.eval()
        with torch.no_grad():
            # generate initial population
            particle_sampled, _ = mpnn_ptr(self.data, 10000)
        particle_sampled = particle_sampled.cpu().numpy()
        particle_sampled_fit = self.comm_cost(particle_sampled)
        indices = particle_sampled_fit.argpartition(num_sampled_particles)[:num_sampled_particles]
        particle_first_half = particle_sampled[indices]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(f'{np.unique(particle_first_half,axis=0)}')
        particle_second_half = self.prtl_init(num_random_particles)
        particle = np.concatenate((particle_first_half, particle_second_half), axis=0)
        return particle
    def run(self):
        if self.prtl_init_method == "random":
            prtl = self.prtl_init(self.num_particles)
        elif self.prtl_init_method == "model":
            prtl = self.prtl_init_model(self.num_particles)
        else:
            raise ValueError(f'{self.prtl_init_method} is not supported for particle initialization')
        prtl_fit = self.comm_cost(prtl)
        gbest, gbfit = self.global_best(prtl, prtl_fit)
        gb_fits = [gbfit]
        gb_prtls = [gbest]
        check = 0
        for j in range(self.max_iterations):
            # np.set_printoptions(threshold=sys.maxsize)
            # next generation of good particles
            childs1 = crossover(prtl)
            childs2 = mutation(prtl)
            childs = np.concatenate((childs1, childs2), axis=0)
            # print(f'childs:\n{childs}')
            childs_fit = self.comm_cost(childs)
            all_prtls = np.concatenate((prtl, childs), axis=0)
            all_prtls_fit = np.concatenate((prtl_fit, childs_fit), axis=0)
            # select the top num_particles particles
            indices = all_prtls_fit.argpartition(self.num_particles)[:self.num_particles]
            prtl = all_prtls[indices]
            prtl_fit = all_prtls_fit[indices]
            # calculate global and local best for this generation
            gbest, gbfit = self.global_best(prtl, prtl_fit)
            if (j % 10 == 0):
                print(f'iteration: {j} gbest fitness {gbfit}')
            if (gb_fits[-1] <= gbfit):
                check += 1
                if (check >= 500):
                    print(f'global best not improved for {check} generations \nexit')
                    break
            else:
                check = 0
            gb_fits.append(gbfit)
            gb_prtls.append(gbest)
        gb_fits = np.array(gb_fits)
        min_fit_idx = np.argmin(gb_fits)
        return gb_prtls[min_fit_idx], gb_fits[min_fit_idx]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="path to the data file", type=str)
    parser.add_argument('-p','--num_prtl', help="number of particles in each generation", type=int, default=100)
    parser.add_argument('-i','--max_iter', help="max number of iterations", type=int, default=1000)
    parser.add_argument('-m','--model', help="path to the model file for initial population generation", type=str, default=None)
    parser.add_argument('--three_D', help='use a fully connected 3D NoC with 2 layers in the Z direction', action='store_true')
    args = parser.parse_args()
    prtl_init_method = "random" if args.model is None else "model"
    ga = GA(args.dataset, args.num_prtl, args.max_iter, prtl_init_method, args.model, args.three_D)
    best_prtl,cost = ga.run()
    print(f'best particle\n')
    print(f'{torch.tensor(best_prtl)}')
    print(f'best cost {cost}')
    # measure time taken and best cost by running the algorithm 5 times
    # best_cost = float('inf')
    # best_time = float('inf')
    # for i in range(5):
    #     start_time = timer()
    #     best_prtl, cost = dpso.run()
    #     end_time = timer()
    #     time_taken = end_time - start_time
    #     np.set_printoptions(threshold=sys.maxsize)
    #     print(f'i = {i}: best cost {cost} best particle {best_prtl} in {time_taken} seconds')
    #     if cost < best_cost:
    #         best_cost = cost
    #     if time_taken < best_time:
    #         best_time = time_taken
    # print(f'Time taken to run the algorithm: {best_time}')
    # print(f'Best cost: {best_cost}')
# command to run the code for initial population generation algorithm
# python3 ga_tosun.py data_tgff/single/traffic_32.pt --num_prtl 1024 --max_iter 5000 --model models_data_final/model_16_01-10.pt
# command to run the code for 3D NoC
# python3 ga_tosun.py data_tgff/single/traffic_32.pt --num_prtl 1024 --max_iter 5000 --three_D