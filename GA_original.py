#%%
import numpy as np
import torch
import math
from utils.utils import communication_cost_multiple_samples
from utils.datagenerate import generate_distance_matrix
from torch_geometric.loader import DataLoader
from numba import njit
from models.mpnn_ptr import MpnnPtr
from timeit import default_timer as timer
import random
import argparse
import sys
from joblib import Parallel, delayed
import multiprocessing
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

def repair(child):
    # child_n = np.empty([childs.shape[0],childs.shape[1]])
    child_n = [elem if elem not in child[:i] else 'x' for i, elem in enumerate(child)]
    missing_elemnts = [item for item in range(len(child)) if item not in child_n]
    random.shuffle(missing_elemnts) 
    
    k = 0
    for j,item in enumerate(child_n):
        if item == 'x':
            child_n[j] = missing_elemnts[k]
            k             = k+1
        
    return child_n
#%%
class GA:
    def __init__(self, dataset_path, num_particles,\
        max_iterations, prtl_init_method="random", model_path=None):
        device = torch.device("cpu")
        single_graph_data = torch.load(dataset_path, map_location=device)
        graph_size = single_graph_data.num_nodes
        n = math.ceil(math.sqrt(graph_size))
        m = math.ceil(graph_size/n)
        datalist = [single_graph_data]
        # just for using communication_cost function
        dataloader = DataLoader(datalist, batch_size=1)
        self.data = next(iter(dataloader))
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
        num_sampled_particles = int(num_particles * .2)
        num_random_particles = num_particles - num_sampled_particles
        mpnn_ptr = MpnnPtr(input_dim=self.particle_size, embedding_dim=self.particle_size + 10, hidden_dim=self.particle_size + 20, K=3, n_layers=2, p_dropout=0.1, device=self.device, logit_clipping=True)
        if self.model_path is None:
            raise ValueError('model_path is None')
        mpnn_ptr.load_state_dict(torch.load(self.model_path))
        mpnn_ptr.eval()
        with torch.no_grad():
            # generate initial population
            particle_sampled, _ = mpnn_ptr(self.data, 100000)
        particle_sampled = np.unique(particle_sampled.detach().numpy(),axis=0)
        particle_sampled_fit = self.comm_cost(particle_sampled)
        indices = particle_sampled_fit.argpartition(num_sampled_particles)[:num_sampled_particles]
        particle_first_half = particle_sampled[indices]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(f'{np.unique(particle_first_half,axis=0)}')
        particle_second_half = self.prtl_init(num_random_particles)
        particle = np.concatenate((particle_first_half, particle_second_half), axis=0)
        return particle
        
    def mutated_genes(self):
        '''
        create random genes for mutation
        '''
        GENES = range(self.particle_size)
        gene = random.choice(GENES)
        return gene        
        
        

    def mate(self, par1,par2):
        # '''
        # Perform mating and produce new offspring
        # '''
        # chromosome for offspring
        child_chromosome = []
        for gp1, gp2 in zip(par1, par2):   
            # random probability 
            prob = random.random()
            # if prob is less than 0.45, insert gene
            # from parent 1
            if prob < 0.45:
                child_chromosome.append(gp1)
            # if prob is between 0.45 and 0.90, insert
            # gene from parent 2
            elif prob < 0.90:
                child_chromosome.append(gp2)
            # otherwise insert random gene(mutate),
            # for maintaining diversity
            else:
                child_chromosome.append(self.mutated_genes())
        # create new Individual(offspring) using
        # generated chromosome for offspring
        return repair(child_chromosome)

    def run(self):
        if self.prtl_init_method == "random":
            prtls = self.prtl_init(self.num_particles)
        elif self.prtl_init_method == "model":
            prtls = self.prtl_init_model(self.num_particles)
        else:
            raise ValueError(f'{self.prtl_init_method} is not supported for particle initialization')
        prtls_fit = self.comm_cost(np.array(prtls))
        gbest, gbfit = self.global_best(prtls, prtls_fit)
        gb_fits = [gbfit]
        gb_prtls = [gbest]
        check = 0
        for j in range(self.max_iterations):
            # np.set_printoptions(threshold=sys.maxsize)
            # next generation of good particles
            new_generation = []
            indices = prtls_fit.argsort()
            new_prtls = np.array(prtls)[indices]
            new_prtls_fit = prtls_fit[indices]
            s = int((10*self.num_particles)/100)
            new_generation.extend(prtls[:s])
            s = int((90*self.num_particles)/100)
            # st1_time = timer()
            for _ in range(s):
                parent1 = random.choice(new_prtls[:50])
                parent2 = random.choice(new_prtls[:50])
                child   = self.mate(parent1,parent2)
                new_generation.append(child)
            # num_cores = multiprocessing.cpu_count()
            # results = Parallel(n_jobs=num_cores)(delayed(self.mate)(random.choice(new_prtls[:50]),random.choice(new_prtls[:50])) for i in range(s))
            # get_ranks = multiprocessing.Pool()

            # answer = get_ranks.map(self.mate,range(s))
            # end_time = timer()
            # print(f'mate exec time:{end_time-st1_time}')
            prtls = new_generation
            prtls_fit = self.comm_cost(np.array(prtls))
            # calculate global and local best for this generation
            gbest, gbfit = self.global_best(prtls, prtls_fit)
            if (gb_fits[-1] == gbfit):
                check += 1
                if (check >= 500):
                    print(f'global best not improved for {check} generations \n exit')
                    break
            else:
                check = 0
            if (gb_fits[-1] > gbfit):
                gb_fits.append(gbfit)
                gb_prtls.append(gbest)
            if (j % 10 == 0):
                print(f'iteration: {j} gbest fitness {gb_fits[-1]}')
        gb_fits = np.array(gb_fits)
        min_fit_idx = np.argmin(gb_fits)
        # print(torch.tensor(gb_fits))
        return gb_prtls[min_fit_idx], gb_fits[min_fit_idx]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="path to the data file", type=str)
    parser.add_argument('-p','--num_prtl', help="number of particles in each generation", type=int, default=100)
    parser.add_argument('-i','--max_iter', help="max number of iterations", type=int, default=1000)
    parser.add_argument('-m','--model', help="path to the model file for initial population generation", type=str, default=None)
    args = parser.parse_args()
    prtl_init_method = "random" if args.model is None else "model"
    ga = GA(args.dataset, args.num_prtl, args.max_iter, prtl_init_method, args.model)
    # st_time = timer() 
    # best_prtl,cost = ga.run()
    # end_time = timer()
    # print(f'best particle {best_prtl}')
    # print(f'best cost {cost}')
    # print(f'execution time {end_time-st_time}secs')
    
    # measure time taken and best cost by running the algorithm 5 times
    # best_cost = float('inf')
    # best_time = float('inf')
    for i in range(5):
        print(f'iteration: {i} started \n')
        start_time = timer()
        best_prtl, cost = ga.run()
        end_time = timer()
        time_taken = end_time - start_time
        file1 = open("results/GA_geek4geek_"+args.dataset[5:-3]+'.csv','a')
        file1.write(args.dataset[5:-3]+'_'+str(i)+","+str(cost)+","+str(time_taken)+'\n')    
    # print(f'Time taken to run the algorithm: {best_time}')
    # print(f'Best cost: {best_cost}')
# %%

