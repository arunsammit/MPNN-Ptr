from typing import Tuple
from utils.baseline import calculate_baseline
from utils.utils import communication_cost, communication_cost_multiple_samples
from utils.baseline import paired_t_test
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch import Tensor
import copy
import torch
class Trainer:
    def __init__(self, model):
        self.device = model.device
        self.model = model
    def train_step(self, data:Data, distance_matrix)->Tuple[Tensor, float]:
        raise NotImplementedError
    def train(self, dataloader:DataLoader, distance_matrix_dict, optimizer):
        self.model.train()
        epoch_loss = 0
        for data in dataloader:
            data = data.to(self.device)
            optimizer.zero_grad()
            # print(f"devices are: {data.x.device} and {data.edge_index.device} and {data.edge_attr.device} and {data.batch.device}")
            distance_matrix = distance_matrix_dict[data.num_nodes // data.num_graphs].to(self.device)
            loss, comm_cost_sum = self.train_step(data, distance_matrix)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            epoch_loss += comm_cost_sum
        return epoch_loss / len(dataloader.dataset)
class TrainerInitPop(Trainer):
    def __init__(self, model, num_samples=9):
        self.num_samples = num_samples
        super().__init__(model)
    def train_step(self, data:Data, distance_matrix):
        samples, log_likelihoods_sum = self.model(data, self.num_samples)
        # select the first sample for each graph in the batch
        predicted_mappings = samples[:data.num_graphs]
        log_likelihoods_sum = log_likelihoods_sum[:data.num_graphs]
        # remaining samples for baseline calculation
        samples = samples[data.num_graphs:]
        comm_cost = communication_cost(data.edge_index, data.edge_attr, data.batch, distance_matrix, predicted_mappings)
        penalty_baseline = calculate_baseline(data.edge_index, data.edge_attr, data.batch, distance_matrix, samples, self.num_samples - 1)
        loss = torch.mean((comm_cost.detach() - penalty_baseline.detach()) * log_likelihoods_sum)
        return loss, float(comm_cost.sum())
class TrainerSR(Trainer):
    def __init__(self, model, num_samples=8):
        self.num_samples = num_samples
        super().__init__(model)
    def train_step(self, data:Data, distance_matrix):
        mappings, ll_sum = self.model(data, self.num_samples)
        comm_cost, baseline = communication_cost_multiple_samples(data.edge_index, data.edge_attr, data.batch, distance_matrix, mappings, self.num_samples, calculate_baseline=True)
        penalty_baseline = baseline.repeat(self.num_samples)
        loss = torch.mean((comm_cost.detach() - penalty_baseline.detach()) * ll_sum)
        return loss, float(comm_cost.sum())
class TrainerEMA(Trainer):
    def __init__(self, model, alpha=0.9):
        self.baseline = None
        self.alpha = alpha
        super().__init__(model)
    def train_step(self, data:Data, distance_matrix):
        predicted_mappings, log_likelihoods_sum = self.model(data, 1)
        # predicted_mappings shape: (batch_size, max_graph_size_in_batch)
        # log_likelihoods_sum shape: (batch_size,)
        penalty = communication_cost(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix, predicted_mappings)
        if self.baseline is None:
            self.baseline = penalty.mean()
        else:
            self.baseline = 0.9 * self.baseline + 0.1 * penalty.mean()
        loss = torch.mean((penalty.detach() - self.baseline.detach()) * log_likelihoods_sum)
        return loss, float(penalty.sum())
class TrainerGR(Trainer):
    def __init__(self, model, baseline_model, num_batches, stablize_baseline=False):
        self.num_batches = num_batches
        self.baseline_model = baseline_model
        self.baseline_model.decoding_type = 'greedy'
        self.baseline_model.eval()
        self.baseline = None
        self.batch_counter = 0
        self.stablize_baseline = stablize_baseline
        self.epoch_counter = 0 
        super().__init__(model)
    def train_step(self, data:Data, distance_matrix):
        predicted_mappings, log_likelihoods_sum = self.model(data, 1)
        penalty = communication_cost(data.edge_index, data.edge_attr, data.batch, distance_matrix, predicted_mappings)
        if self.stablize_baseline and self.epoch_counter == 0:
            if self.baseline is None:
                self.baseline = penalty.mean()
            else:
                self.baseline = 0.8 * self.baseline + 0.2 * penalty.mean()
        else:
            with torch.no_grad():
                baseline_mappings, _ = self.baseline_model(data, 1)
                self.baseline = communication_cost(data.edge_index, data.edge_attr, data.batch, distance_matrix, baseline_mappings)
        loss = torch.mean((penalty.detach() - self.baseline.detach()) * log_likelihoods_sum)
        self.batch_counter += 1
        if self.batch_counter == self.num_batches:
            self.batch_counter = 0
            self.epoch_counter += 1
        return loss, float(penalty.sum())
        

