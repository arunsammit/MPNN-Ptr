from utils.baseline import calculate_baseline
from utils.utils import communication_cost, communication_cost_multiple_samples
from torch_geometric.data import Data
import torch
class Trainer:
    def __init__(self, model, device):
        self.device = device
        self.model = model
    def pre_steps(self, data:Data):
        self.model.train()
        return data.to(self.device)
    def train_step(self, data:Data, distance_matrix):
        raise NotImplementedError
class TrainerInitPop(Trainer):
    def __init__(self, model, device, num_samples=9):
        self.num_samples = num_samples
        super().__init__(model, device)
    def train_step(self, data:Data, distance_matrix):
        data = self.pre_steps(data)
        samples, log_likelihoods_sum = self.model(data, self.num_samples)
        # select the first sample for each graph in the batch
        predicted_mappings = samples[:data.num_graphs]
        log_likelihoods_sum = log_likelihoods_sum[:data.num_graphs]
        # remaining samples for baseline calculation
        samples = samples[data.num_graphs:]
        comm_cost = communication_cost(data.edge_index, data.edge_attr, data.batch, distance_matrix, predicted_mappings)
        penalty_baseline = calculate_baseline(data.edge_index, data.edge_attr, data.batch, distance_matrix, samples, self.num_samples - 1)
        loss = torch.mean((comm_cost.detach() - penalty_baseline.detach()) * log_likelihoods_sum)
        return loss, comm_cost.sum()
class TrainerSR(Trainer):
    def __init__(self, model, device, num_samples=8):
        self.num_samples = num_samples
        super().__init__(model, device)
    def train_step(self, data:Data, distance_matrix):
        data = self.pre_steps(data)
        mappings, ll_sum = self.model(data, self.num_samples)
        comm_cost, baseline = communication_cost_multiple_samples(data.edge_index, data.edge_attr, data.batch, distance_matrix, mappings, self.num_samples, calculate_baseline=True)
        penalty_baseline = baseline.repeat(self.num_samples)
        loss = torch.mean((comm_cost.detach() - penalty_baseline.detach()) * ll_sum)
        return loss, comm_cost.sum()
class TrainerEMA(Trainer):
    def __init__(self, model, device, alpha=0.9):
        self.baseline = None
        self.alpha = alpha
        super().__init__(model, device)
    def train_step(self, data:Data, distance_matrix):
        data = self.pre_steps(data)
        predicted_mappings, log_likelihoods_sum = self.model(data, 1)
        # predicted_mappings shape: (batch_size, max_graph_size_in_batch)
        # log_likelihoods_sum shape: (batch_size,)
        penalty = communication_cost(data.edge_index, data.edge_attr, data.batch, data.num_graphs, distance_matrix, predicted_mappings)
        if self.baseline is None:
            self.baseline = penalty.mean()
        else:
            self.baseline = 0.9 * self.baseline + 0.1 * penalty.mean()
        # penalty_baseline = calculate_baseline(data.edge_index, data.edge_attr, data.batch, data.num_graphs,  distance_matrix, samples)
        loss = torch.mean((penalty.detach() - self.baseline.detach()) * log_likelihoods_sum)
        return loss, penalty.sum()