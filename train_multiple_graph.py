# %%
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.data import Data
from models.mpnn_ptr import MpnnPtr, MpnnTransformer
from utils.utils import init_weights
from utils.datagenerate import DistanceMatrix, DistanceMatrixNew
import torch
from torch import nn
import matplotlib.pyplot as plt
from datetime import datetime
from graphdataset import MultiSizeGraphDataset, getDataLoader
from train.trainers import TrainerInitPop, TrainerSR
from train.validation import validate_dataloader
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np

# %% initializing the parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_graph_size = 121
batch_size_train = 32
batch_size_dev = 128
# change the number to 0 to use a random initialize model parameters
# saved_model_path = 'models_data_multiple/full/models_data/model_init_pop_03-08_19-57.pt'
saved_model_path = None
lr = 0.0001
# lr_decay_gamma = .96
num_epochs = 10 #27
num_samples = 8
beam_width = 8
training_algorithm = 'pretrain'  # 'init_pop' or 'pretrain'
model = "lstm"
transformer_version = "v1"
root_folder = Path('./models_data_multiple') # to save the trained model, the logs and the validation results
save_folder = root_folder / "small"  # ('models_data_final', 'full', 'small')
distance_matrix_dict = DistanceMatrixNew(121) # DistanceMatrixNew(max_graph_size) or DistanceMatrix()
# %%
# setting the random states for reproducibility
torch.manual_seed(0)
np.random.seed(0)
# %% change these to change the size of dataset
root_train = 'data_tgff/multiple_small/train'
root_dev = 'data_tgff/multiple_small/test'
train_good_files = (None, ['traindata_multiple_TGFF_norm_64.pt'])[0]
dev_good_files = (None,['testdata_multiple_TGFF_norm_64.pt'])[0]
train_dataloader = getDataLoader(
    root_train, batch_size_train, max_graph_size=max_graph_size, raw_file_names=train_good_files)
dev_dataloader = getDataLoader(
    root_dev, batch_size_dev, max_graph_size=max_graph_size, raw_file_names=dev_good_files)
# %% initialize the models
if model == "lstm":
    mpnn_ptr = MpnnPtr(input_dim=max_graph_size, embedding_dim=max_graph_size + 7,
                   hidden_dim=max_graph_size + 7, K=3, n_layers=1, p_dropout=0, device=device, logit_clipping=False)
elif model == "transformer":
    mpnn_ptr = MpnnTransformer(input_dim=max_graph_size, embedding_dim=max_graph_size + 7, hidden_dim=max_graph_size + 7, K=3, n_layers=1, p_dropout=0, device=device, logit_clipping=True, version=transformer_version)
mpnn_ptr.to(device)
# %% load model if saved
if saved_model_path:
    mpnn_ptr.load_state_dict(torch.load(saved_model_path, map_location=device))
else:
    mpnn_ptr.apply(init_weights)
# %% initialize the training algorithm
if training_algorithm == 'init_pop':
    trainer = TrainerInitPop(mpnn_ptr, num_samples)
elif training_algorithm == 'pretrain':
    trainer = TrainerSR(mpnn_ptr, num_samples)
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=lr)
# lr_schedular = torch.optim.lr_scheduler.StepLR(
# optim, step_size=1, gamma=lr_decay_gamma)
loss_list_train = []
loss_list_dev = []
datetime_suffix = datetime.now().strftime('%m-%d_%H-%M')
save_folder.mkdir(parents=True, exist_ok=True)
# %%
torch.autograd.set_detect_anomaly(True)
avg_valid_comm_cost = validate_dataloader(
    mpnn_ptr, tqdm(dev_dataloader, leave=False), distance_matrix_dict, beam_width) / len(dev_dataloader.dataset)
loss_list_dev.append(avg_valid_comm_cost)
print_str = f'Epoch: 0/{num_epochs} Dev Comm cost: {avg_valid_comm_cost}'
print(print_str)
logs_save_folder = save_folder / "logs"
logs_save_folder.mkdir(parents=True, exist_ok = True)
f = open(logs_save_folder / f'{datetime_suffix}_train_loss.txt', 'a')
# f.write(f"{print_str}\n")
model_save_folder = save_folder / "models_data"
model_save_folder.mkdir(parents=True, exist_ok=True)
per_epoch_save_folder = model_save_folder / f'per_epoch'
per_epoch_save_folder.mkdir(parents=True, exist_ok=True)
# %% training starts
for epoch in range(num_epochs):
    avg_train_comm_cost = trainer.train(
        tqdm(train_dataloader, leave=False), distance_matrix_dict, optim) / len(train_dataloader.dataset)
    # lr_schedular.step()
    avg_valid_comm_cost = validate_dataloader(
        mpnn_ptr, tqdm(dev_dataloader, leave=False), distance_matrix_dict, beam_width) / len(dev_dataloader.dataset)
    print_str = f'Epoch: {epoch + 1}/{num_epochs}, Train Comm Cost: {avg_train_comm_cost:.4f}, Dev Comm Cost: {avg_valid_comm_cost:.4f}'
    print(print_str)
    # save the model
    f.write(f"{print_str}\n")
    torch.save(mpnn_ptr.state_dict(), per_epoch_save_folder /f'mpnn_ptr_{training_algorithm}_{datetime_suffix}_{epoch + 1}.pt')
    loss_list_train.append(avg_train_comm_cost)
    loss_list_dev.append(avg_valid_comm_cost)
    f.flush()
# %%
# save the model
torch.save(mpnn_ptr.state_dict(
), model_save_folder / f'model_{training_algorithm}_{datetime_suffix}.pt')

# %%
# plot loss_list_pre
fig, ax = plt.subplots()
ax.plot(loss_list_train, label='train')
ax.plot(loss_list_dev, label='dev')
ax.set_xlabel('Epoch')
ax.set_ylabel('Communication cost')
ax.legend()
# %% 
plot_save_folder = save_folder / 'plots'
plot_save_folder.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_save_folder / f'loss_list_{max_graph_size}_{datetime_suffix}.png', dpi=300)
# %% 
# root_test = "data_tgff/multiple/test"
# test_good_files = ['testdata_multiple_TGFF_norm_64.pt']
# test_dataloader = getDataLoader(
#     root_test, batch_size_dev, max_graph_size=max_graph_size, raw_file_names=test_good_files)
# # %% 
# avg_test_comm_cost = validate_dataloader(
#     mpnn_ptr, tqdm(test_dataloader, leave=False), distance_matrix_dict, beam_width) / len(test_dataloader.dataset)
# print_str = f'Epoch: {num_epochs}/{num_epochs} Test Comm cost: {avg_test_comm_cost}'
# print(print_str)
# f.write(f"{print_str}\n")
# f.close()
# %% save the loss list
loss_list_save_folder = save_folder / "loss_list"
loss_list_save_folder.mkdir(parents=True, exist_ok = True)
torch.save(loss_list_train,
           loss_list_save_folder / f'loss_list_train_{training_algorithm}_{max_graph_size}_{datetime_suffix}.pt')
torch.save(loss_list_dev,
           loss_list_save_folder / f'loss_list_dev_{training_algorithm}_{max_graph_size}_{datetime_suffix}.pt')
# %%