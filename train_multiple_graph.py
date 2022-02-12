# %%
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.data import Data
from models.mpnn_ptr import MpnnPtr
from utils.utils import init_weights
from utils.datagenerate import generate_distance_matrix, DistanceMatrix, DistanceMatrixNew
import torch
from torch import nn
import matplotlib.pyplot as plt
from datetime import datetime
from graphdataset import MultiSizeGraphDataset, getDataLoader
from train.trainers import TrainerInitPop, TrainerSR
from train.validation import validate_dataloader
from tqdm.auto import tqdm
from pathlib import Path
# %% initializing the parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_graph_size = 121
batch_size_train = 128
batch_size_dev = 256
saved_model_path = None
lr = 0.001
# lr_decay_gamma = .96
num_epochs = 27
num_samples = 8
beam_width = 8
training_algorithm = 'init_pop'  # 'init_pop' or 'pretrain'
save_folder = Path('models_data_multiple') / "small"  # 'models_data_final'
# %%
root_train = 'data_tgff/multiple_small/train'
root_dev = 'data_tgff/multiple_small/test'
train_good_files = None  # ['traindata_multiple_TGFF_norm_64.pt']
dev_good_files = ['testdata_multiple_TGFF_norm_64.pt']
train_dataloader = getDataLoader(
    root_train, batch_size_train, max_graph_size=max_graph_size, raw_file_names=train_good_files)
dev_dataloader = getDataLoader(
    root_dev, batch_size_dev, max_graph_size=max_graph_size, raw_file_names=dev_good_files)
# %% initialize the models
mpnn_ptr = MpnnPtr(input_dim=max_graph_size, embedding_dim=max_graph_size + 7, hidden_dim=max_graph_size + 7, K=3, n_layers=1, p_dropout=0, device=device, logit_clipping=False)
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
# DistanceMatrixNew(max_graph_size) or DistanceMatrix()
distance_matrix_dict = DistanceMatrixNew(max_graph_size)
optim = torch.optim.Adam(mpnn_ptr.parameters(), lr=lr)
# lr_schedular = torch.optim.lr_scheduler.StepLR(
    # optim, step_size=1, gamma=lr_decay_gamma)
loss_list_train = []
loss_list_dev = []
datetime_suffix = datetime.now().strftime('%m-%d_%H-%M')
save_folder.mkdir(parents=True, exist_ok=True)
# %%
avg_valid_comm_cost = validate_dataloader(
    mpnn_ptr, tqdm(dev_dataloader, leave=False), distance_matrix_dict, beam_width) / len(dev_dataloader.dataset)
loss_list_dev.append(avg_valid_comm_cost)
print_str = f'Epoch: 0/{num_epochs} Dev Comm cost: {avg_valid_comm_cost}'
print(print_str)
f = open(Path(save_folder) / f'{datetime_suffix}_train_loss.txt', 'a')
f.write(print_str)
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
    f.write(print_str)
    # torch.save(mpnn_ptr.state_dict(), save_folder /f'mpnn_ptr_{epoch + 1}_{datetime_suffix}.pt')
    loss_list_train.append(avg_train_comm_cost)
    loss_list_dev.append(avg_valid_comm_cost)
f.close()
# %%
# save the model
torch.save(mpnn_ptr.state_dict(
), f'models_data/model_{training_algorithm}_{max_graph_size}_{datetime_suffix}.pt')

# %%
# plot loss_list_pre
fig, ax = plt.subplots()
ax.plot(loss_list_train, label='train')
ax.plot(loss_list_dev, label='dev')
ax.set_xlabel('Epoch')
ax.set_ylabel('Communication cost')
ax.legend()
# %%
fig.savefig(f'plots/loss_list_{max_graph_size}_{datetime_suffix}.png', dpi=300)

# %% save the loss list
torch.save(loss_list_train,
           f'plots/loss_list_train_{training_algorithm}_{max_graph_size}_{datetime_suffix}.pt')
torch.save(loss_list_dev,
           f'plots/loss_list_dev_{training_algorithm}_{max_graph_size}_{datetime_suffix}.pt')
# %%
