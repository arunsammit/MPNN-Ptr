#%%
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from typing import List, Optional, Union, Tuple
from torch_geometric.loader import DataLoader
import os
import re
from torch.utils.data.sampler import Sampler, BatchSampler, SubsetRandomSampler
#%%
def get_transform(max_num_nodes=121):
    def transform(data:Data)->Data:
        x = torch.zeros(data.num_nodes, max_num_nodes)
        x[data.edge_index[0], data.edge_index[1]] = data.edge_attr.squeeze(-1)
        data_new = Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        return data_new
    return transform

class MultipleGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=get_transform(), pre_transform=None, pre_filter=None, raw_file_names: Union[str, List[str], Tuple] = None):
        self.raw_file_names = raw_file_names
        self.processed_file_names = ["data.pt", "start_pos_data.pt"]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # TODO: think how you can take the below attribute into account when slicing the data 
        self.start_pos_data = torch.load(self.processed_paths[1])
        self.transform
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names
    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names
    @raw_file_names.setter
    def raw_file_names(self, value: Union[str, List[str], Tuple]):
        self._raw_file_names = value
    @processed_file_names.setter
    def processed_file_names(self, value: Union[str, List[str], Tuple]):
        self._processed_file_names = value
    def process(self,):
        datalist_all = []
        # max_num_nodes = 0
        start_pos_data = []
        cumm_count = 0
        for raw_path in self.raw_paths:
            datalist = torch.load(raw_path, map_location=torch.device('cpu'))
            start_pos_data.append(cumm_count)
            print(f"{raw_path} loaded: {len(datalist)}")
            # max_num_nodes = max(max_num_nodes, datalist[0].num_nodes)
            datalist_all += datalist
            cumm_count += len(datalist)
        # for data in datalist_all:
        #     data.x = torch.cat([data.x, torch.zeros(data.x.size(0), max_num_nodes - data.x.size(1))], dim=1)
        if self.pre_filter is not None:
            datalist_all = [data for data in datalist_all if self.pre_filter(data)]
        if self.pre_transform is not None:
            datalist_all = [self.pre_transform(data) for data in datalist_all]
        # pprint(datalist_all)
        data, slices = self.collate(datalist_all)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(start_pos_data, self.processed_paths[1])

class BucketSampler(Sampler):
    def __init__(self, dataset, batch_size, generator=None) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = generator
        start_pos_data = dataset.start_pos_data
        start_end_indices = []
        for i in range(len(start_pos_data) - 1):
            start_end_indices.append((start_pos_data[i], start_pos_data[i+1]))
        start_end_indices.append((start_pos_data[-1], len(self.dataset)))
        # print(start_end_indices)
        ranges  = [range(start, end) for start, end in start_end_indices]
        subset_samplers = [SubsetRandomSampler(range_, generator=generator) for range_ in ranges]
        self.samplers = [
            BatchSampler(subset_sampler, batch_size, drop_last=False) for subset_sampler in subset_samplers
        ]
        self._len = 0
        for sampler in self.samplers:
            self._len += len(sampler)
        
    def __iter__(self):
        iterators = [iter(sampler) for sampler in self.samplers]
        while iterators:
            randint = torch.randint(0, len(iterators),size=(1,), generator=self.generator)[0]
            try:
                yield next(iterators[randint])
            except StopIteration:
                iterators.pop(randint)
    def __len__(self):
        return self._len
def getDataLoader(root, batch_size, raw_file_names=None, max_graph_size=121):
    if raw_file_names is None:
        raw_file_names = [f for f in os.listdir(f'{root}/raw') if os.path.isfile(f'{root}/raw/{f}') and int(re.split('_|[.]',f)[-2]) <= max_graph_size]
    dataset = MultipleGraphDataset(root, raw_file_names=raw_file_names, transform=get_transform(max_num_nodes=max_graph_size))
    if len(raw_file_names) > 1:
        sampler = BucketSampler(dataset, batch_size)
    else:
        sampler = BatchSampler(SubsetRandomSampler(range(len(dataset))), batch_size, drop_last=False)
    return DataLoader(dataset, batch_sampler=sampler)

def main():
    root = 'data_tgff/multiple/train'
    raw_file_names = [f for f in os.listdir(f'{root}/raw') if os.path.isfile(f'{root}/raw/{f}') and f != "train_data.pt" and int(re.split('_|[.]',f)[-2]) <= 121]
    dataset = MultipleGraphDataset(root, raw_file_names=raw_file_names)
    print(f"{len(dataset)} graphs loaded")
    print(f"{dataset[0]}")
    print(f"{dataset[0].edge_index}")
    print(f"{dataset[0].edge_attr}")
    print(f"{dataset[0]}")
#%%
def write_sizes(root, size_file_dir):
    files = sorted(
        list(filter(lambda f: os.path.isfile(os.path.join(root,f)), os.listdir(root))), 
        key=lambda f: int(re.split('_|[.]',f)[-2])
    )
    print(files)
    with open(os.path.join(size_file_dir, 'data_sizes.txt'), 'w') as sizes_file:
        for file in files:
            data = torch.load(os.path.join(root, file), map_location=torch.device('cpu'))
            sizes_file.write(f"{file}: {len(data)}\n")
    return

#%%
if __name__ == '__main__':
    # cleanup()
    # main()
    # debug_dataset('data_tgff/multiple/train/raw','data_tgff/multiple/train')
    # print(get_good_files('/home/arun/Desktop/train_data/mod_data/'))
    pass