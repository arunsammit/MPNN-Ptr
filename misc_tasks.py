import torch
import os
from pathlib import Path
import re

def cleanup():
    import subprocess
    subprocess.run(['rm', '-r', 'data_tgff/multiple/train/processed'])

def debug_dataset(root, bugs_output_file_dir):
    files = sorted(
        list(filter(lambda f: os.path.isfile(os.path.join(root,f)), os.listdir(root))), 
        key=lambda f: int(re.split('_|[.]',f)[-2])
    )
    with open(os.path.join(bugs_output_file_dir, 'data_bugs.txt'), 'w') as data_bugs_file:
        problematic_files = set()
        for file in files:
            datalist = torch.load(os.path.join(root, file), map_location=torch.device('cpu'))
            for i, data in enumerate(datalist):
                try:
                    max_val_edge = data.edge_index.max().item()
                except RuntimeError as e:
                    data_bugs_file.write(f"exception {e}:\t")
                    data_bugs_file.write(f"{file} has graph {i} with {data.num_nodes} nodes, edge_index.size(): {data.edge_index.size()}, edge_attr.size(): {data.edge_attr.size()}\n") 
                    problematic_files.add(file)
                if data.num_nodes <= max_val_edge:
                    data_bugs_file.write(f"{file}: graph {i} {data.num_nodes} nodes, {max_val_edge} is present in the edge list\n")
                    problematic_files.add(file)
        data_bugs_file.write(f"{len(problematic_files)} files have problems\n")
        for file in problematic_files:
            data_bugs_file.write(f"{file}\n")
    return
def get_good_files(root):
    files = sorted(
        list(filter(lambda f: os.path.isfile(os.path.join(root,f)), os.listdir(root))), 
        key=lambda f: int(re.split('_|[.]',f)[-2])
    )
    good_files = []
    for file in files:
        datalist = torch.load(os.path.join(root, file), map_location=torch.device('cpu'))
        is_good = True
        for data in datalist:
            try:
                max_val_edge = data.edge_index.max().item()
            except RuntimeError as e:
                is_good = False
                break
            if data.num_nodes <= max_val_edge:
                is_good = False
                break
        if is_good:
            good_files.append(file)
    return good_files
def create_small_dataset(path_from, path_to, sizes):
    import os
    import random
    Path(path_to).mkdir(parents=True, exist_ok=True)
    files = sorted(
        list(filter(lambda f: os.path.isfile(os.path.join(path_from,f)), os.listdir(path_from))), 
        key=lambda f: int(re.split('_|[.]',f)[-2])
    )
    total_num_graphs = 0
    for file in files:
        datalist = torch.load(os.path.join(path_from, file), map_location=torch.device('cpu'))
        curr_size = min(next(sizes), len(datalist))
        total_num_graphs += curr_size
        random.shuffle(datalist)
        datalist = datalist[:curr_size]
        torch.save(datalist, os.path.join(path_to, file))
    print(f"{total_num_graphs} graphs in total")
if __name__ == "__main__":
    path_from = 'data_tgff/multiple/train/raw'
    path_to = 'data_tgff/multiple/train_final/raw'
    def get_sizes(start, diff):
        x = start
        while True:
            yield x
            x += diff
    create_small_dataset(path_from, path_to, get_sizes(3000, 3600))
