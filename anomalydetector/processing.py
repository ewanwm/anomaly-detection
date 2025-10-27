
import os
import abc
import typing

import uproot
import numpy as np
import torch
from torch import masked 
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset, abc.ABC):

    def __init__(self, label:typing.Optional[int] = None):

        self._label: int | None = label

    @classmethod
    @abc.abstractmethod
    def _get_item(index:int): 
        raise NotImplementedError()
    
    @classmethod
    @abc.abstractmethod
    def get_n_features(self):
        raise NotImplementedError()
    
    def __getitem__(self, index:int):
        
        event = self._get_item(index)

        if self._label is None:
            return event, event
        
        else:
            return event, self._label
    
    

class InMemoryDataset(Dataset, abc.ABC):

    def __init__(self, filenames:list[str], label:typing.Optional[int] = None):
        
        super().__init__(label)

        self._raw_filenames = filenames
        self._data: torch.Tensor | None = None


    def __len__(self):

        return self._data.shape[0]


    def _get_item(self, index):
        
        return self._data[index]


    def get_data(self):

        return self._data

    
    @classmethod
    @abc.abstractmethod
    def process(self):

        raise NotImplementedError()
    


class InMemoryND280EventDataset(InMemoryDataset):

    def __init__(
            self, 
            filenames:list[str], 
            branches:list[str], 
            branch_scaling, 
            branch_mask_vals, 
            branch_mask_replace_vals, 
            use_masked_tensor:bool=False, 
            filter:str=None
        ):

        super().__init__(filenames)

        self._branches = branches
        self._branch_mask_vals = branch_mask_vals
        self._branch_scaling = branch_scaling
        self._branch_mask_replace_vals = torch.tensor(branch_mask_replace_vals)
        self._processed_filenames = []
        self._filter = filter

        self.use_masked_tensor = use_masked_tensor

    
    def get_n_features(self):
        return len(self._branches)


    def process(self):

        file_data_list = []

        # go through each raw file
        for raw_filename in self._raw_filenames:
            # open the file
            with uproot.open(raw_filename) as file:

                # get the sample_sum tree as a pandas array
                sample_sum = file['sample_sum'].arrays(
                    filter_branch=lambda b: b.name.find("Graph") == -1 and b.name.find("True") == -1 and b.typename.find("std::vector") == -1,
                    library='pd',
                    cut = self._filter
                )

                file_data:torch.Tensor = torch.tensor(sample_sum[self._branches].values.astype(np.float32))
                file_data = torch.clip(file_data, min = -1.0, max = 10000)
                file_data *= self._branch_scaling

                to_mask = sample_sum[self._branches].values.astype(np.float32) == self._branch_mask_vals

                file_data[to_mask] = self._branch_mask_replace_vals.expand(file_data.shape)[to_mask]

                if self.use_masked_tensor:
                    mask = torch.tensor(sample_sum[self._branches].values.astype(np.float32) != self._branch_mask_vals)
                    file_data:torch.masked.MaskedTensor = masked.masked_tensor(file_data, mask)

                file_data_list.append(file_data)

        self._data = torch.cat(file_data_list)

    def dump_branches(self):

        ## open up the first file in the file list and print it's available branches
        with uproot.open(self._raw_filenames[0]) as file:
            sample_sum = file['sample_sum'].arrays(
                filter_branch=lambda b: b.name.find("Graph") == -1 and b.typename.find("std::vector") == -1,
                library='pd'
            )

            print (":::::: Available branches ::::::::")

            for branch in file['sample_sum'].branches:
                print(f'  - {branch.name}: {branch.typename}')


class nd280EventDataset(Dataset):
    def __init__(
            self, 
            root, 
            filenames:list[str], 
            branches:list[str], 
            branch_scaling, 
            branch_mask_vals, 
            branch_mask_replace_vals, 
            use_masked_tensor:bool=False, 
            filter:str=None
        ):

        super().__init__()

        self.root = root
        self._raw_filenames = filenames
        self._branches = branches
        self._branch_mask_vals = branch_mask_vals
        self._branch_scaling = branch_scaling
        self._branch_mask_replace_vals = torch.tensor(branch_mask_replace_vals)
        self._processed_filenames = []
        self._filter = filter

        self.use_masked_tensor = use_masked_tensor

    def get_n_features(self):
        return len(self._branches)

    def process(self):

        event_idx = 0

        # go through each raw file
        for raw_filename in self._raw_filenames:
            # open the file
            with uproot.open(raw_filename) as file:

                # get the sample_sum tree as a pandas array
                sample_sum = file['sample_sum'].arrays(
                    filter_branch=lambda b: b.name.find("Graph") == -1 and b.name.find("True") == -1 and b.typename.find("std::vector") == -1,
                    library='pd',
                    cut = self._filter
                )

                for _, row in sample_sum[self._branches].iterrows():
    
                    event_tensor:torch.Tensor = torch.tensor(row.values.astype(np.float32))
                    event_tensor = torch.clip(event_tensor, min = -1.0, max = 5000)
                    event_tensor *= self._branch_scaling

                    to_mask = row.values.astype(np.float32) == self._branch_mask_vals
                    event_tensor[to_mask] = self._branch_mask_replace_vals[to_mask]

                    if self.use_masked_tensor:
                        mask = torch.tensor(row.values.astype(np.float32) != self._branch_mask_vals)
                        event_tensor:torch.masked.MaskedTensor = masked.masked_tensor(event_tensor, mask)

                    filename = os.path.join(self.root, f'event_{event_idx}_proc.pt')
                    
                    self._processed_filenames.append(filename)
                    torch.save(event_tensor, filename)

                    if event_idx %10000 == 0:
                        print(f"Processed {event_idx} events")

                    event_idx += 1

    def __len__(self):
        
        return len(self._processed_filenames)


    def _get_item(self, index):

        with torch.serialization.safe_globals([masked.MaskedTensor]):

            event = torch.load(os.path.join(self.root, f'event_{index}_proc.pt'))

        return event
    

    def dump_branches(self):

        ## open up the first file in the file list and print it's available branches
        with uproot.open(self._raw_filenames[0]) as file:
            sample_sum = file['sample_sum'].arrays(
                filter_branch=lambda b: b.name.find("Graph") == -1 and b.typename.find("std::vector") == -1,
                library='pd'
            )

            print (":::::: Available branches ::::::::")

            for branch in file['sample_sum'].branches:
                print(f'  - {branch.name}: {branch.typename}')