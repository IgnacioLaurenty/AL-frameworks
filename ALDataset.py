from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset
from typing import Optional, Tuple, Union


class ALDataset(Dataset):
    '''
    Dataset built for Active Learning, with specific methods allowing direct or partial annotation.

    Parameters
    ----------
    dataset : Dataset
        a dataset.

    Arguments
    ----------
    valid_size : float, default=0.
        To define a validation set, perfectly labeled. Not used during training.

    Attributes
    ----------
    num_classes : int
        number of classes to predict in the dataset.

    class_to_idx : dict
        of form {class_name : index}
    
    partial_labels : torch.Tensor, shape : (dataset.__len__(), num_classes)
        Partial Labels one-hot-encoding. Initialized with torch.ones, this tensor is updated with the update_label method.
    
    labels : torch.Tensor, shape : (dataset.__len__(), num_classes)
        one-hot-encoding of the ground-thrue classes.

    input_dim : tuple
        shape of the input datas
        
    '''
    def __init__(self, dataset : Dataset, valid_size :int = 0 , **kwargs) -> None:
        self.dataset = dataset        
        self.class_to_idx = self.dataset.class_to_idx
        self.num_classes = len(self.class_to_idx.keys())        
        self.partial_labels = torch.ones((len(self.dataset),self.num_classes)).float()
        self.labels = torch.zeros_like(self.partial_labels).float()
        for label, one_hot in zip(self.dataset.targets, self.labels):
            one_hot[label] = 1.
        self.input_dim = self.__getitem__(0)[0].shape
        self.set_val_idx(valid_size = valid_size)

    def __getitem__(self, idx :int) -> Tuple[Tensor, Tensor, int] :
        inputs, _ = self.dataset.__getitem__(idx)
        return inputs, self.partial_labels[idx], idx
        
    def update_label(self, idx : Union[int , ndarray , Tensor] , new_label : Tensor) ->None:
        self.partial_labels[idx]= new_label
    
    def true_labels(self, indices : Optional[Union[int, ndarray, Tensor]] = None ) -> None:
        if indices is None :
            self.partial_labels = self.labels
        else :
            self.partial_labels[indices] = self.labels[indices]

            
    def set_val_idx(self, valid_size : Union[float, int] ) -> None :
        if valid_size != 0:
            _, self.val_idx = train_test_split(list(range(len(self.dataset))), test_size=valid_size)
            self.true_labels(self.val_idx)
        else :
            self.val_idx = []
        self.train_idx = self.remaining_indices()
                
    def train_indices(self) -> ndarray :
        indices = torch.where(self.partial_labels.sum(-1)<self.num_classes)[0].tolist()
        return np.array(list(set(indices)-set(self.val_idx)))
    
    def classified_indices(self) -> ndarray :
        '''
        return indices of inputs which current label is atomic.
        '''
        indices = torch.where(self.partial_labels.sum(-1)==1)[0].tolist()
        return np.array(list(set(indices)-set(self.val_idx)))
    
    def remaining_indices(self) -> ndarray :
        return torch.where(self.partial_labels.sum(-1)>1)[0].numpy()

    def __len__(self) -> int :
        return len(self.dataset)