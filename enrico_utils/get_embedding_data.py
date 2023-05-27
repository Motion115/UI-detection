"""Implements dataloaders for ENRICO_embedding dataset."""

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
import csv
import os
import pickle
import torch

class EnricoInferDataset(Dataset):
    def __init__(self, data_dir, mode, class_num = 20, random_seed=42, train_split=0.65, val_split=0.15, test_split=0.2):
        """Instantiate ENRICO_embedding dataset.
        Args:
            data_dir (str): Data directory.
        """
        super(EnricoInferDataset, self).__init__()
        self.class_num = class_num
        # load data
        pkl_file = os.path.join(data_dir, "enrico_embedding_150.pkl")
        with open(pkl_file, 'rb') as f:
            corpus = pickle.load(f)

        # stable randon seed, so that the split is the same for all runs
        random.seed(random_seed)
        # shuffle
        random.shuffle(corpus)
        
        # use several lists to store the data
        keys = []
        labels = []        # the random sample choice
        numpy_embedding = []
        for element in corpus:
            keys.append(element['id'])
            labels.append(element['label_id'])
            numpy_embedding.append(element['numpy_embedding'])

        if mode == 'train':
            start_index = 0
            end_index = int(len(keys) * (train_split + val_split))
        elif mode == 'val':
            start_index = int(len(keys) * train_split)
            end_index = int(len(keys) * (train_split + val_split))
        elif mode == 'test':
            start_index = int(len(keys) * (train_split + val_split))
            end_index = len(keys)
        else:
            raise ValueError('Mode must be either "train", "val", or "test".')

        self.keys = keys[start_index:end_index]
        self.labels = labels[start_index:end_index]
        self.numpy_embedding = numpy_embedding[start_index:end_index]

    def __len__(self):
        """Get number of samples in dataset."""
        return len(self.keys)

    def __getitem__(self, idx):
        """Get item in dataset.
        Args:
            idx (int): Index of data to get.
        Returns:
            list: List of (anchor, positive, negative) from triplet loss.
        """
        return [self.numpy_embedding[idx], self.one_hot(self.labels[idx])]
    
    def one_hot(self, label_id):
        """Convert label_id to one-hot vector."""
        one_hot = [0] * self.class_num
        one_hot[label_id] = 1
        # to tensor
        one_hot = torch.tensor(one_hot)
        # set one_hot data type to int
        one_hot = one_hot.type(torch.long)
        return one_hot


def get_learned_embedding_dataset(data_dir, batch_size, num_workers=0):
    train_set = EnricoInferDataset(data_dir, mode='train')
    val_set = EnricoInferDataset(data_dir, mode='val')
    test_set = EnricoInferDataset(data_dir, mode='test')

    train_dataloader = DataLoader(train_set, num_workers=num_workers, batch_size=batch_size)
    val_dataloader = DataLoader(val_set, num_workers=num_workers, batch_size=batch_size)
    test_dataloader = DataLoader(test_set, num_workers=num_workers, batch_size=batch_size)
    return tuple([train_dataloader, val_dataloader, test_dataloader])




class EnricoEmbeddingDataset(Dataset):    
    def __init__(self, data_dir, file_name, random_seed=42):
        """Instantiate ENRICO_embedding dataset.
        Args:
            data_dir (str): Data directory.
            random_seed (int, optional): Seed to split dataset on and shuffle data on. Defaults to 42.
        """
        super(EnricoEmbeddingDataset, self).__init__()

        # load data
        pkl_file = os.path.join(data_dir, file_name)
        with open(pkl_file, 'rb') as f:
            corpus = pickle.load(f)
        # stable randon seed, so that the split is the same for all runs
        random.seed(random_seed)
        # shuffle
        random.shuffle(corpus)
        
        # use several lists to store the data
        self.keys = []
        self.choice = []        # the random sample choice
        self.anchor_embeddings = []
        self.positive_embeddings = []
        self.negative_embeddings = []
        for element in corpus:
            self.keys.append(element['class'])
            self.choice.append(element['choice'])
            self.anchor_embeddings.append(element['anchor_embedding'])
            self.positive_embeddings.append(element['positive_embedding'])
            self.negative_embeddings.append(element['negative_embedding'])

    def __len__(self):
        """Get number of samples in dataset."""
        return len(self.keys)

    def __getitem__(self, idx):
        """Get item in dataset.
        Args:
            idx (int): Index of data to get.
        Returns:
            list: List of (anchor, positive, negative) from triplet loss.
        """
        return [self.anchor_embeddings[idx], self.positive_embeddings[idx], self.negative_embeddings[idx]]


def get_embedding_dataloader(data_dir, file_name, batch_size, num_workers=0):
    """Get dataloaders for this dataset.

    Args:
        data_dir (str): Data directory.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 0.

    Returns:
        tuple: Dataloader for dataset
    """

    dataset = EnricoEmbeddingDataset(data_dir, file_name)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
    return dataloader