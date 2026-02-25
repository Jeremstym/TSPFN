import torch
from torch.utils.data import Sampler, DataLoader
import math
import random

class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.batch_size = batch_size
        self.indices_by_label = {}
        for idx, label in enumerate(labels):
            if label not in self.indices_by_label:
                self.indices_by_label[label] = []
            self.indices_by_label[label].append(idx)
        
        self.labels = list(self.indices_by_label.keys())
        self.total_size = len(labels)
        
        # Calculate proportions for each class
        self.proportions = {
            label: len(idxs) / self.total_size 
            for label, idxs in self.indices_by_label.items()
        }

    def __iter__(self):
        # Shuffle indices within each class at the start of epoch
        for label in self.indices_by_label:
            random.shuffle(self.indices_by_label[label])
        
        # Create copies of indices to "pop" from
        pools = {label: list(idxs) for label, idxs in self.indices_by_label.items()}
        
        for _ in range(self.__len__()):
            batch = []
            # Fill the batch according to proportions
            for label in self.labels:
                # Calculate how many of this label should be in the batch
                # Using round to handle the "33 vs 34" logic
                num_to_take = min(
                    len(pools[label]), 
                    math.ceil(self.proportions[label] * self.batch_size)
                )
                
                # Ensure we don't exceed remaining batch capacity
                capacity = self.batch_size - len(batch)
                num_to_take = min(num_to_take, capacity)
                
                for _ in range(num_to_take):
                    batch.append(pools[label].pop())
            
            # If batch still has room due to rounding, fill from whoever is left
            while len(batch) < self.batch_size:
                available_labels = [l for l in self.labels if pools[l]]
                if not available_labels: break
                batch.append(pools[available_labels[0]].pop())
                
            yield batch

    def __len__(self):
        return math.ceil(self.total_size / self.batch_size)