# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

class DataProvider():
    def __init__(self, dataset, **kw):
        self.args = kw
        self.dataset = dataset
        self.epoch = 0
        self.DataLoader = None #data.DataLoader(self.dataset, **self.args)
        self.iteration = 0
        self.build()
        pass
    
    def build(self):
        self.DataLoader = data.DataLoader(self.dataset, **self.args)
        self.DataLoader = enumerate(self.DataLoader)

    def __next__(self):
        if self.DataLoader == None:
            self.build()
        
        try:
            _, batch = self.DataLoader.__next__()
            # img, label = batch
            self.iteration += 1
            return batch
        
        except StopIteration:
            self.epoch += 1
            self.iteration = 0
            self.build()
            _, batch = self.DataLoader.__next__()
            # img, label = batch
            return batch
    next = __next__

    def __iter__(self):
        return self