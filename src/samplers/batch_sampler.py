import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler
import ipdb

class BatchSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_indices, data_label, batch_size, num_instances):
    #    ipdb.set_trace()
        self.data_indices = data_indices
        self.data_label = data_label        
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_actions_per_batch = self.batch_size // self.num_instances

        self.index_dic = defaultdict(list)
        for index, data_ind in enumerate(self.data_indices):
            action = self.data_label[index]
            self.index_dic[action].append(data_ind)
        self.actions = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for action in self.actions:
            idxs = self.index_dic[action]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
    #    ipdb.set_trace()
        batch_idxs_dict = defaultdict(list)

        for action in self.actions:
            idxs = copy.deepcopy(self.index_dic[action])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[action].append(batch_idxs)
                    batch_idxs = []

        avai_actions = copy.deepcopy(self.actions)
        final_idxs = []

        while len(avai_actions) >= self.num_actions_per_batch:
            selected_actions = random.sample(avai_actions, self.num_actions_per_batch)
            for action in selected_actions:
                batch_idxs = batch_idxs_dict[action].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[action]) == 0:
                    avai_actions.remove(action)

        return iter(final_idxs)

    def __len__(self):
        return self.length
