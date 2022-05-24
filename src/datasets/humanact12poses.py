import pickle as pkl
import numpy as np
import os
from .dataset import Dataset
import ipdb
import torch
import random
import pandas as pd
class HumanAct12Poses(Dataset):
    dataname = "humanact12"

    def __init__(self, datapath="data/HumanAct12Poses", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        pkldatafilepath = os.path.join(datapath, "humanact12poses.pkl")
        data = pkl.load(open(pkldatafilepath, "rb"))
#        ipdb.set_trace()
        self._pose = [x for x in data["poses"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [x for x in data["joints3D"]]
        self._actions = [x for x in data["y"]]

        total_num_actions = 12 + 9
        self.num_classes = total_num_actions
        self._train = list(range(len(self._pose)))
        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}
        self._action_classes = humanact12_coarse_action_enumerator

        new_tensor = torch.tensor((), dtype=torch.float64)
        new_tensor2 = torch.tensor((), dtype=torch.int64)
        self.compose_poses = new_tensor.new_zeros((200*9, 60, 24, 6))
        self.compose_labels = new_tensor2.new_zeros((200*9, 1))
        
        action0 = self.label_to_action(0)
        action1 = self.label_to_action(1)
        action2 = self.label_to_action(2)
        action3 = self.label_to_action(3)
        action4 = self.label_to_action(4)
        action5 = self.label_to_action(5)
        action6 = self.label_to_action(6)
        action7 = self.label_to_action(7)
        action8 = self.label_to_action(8)
        action9 = self.label_to_action(9)
        action10 = self.label_to_action(10)
        action11 = self.label_to_action(11)

        choices0 = np.argwhere(np.array(self._actions)[self._train] == action0).squeeze(1)
        choices1 = np.argwhere(np.array(self._actions)[self._train] == action1).squeeze(1)
        choices2 = np.argwhere(np.array(self._actions)[self._train] == action2).squeeze(1)
        choices3 = np.argwhere(np.array(self._actions)[self._train] == action3).squeeze(1)
        choices4 = np.argwhere(np.array(self._actions)[self._train] == action4).squeeze(1)
        choices5 = np.argwhere(np.array(self._actions)[self._train] == action5).squeeze(1)
        choices6 = np.argwhere(np.array(self._actions)[self._train] == action6).squeeze(1)
        choices7 = np.argwhere(np.array(self._actions)[self._train] == action7).squeeze(1)
        choices8 = np.argwhere(np.array(self._actions)[self._train] == action8).squeeze(1)
        choices9 = np.argwhere(np.array(self._actions)[self._train] == action9).squeeze(1)
        choices10 = np.argwhere(np.array(self._actions)[self._train] == action10).squeeze(1)
        choices11 = np.argwhere(np.array(self._actions)[self._train] == action11).squeeze(1)

        action_dataindex = {0:choices0, 1:choices1, 2:choices2, 3:choices3, 4:choices4,
                            5:choices5, 6:choices6, 7:choices7, 8:choices8, 9:choices9, 10:choices10, 11:choices11}
        leg_arr = [1,2,4,5,7,8,10,11]
        orig_actions = np.arange(0, 12)
        alllist = []
        avg_list = []
        for i in range(len(action_dataindex)):
            meandelta = []
            choices = action_dataindex[i]
            for j in choices:
                video, target = self._get_item_data_index(self._train[j])
                assert i==target
                videodelta = video_delta(video)
                videodelta = torch.abs(videodelta)
                meandelta.append(videodelta)
            meandelta = torch.cat(meandelta)
            meanvalue = torch.mean(meandelta, dim=0)
            meanvalue = meanvalue.numpy()
            leg_meanvalue = meanvalue[leg_arr]
            leg_mean = np.mean(leg_meanvalue, axis=0)
        #    ipdb.set_trace()
            leg_avg = np.mean(leg_mean)
            avg_list.append(leg_avg)
            leg_list = list(leg_mean)
            leg_list.append(target)
            alllist.append(leg_list)
            print(target, leg_mean)
        column=['v', 'v', 'v', 'v', 'v', 'v', 'target']
        test=pd.DataFrame(columns=column,data=alllist)
        test.to_csv('test.csv')
        avg_list = np.array(avg_list)
        leg_actions = (-avg_list).argsort()[:4]
        print("leg actions:", leg_actions)
        arm_actions = np.delete(orig_actions, leg_actions)
        leg_mask = torch.tensor([0.1, 0.8, 0.8, 0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        arm_mask = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.1, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 1, 1])

        human12_actionmask = {}
        for action in leg_actions:
            human12_actionmask[action] = leg_mask
        for action in arm_actions:
            human12_actionmask[action] = arm_mask
        self.human12_actionmask =  human12_actionmask

#        self.compositional_actions = []
#        for act1 in leg_actions:
#            for act2 in arm_actions:
#                group = (act1, act2)
#                self.compositional_actions.append(group)

        np.random.seed(1000)
        data_index1 = np.random.choice(choices1, 10, replace=False)
        data_index2 = np.random.choice(choices2, 10, replace=False)
        data_index6 = np.random.choice(choices6, 10, replace=False)       
        data_index4 = np.random.choice(choices4, 10, replace=False)
        data_index5 = np.random.choice(choices5, 10, replace=False)
        data_index9 = np.random.choice(choices9, 10, replace=False)
        data_index11 = np.random.choice(choices11, 10, replace=False)
        data_index3 = np.random.choice(choices3, 10, replace=False)
        data_index7 = np.random.choice(choices7, 10, replace=False)
        data_index10 = np.random.choice(choices10, 10, replace=False) 
        data_index8 = np.random.choice(choices8, 10, replace=False)
        data_index0 = np.random.choice(choices0, 10, replace=False)
        
        action_dataindex = {action1:data_index1, action2:data_index2, action4:data_index4, action3:data_index3, action8:data_index8,
                            action5:data_index5, action6:data_index6, action7:data_index7, action9:data_index9, action11:data_index11,
                            action10:data_index10, action0:data_index0 }

        print("action_dataindex", action_dataindex)
#        allactions = np.arange(0,12)
#        self.compositional_actions = []
#        for i in allactions:
#            for j in np.arange(i+1,12):
#                self.compositional_actions.append((i,j))    
        self.compositional_actions = compositional_actions = [(1,4), (1,5), (1,9), (1,11), (2,9), (2,11), (6,9), (6,7), (6,4)]

        self.compose_alphas = []
        self.compose_video1 = []
        self.compose_video2 = []
        for i in range(len(self.compositional_actions)):
            element = self.compositional_actions[i]
            action_1 = element[0]
            action_2 = element[1]
            dataind1 = action_dataindex[action_1] 
            dataind2 = action_dataindex[action_2]

            count=0
            for ind1 in dataind1:
                for ind2 in dataind2:
                #    alpha = random.random()
                #    print("alpha", alpha)
                    alpha1=np.random.normal(0.5, 0.1, 1)[0]
            #        print("alpha", alpha1)
                    alpha2=np.random.normal(0.5, 0.1, 1)[0]
                    action1 = action_1
                    action2 = action_2
                    video1 = self._get_item_data_index(self._train[ind1])[0]
                    video2 = self._get_item_data_index(self._train[ind2])[0]
                    new_sequence1 = video_combine(human12_actionmask, video1, video2, action1, action2, alpha1)
                    new_sequence2 = video_combine(human12_actionmask, video1, video2, action1, action2, alpha2)
                    self.compose_poses[count+i*200] = new_sequence1
                    self.compose_poses[count+1+i*200] = new_sequence2
                    self.compose_labels[count+i*200] = i+12
                    self.compose_labels[count+1+i*200] = i+12
                    self.compose_alphas.append(alpha1)
                    self.compose_alphas.append(alpha2)
                    self.compose_video1.append(video1)
                    self.compose_video1.append(video1)
                    self.compose_video2.append(video2)
                    self.compose_video2.append(video2)   
                    count = count+2

        self.compose_alphas = torch.tensor(self.compose_alphas)
      #  print("alphas:", self.compose_alphas)
        newnum = len(self._train) + 200*9
        print("data length", newnum)
        self._train = list(range(newnum))

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 24, 3)
        return pose

def video_delta(video):
    video_frame = video[0,:,:]
    video_frame = video_frame.unsqueeze(0).repeat(60, 1, 1)
    video_delta = video - video_frame
    return video_delta

#compositional_actions = [(1,4), (1,5), (1,9), (1,11), (2,9), (2,11), (6,9), (6,7), (6,4)]


humanact12_coarse_action_enumerator = {
    0: "warm_up",
    1: "walk",
    2: "run",
    3: "jump",
    4: "drink",
    5: "lift_dumbbell",
    6: "sit",
    7: "eat",
    8: "turn steering wheel",
    9: "phone",
    10: "boxing",
    11: "throw"
}


#humanact12_action_mask = {
#    0: torch.tensor([0.6, 0.7, 0.7, 0.5, 0.8, 0.8, 0.5, 0.9, 0.9, 0.6, 1, 1, 0.5, 0.7, 0.7, 0.4, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
#    1: torch.tensor([0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 0.1, 1, 1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
#    2: torch.tensor([0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 0.1, 1, 1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
#    3: torch.tensor([0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
#    4: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
#    5: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.1, 0.8, 0.8, 0.9, 0.9, 1, 1, 1, 1]),
#    6: torch.tensor([0.1, 0.8, 0.8, 0.2, 0.9, 0.9, 0.3, 1, 1, 0.4, 1, 1, 0.4, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
#    7: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
#    8: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
#    9: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.8, 0.8, 0.9, 0.9, 1, 1, 1, 1]),
#    10: torch.tensor([0.1, 0.4, 0.4, 0.1, 0.4, 0.4, 0.1, 0.5, 0.5, 0.1, 0.6, 0.6, 0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 1, 1, 1, 1, 1, 1]),
#    11: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 1, 1, 1, 1, 1, 1, 1, 1]),
#}



def video_combine(humanact12_action_mask, video1, video2, action1, action2, alpha):

    old_mask1 = humanact12_action_mask[action1]
    old_mask2 = humanact12_action_mask[action2]

    video_mask1 = alpha*old_mask1 / (alpha*old_mask1 + (1-alpha)*old_mask2)
    video_mask2 = (1-alpha)*old_mask2 / (alpha*old_mask1 + (1-alpha)*old_mask2)

    video_mask1 = video_mask1.unsqueeze(0).unsqueeze(2)
    video_mask1 = video_mask1.repeat(60, 1, 6)
   # walk_mask =  video_mask1.to(model.device)
    video_mask2 = video_mask2.unsqueeze(0).unsqueeze(2)
    video_mask2 = video_mask2.repeat(60, 1, 6)
  #  drink_mask = drink_mask.to(model.device)

    video1_firstframe_mask = video_mask1[0,:,:]
    video2_firstframe_mask = video_mask2[0,:,:]

    video1_frame = video1[0,:,:]
    video1_frame = video1_frame.unsqueeze(0).repeat(60, 1, 1)
    video1_delta = video1 - video1_frame


    video2_frame = video2[0,:,:]
    video2_frame = video2_frame.unsqueeze(0).repeat(60, 1, 1)
    video2_delta = video2 - video2_frame

    standardframe =  video1[0,:,:]*video1_firstframe_mask + video2[0,:,:]*video2_firstframe_mask
    standardframe = standardframe.unsqueeze(0).repeat(60, 1, 1)

    compose_delta = video1_delta * video_mask1 + video2_delta * video_mask2
    compose_delta = compose_delta + standardframe

    return compose_delta
