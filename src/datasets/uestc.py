import os
import pandas as pd
import numpy as np
import pickle as pkl
import src.utils.rotation_conversions as geometry
import torch

from .dataset import Dataset
from .ntu13 import action2motion_joints
import ipdb
from src.config import UESTC_PATH
import random

def get_z(cam_s, cam_pos, joints, img_size, flength):
    """
    Solves for the depth offset of the model to approx. orth with persp camera.
    """
    # Translate the model itself: Solve the best z that maps to orth_proj points
    joints_orth_target = (cam_s * (joints[:, :2] + cam_pos) + 1) * 0.5 * img_size
    height3d = np.linalg.norm(np.max(joints[:, :2], axis=0) - np.min(joints[:, :2], axis=0))
    height2d = np.linalg.norm(np.max(joints_orth_target, axis=0) - np.min(joints_orth_target, axis=0))
    tz = np.array(flength * (height3d / height2d))
    return float(tz)


def get_trans_from_vibe(vibe, index, use_z=True):
    alltrans = []
    for t in range(vibe["joints3d"][index].shape[0]):
        # Convert crop cam to orig cam
        # No need! Because `convert_crop_cam_to_orig_img` from demoutils of vibe
        # does this already for us :)
        # Its format is: [sx, sy, tx, ty]
        cam_orig = vibe["orig_cam"][index][t]
        x = cam_orig[2]
        y = cam_orig[3]
        if use_z:
            z = get_z(cam_s=cam_orig[0],  # TODO: There are two scales instead of 1.
                      cam_pos=cam_orig[2:4],
                      joints=vibe['joints3d'][index][t],
                      img_size=540,
                      flength=500)
            # z = 500 / (0.5 * 480 * cam_orig[0])
        else:
            z = 0
        trans = [x, y, z]
        alltrans.append(trans)
    alltrans = np.array(alltrans)
    return alltrans - alltrans[0]


def video_delta(video):
    video_frame = video[0,:,:]
    video_frame = video_frame.unsqueeze(0).repeat(60, 1, 1)
    video_delta = video - video_frame
    return video_delta


def video_combine(human12_actionmask, video1, video2, action1, action2, alpha):
#    old_mask1 = humanact12_action_mask[action1]
#    old_mask2 = humanact12_action_mask[action2]
    old_mask1 = human12_actionmask[action1]
    old_mask2 = human12_actionmask[action2]
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


convert_label = {0:0, 1:1, 2:4, 3:5, 4:8, 5:11, 6:12, 7:16, 8:22, 9:28}
back_convert_label = {0:0, 1:1, 4:2, 5:3, 8:4, 11:5, 12:6, 16:7, 22:8, 28:9}

class UESTC(Dataset):
    dataname = "uestc"

    def __init__(self, datapath="data/uestc", method_name="vibe", view="all", **kargs):
        datapath = UESTC_PATH 
        self.method_name = method_name
        self.view = view
        super().__init__(**kargs)

        # Load pre-computed #frames data
        with open(os.path.join(UESTC_PATH, 'info', 'num_frames_min.txt'), 'r') as f:
            num_frames_video = np.asarray([int(s) for s in f.read().splitlines()])

        # Out of 118 subjects -> 51 training, 67 in test
        all_subjects = np.arange(1, 119)
        self._tr_subjects = [
            1, 2, 6, 12, 13, 16, 21, 24, 28, 29, 30, 31, 33, 35, 39, 41, 42, 45, 47, 50,
            52, 54, 55, 57, 59, 61, 63, 64, 67, 69, 70, 71, 73, 77, 81, 84, 86, 87, 88,
            90, 91, 93, 96, 99, 102, 103, 104, 107, 108, 112, 113]
        self._test_subjects = [s for s in all_subjects if s not in self._tr_subjects]

        # Load names of 25600 videos
        with open(os.path.join(datapath, 'info', 'names.txt'), 'r') as f:
            videos = f.read().splitlines()

        self._videos = videos

        if self.method_name == "vibe":
            vibe_data_path = os.path.join(datapath, "vibe_cache_refined.pkl")
            vibe_data = pkl.load(open(vibe_data_path, "rb"))

            self._pose = vibe_data["pose"]
            num_frames_method = [p.shape[0] for p in self._pose]
            globpath = os.path.join(datapath, "globtrans_usez.pkl")

            if os.path.exists(globpath):
                self._globtrans = pkl.load(open(globpath, "rb"))
            else:
                self._globtrans = []
                from tqdm import tqdm
                for index in tqdm(range(len(self._pose))):
                    self._globtrans.append(get_trans_from_vibe(vibe_data, index, use_z=True))
                pkl.dump(self._globtrans, open("globtrans_usez.pkl", "wb"))
            self._joints = vibe_data["joints3d"]
            self._jointsIx = action2motion_joints
        else:
            raise ValueError("This method name is not recognized.")

        num_frames_video = np.minimum(num_frames_video, num_frames_method)
        num_frames_video = num_frames_video.astype(int)
        self._num_frames_in_video = [x for x in num_frames_video]

        N = len(videos)
        self._actions = np.zeros(N, dtype=int)
        for ind in range(N):
            self._actions[ind] = self.parse_action(videos[ind])

        self._actions = [x for x in self._actions]

        total_num_actions = 40
        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}
        self.num_classes = len(keep_actions)

        self._train = []
        self._test = []

        self.info_actions = []

        def get_rotation(view):
            theta = - view * np.pi/4
            axis = torch.tensor([0, 1, 0], dtype=torch.float)
            axisangle = theta*axis
            matrix = geometry.axis_angle_to_matrix(axisangle)
            return matrix

        # 0 is identity if needed
        rotations = {key: get_rotation(key) for key in [0, 1, 2, 3, 4, 5, 6, 7]}

        for index, video in enumerate(videos):
            act, view, subject, side = self._get_action_view_subject_side(video)
            self.info_actions.append({"action": act,
                                      "view": view,
                                      "subject": subject,
                                      "side": side})
            if self.view == "frontview":
                if side != 1:
                    continue
            # rotate to front view
            if side != 1:
                # don't take the view 8 in side 2
                if view == 8:
                    continue
                rotation = rotations[view]
                global_matrix = geometry.axis_angle_to_matrix(torch.from_numpy(self._pose[index][:, :3]))
                # rotate the global pose
                self._pose[index][:, :3] = geometry.matrix_to_axis_angle(rotation @ global_matrix).numpy()
                # rotate the joints
                self._joints[index] = self._joints[index] @ rotation.T.numpy()
                self._globtrans[index] = (self._globtrans[index] @ rotation.T.numpy())

            # add the global translation to the joints
            self._joints[index] = self._joints[index] + self._globtrans[index][:, None]

            if subject in self._tr_subjects:
                self._train.append(index)
            elif subject in self._test_subjects:
                self._test.append(index)
            else:
                raise ValueError("This subject doesn't belong to any set.")

        # Select only sequences which have a minimum number of frames
        if self.num_frames > 0:
            threshold = self.num_frames*3/4
        else:
            threshold = 0

        method_extracted_ix = np.where(num_frames_video >= threshold)[0].tolist()
        self._train = list(set(self._train) & set(method_extracted_ix))
        # keep the test set without modification
        self._test = list(set(self._test))

        action_classes_file = os.path.join(datapath, "info/action_classes.txt")
        with open(action_classes_file, 'r') as f:
            self._action_classes = np.array(f.read().splitlines())

     #   ipdb.set_trace()
        action_dataindex = {}
        for i in range(total_num_actions):
            action = self.label_to_action(i)
            choices = np.argwhere(np.array(self._actions)[self._train] == action).squeeze(1)
            action_dataindex[action] = choices

        leg_arr = [1,2,4,5,7,8,10,11]
#        orig_actions = np.arange(0, 40)
        alllist = []
        avg_list = []
        for i in range(len(action_dataindex)):
            meandelta = []
            choices = action_dataindex[i]
            for j in choices:
            #    assert i==target
                video, target = self._get_item_data_index(self._train[j])
                assert i==target
                videodelta = video_delta(video)
                videodelta = torch.abs(videodelta)
            #    assert videodelta>=0
            #    meanvalue = torch.mean(videodelta)
                meandelta.append(videodelta)
      #      ipdb.set_trace()
            meandelta = torch.cat(meandelta)
            meanvalue = torch.mean(meandelta, dim=0)
            meanvalue = meanvalue.numpy()
            leg_meanvalue = meanvalue[leg_arr]
      #      leg_meanvalue = meanvalue[arm_arr]
            leg_mean = np.mean(leg_meanvalue, axis=0)
            leg_avg = np.mean(leg_mean)
            avg_list.append(leg_avg)
            leg_list = list(leg_mean)
            leg_list.append(target)
            alllist.append(leg_list)
            print(target, leg_mean)
    #    ipdb.set_trace()           
        column=['v', 'v', 'v', 'v', 'v', 'v', 'target']
        test=pd.DataFrame(columns=column,data=alllist)
        test.to_csv('test.csv')
    #    ipdb.set_trace()
        avg_list = np.array(avg_list)
        selected_actions = np.array([0,1,4,5,8,11,12,16,22,28])
        print("selected actions", self._action_classes[selected_actions])
        avg_list = avg_list[selected_actions] 
        leg_actions = (-avg_list).argsort()[:5]
      #  print("leg actions:", leg_actions)
        convert_legactions = []
        for leg_act in leg_actions:
            convert_legactions.append( convert_label[leg_act] )
    #    convert_legactions = np.array(convert_legactions)
        print("convert leg actions", convert_legactions)
        arm_actions = np.delete(selected_actions, leg_actions)
        print("arm actions", arm_actions)
        leg_mask = torch.tensor([0.1, 0.8, 0.8, 0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        arm_mask = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.1, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 1, 1])

        human12_actionmask = {}
        for action in convert_legactions:
            human12_actionmask[action] = leg_mask
        for action in arm_actions:
            human12_actionmask[action] = arm_mask
        self.human12_actionmask = human12_actionmask

        self.compositional_actions = []
        for act1 in convert_legactions:
            for act2 in arm_actions:
                group = (back_convert_label[act1], back_convert_label[act2])
                self.compositional_actions.append(group)
#        ipdb.set_trace()
        new_tensor = torch.tensor((), dtype=torch.float64)
        new_tensor2 = torch.tensor((), dtype=torch.int64)
        self.compose_poses = new_tensor.new_zeros((200*25, 60, 24, 6))
        self.compose_labels = new_tensor2.new_zeros((200*25, 1))
        
        action_dataindices = {}
        np.random.seed(1000) 
        for i in selected_actions:
            choices = action_dataindex[i]
            if i<10:     
                dataindices = np.random.choice(choices, 10, replace=False)
                action_dataindices[i] = dataindices
            else:
                dataindices = np.random.choice(choices, 10, replace=False) 
                action_dataindices[i] = dataindices   
        print("action_dataindex", action_dataindices)        

        self.compose_alphas = []
        self.compose_video1 = []
        self.compose_video2 = []
        save_labels = []
        save_poses = []
        for i in range(len(self.compositional_actions)):
            element = self.compositional_actions[i]
            action_1 = element[0]
            action_2 = element[1]
            action_1 = convert_label[action_1]
            action_2 = convert_label[action_2] 
            dataind1 = action_dataindices[action_1]
            dataind2 = action_dataindices[action_2]
            count=0
            for ind1 in dataind1:
                for ind2 in dataind2:
                    alpha1=np.random.normal(0.5, 0.1, 1)[0]
                    alpha2=np.random.normal(0.5, 0.1, 1)[0]
                #    alpha2 = random.random()
                    action1 = action_1
                    action2 = action_2
                    video1 = self._get_item_data_index(self._train[ind1])[0]
                    video2 = self._get_item_data_index(self._train[ind2])[0]
                    new_sequence1 = video_combine(human12_actionmask, video1, video2, action1, action2, alpha1)
                    new_sequence2 = video_combine(human12_actionmask, video1, video2, action1, action2, alpha2)
                    self.compose_poses[count+i*200] = new_sequence1
                    self.compose_poses[count+1+i*200] = new_sequence2
                    self.compose_labels[count+i*200] = i+40
                    self.compose_labels[count+1+i*200] = i+40
                    self.compose_alphas.append(alpha1)
                    self.compose_alphas.append(alpha2)
                    self.compose_video1.append(video1)
                    self.compose_video1.append(video1)
                    self.compose_video2.append(video2)
                    self.compose_video2.append(video2)
                    count = count+2
             #       save_labels.append(i)
             #       save_poses.append(new_sequence1)

        self.new_train_indices = []
        for i in selected_actions:      
            choices = action_dataindex[i]
            train_choices = np.array(self._train)[choices]
            train_choices = list(train_choices) 
            self.new_train_indices = self.new_train_indices + train_choices
        print("length on selected 10 classes:", len(self.new_train_indices))
        self.compose_alphas = torch.tensor(self.compose_alphas)
        print("alphas:", self.compose_alphas)
        newnum = len(self.new_train_indices) + 200*25
        self.new_train = list(range(newnum))

#        save_labels = torch.tensor(save_labels)
#        save_poses = torch.stack(save_poses)
#        vibestyle = {"poses": [], "y": []}
#        vibestyle["poses"] = np.array(save_poses)
#        vibestyle["y"] = np.array(save_labels.unsqueeze(1))
#        savepath = "uestc_compose_poses1000.pkl"
#        pkl.dump(vibestyle, open(savepath, "wb"))
#        ipdb.set_trace()

    def _load_joints3D(self, ind, frame_ix):
        if len(self._joints[ind]) == 0:
            raise ValueError(
                f"Cannot load index {ind} in _load_joints3D function.")
        if self._jointsIx is not None:
            joints3D = self._joints[ind][frame_ix][:, self._jointsIx]
        else:
            joints3D = self._joints[ind][frame_ix]

        return joints3D

    def _load_rotvec(self, ind, frame_ix):
        # 72 dim smpl
        pose = self._pose[ind][frame_ix, :].reshape(-1, 24, 3)
        return pose

    def _get_action_view_subject_side(self, videopath):
        # TODO: Can be moved to tools.py
        spl = videopath.split('_')
        action = int(spl[0][1:])
        view = int(spl[1][1:])
        subject = int(spl[2][1:])
        side = int(spl[3][1:])
        return action, view, subject, side

    def _get_videopath(self, action, view, subject, side):
        # Unused function
        return 'a{:d}_d{:d}_p{:03d}_c{:d}_color.avi'.format(
            action, view, subject, side)

    def parse_action(self, path, return_int=True):
        # Override parent method
        info, _, _, _ = self._get_action_view_subject_side(path)
        if return_int:
            return int(info)
        else:
            return info


if __name__ == "__main__":
    dataset = UESTC()
