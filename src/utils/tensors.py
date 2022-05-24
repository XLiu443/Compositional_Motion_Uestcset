import torch
import random
import ipdb
import copy
import numpy as np
import src.utils.rotation_conversions as geometry


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    databatch = [b[0] for b in batch]
    labelbatch = [b[1] for b in batch]
    lenbatch = [len(b[0][0][0]) for b in batch]
    alphabatch = [b[2] for b in batch]
    databatch1 = [b[3] for b in batch]
    databatch2 = [b[4] for b in batch]
   # lenbatch = [len(b[0][0][0][0]) for b in batch]
    databatchTensor = collate_tensors(databatch)
    databatchTensor1 = collate_tensors(databatch1)
    databatchTensor2 = collate_tensors(databatch2)

    labelbatchTensor = torch.as_tensor(labelbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    alphabatchTensor = torch.as_tensor(alphabatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    batch = {"x": databatchTensor, "y": labelbatchTensor, "alpha": alphabatchTensor, "compose_video1": databatchTensor1, "compose_video2": databatchTensor2,
             "mask": maskbatchTensor, "lengths": lenbatchTensor}
    return batch


humanact12_action_mask = {
    0: torch.tensor([0.6, 0.7, 0.7, 0.5, 0.8, 0.8, 0.5, 0.9, 0.9, 0.6, 1, 1, 0.5, 0.7, 0.7, 0.4, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
    1: torch.tensor([0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 0.1, 1, 1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    2: torch.tensor([0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 0.1, 1, 1, 0.1, 1, 1, 0.1, 0.6, 0.6, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
    3: torch.tensor([0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    4: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
    5: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.1, 0.8, 0.8, 0.9, 0.9, 1, 1, 1, 1]),
    6: torch.tensor([0.1, 0.8, 0.8, 0.2, 0.9, 0.9, 0.3, 1, 1, 0.4, 1, 1, 0.4, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    7: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
    8: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1]),
    9: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.8, 0.8, 0.9, 0.9, 1, 1, 1, 1]),
    10: torch.tensor([0.1, 0.4, 0.4, 0.1, 0.4, 0.4, 0.1, 0.5, 0.5, 0.1, 0.6, 0.6, 0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 1, 1, 1, 1, 1, 1]),
    11: torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 1, 1, 1, 1, 1, 1, 1, 1]),
}



def video_combine(video1, video2, action1, action2, alpha):
 #   old_walk_mask = torch.tensor([0, 0.8, 0.8, 0.1, 0.9, 0.9, 0.1, 1, 1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
 #   old_drink_mask = torch.tensor([0.1, 0, 0, 0, 0.1, 0.1, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1])


 #   old_warmup_mask = torch.tensor([0.6, 0.7, 0.7, 0.5, 0.8, 0.8, 0.5, 0.9, 0.9, 0.6, 1, 1, 0.5, 0.7, 0.7, 0.4, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1])
 #   old_run__mask = torch.tensor([0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 0.1, 1, 1, 0.1, 1, 1, 0.1, 0.6, 0.6, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1])
 #   old_jump_mask = torch.tensor([0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
 #   old_lift_dumbbell_mask = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.1, 0.8, 0.8, 0.9, 0.9, 1, 1, 1, 1])
    
 #   old_sit_mask = torch.tensor([0.1, 0.8, 0.8, 0.2, 0.9, 0.9, 0.3, 1, 1, 0.4, 1, 1, 0.4, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
 #   old_eat_mask = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1])
 #   old_trun_steering_wheel_mask = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1, 1])
 #   old_phone_mask = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.1, 0.8, 0.8, 0.9, 0.9, 1, 1, 1, 1])
 #   old_boxing_mask = torch.tensor([0.1, 0.4, 0.4, 0.1, 0.4, 0.4, 0.1, 0.5, 0.5, 0.1, 0.6, 0.6, 0.1, 0.8, 0.8, 0.1, 0.9, 0.9, 1, 1, 1, 1, 1, 1])
 #   old_throw_mask = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 1, 1, 1, 1, 1, 1, 1, 1])
        

   # firstframe_walkmask = torch.tensor([1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
   # firstframe_drinkmask = torch.tensor([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
 #   firstframe_walkmask = firstframe_walkmask.unsqueeze(1)
 #   firstframe_walkmask = firstframe_walkmask.repeat(1, 3)
 #   firstframe_drinkmask = firstframe_drinkmask.unsqueeze(1)
 #   firstframe_drinkmask = firstframe_drinkmask.repeat(1, 3)
 #   firstframe_drinkmask = firstframe_drinkmask.to(model.device)
 #   firstframe_walkmask = firstframe_walkmask.to(model.device)

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


compositional_actions = [{1,4}, {1,5}, {1,9}, {1,11}, {1,3}, {2,3}, {2,9}, {2,11}, {6,9}, {6,7}, {6,4}]


def collate_function(batch): 
  #  ipdb.set_trace()
    databatch = [b[0] for b in batch]
    labelbatch = [b[1] for b in batch]
 #   ret_trbatch = [b[2] for b in batch]

    action_num = len(set(labelbatch))
    instance_num = len(databatch) // action_num 
    generate_sequences = []
    actions1 = []
    actions2 = []
    alphas = []
    for i in range(action_num):
        for j in range(instance_num):
            alpha = random.random()
            ind1 = i*instance_num + j
    #        print("ind1",ind1) 
            video1 = databatch[ind1]
            action1 = labelbatch[ind1]
    #        print("action1",action1)
            avai_action = copy.deepcopy( list(  np.arange(action_num) ) )
            avai_action.remove(i)
            ind2 = random.sample(avai_action, 1)[0]*instance_num + random.sample( list(np.arange(instance_num)), 1)[0]
            video2 = databatch[ind2]
            action2 = labelbatch[ind2]
    #        if {action1, action2} in compositional_actions:
    #        print("action2",action2)
            new_sequence = video_combine(video1, video2, action1, action2, alpha)
    #        ret = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(new_sequence))
    #        ret_tr = ret_trbatch[ind1] 
    #        padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
    #        padded_tr[:, :3] = ret_tr
    #        ret = torch.cat((ret, padded_tr[:, None]), 1)
            ret = new_sequence.permute(1, 2, 0).contiguous()                        
 
            generate_sequences.append(ret)
            #generate_sequences.append(new_sequence)
            actions1.append(action1)
            actions2.append(action2)
            alphas.append(alpha)
#    ipdb.set_trace()
  #  generate_sequences = torch.stack(generate_sequences)
    actions1 = torch.tensor(actions1) 
    actions2 = torch.tensor(actions2)
    alphas =  torch.tensor(alphas)
 #   geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose))
 #   padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
 #   padded_tr[:, :3] = ret_tr
 #   ret = torch.cat((ret, padded_tr[:, None]), 1)
 #   ret = ret.permute(1, 2, 0).contiguous()
 
  #  lenbatch = [len(b[0][0][0]) for b in batch]
    lenbatch = [len(batch[0][0]) for b in batch]
    databatchTensor = collate_tensors(generate_sequences)
  #  databatchTensor = collate_tensors(databatch)
  #  labelbatchTensor = torch.as_tensor(labelbatch)

    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor)

    batch = {"x": databatchTensor, "y1": actions1, "y2": actions2,
             "mask": maskbatchTensor, "lengths": lenbatchTensor, "alphas": alphas}
  
    return batch
