import torch
from .cae import CAE
import ipdb
import numpy as np
#from pytorch3d.structures import Pointclouds
frames = np.linspace(0, 59, 10).astype(int)
import time

def unpatchify(model, x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = model.patch_embed.patch_size[0]
    # print("p",model.patch_embed.patch_size)
    h = w = int(x.shape[1]**.5)
    # print("h,w",h,w)
    assert h * w == x.shape[1]
        
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    # print("x.shape", x.shape)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs


class CVAE(CAE):
    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def forward(self, batch):
#        ipdb.set_trace()
#        start = time.time()
        if self.outputxyz:
            batch["x_xyz"] = self.rot2xyz(batch["x"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]
        # encode
        batch.update(self.encoder(batch))
        batch["z"] = self.reparameterize(batch)
        
        # decode
        batch.update(self.decoder(batch))
        # if we want to output xyz
        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]

#        compose_video1_xyz = self.rot2xyz(batch["compose_video1"], batch["mask"])
#        compose_video2_xyz = self.rot2xyz(batch["compose_video2"], batch["mask"])
#        self.param2xyz["jointstype"] = "smpl"
#        batch_skeleton = self.rot2xyz(batch["output"], batch["mask"])
#        self.param2xyz["jointstype"] = "vertices"
#        batch_images_smpl = self.smpl_render(batch["output_xyz"])
#        compose_video1_xyz_smpl = self.smpl_render(compose_video1_xyz)
#        compose_video2_xyz_smpl = self.smpl_render(compose_video2_xyz)
#        batch_resize_smpl = []
#        batch_resize_video1 = []
#        batch_resize_video2 = []
#        batch_video1_scores = []
#        batch_video2_scores = []
#        for index in range(len(batch_images_smpl)):
#            label = batch["y"][index].item()
#            label = label - 12
#            if label>=0: 
#                crop_smpl, crop_yx = self.crop_smpl_skeleton(batch_images_smpl[index])
#                action1, action2 = self.compositional_actions[label]
#                y1,x1,y2,x2 = crop_yx
#                video_scores1 = self.attention_map(batch_skeleton[index],y1,y2,x1,x2,action1)
#                video_scores2 = self.attention_map(batch_skeleton[index],y1,y2,x1,x2,action2)
#                batch_video1_scores.append(video_scores1)
#                batch_video2_scores.append(video_scores2)
#                transimg = self.trans(crop_smpl.permute(0,3,1,2))
#                batch_resize_smpl.append(transimg)
#                crop_video1, _ = self.crop_smpl_skeleton(compose_video1_xyz_smpl[index])
#                crop_video2, _ = self.crop_smpl_skeleton(compose_video2_xyz_smpl[index])
#                transimg_video1 = self.trans(crop_video1.permute(0,3,1,2))
#                transimg_video2 = self.trans(crop_video2.permute(0,3,1,2))
#                batch_resize_video1.append(transimg_video1)
#                batch_resize_video2.append(transimg_video2)
#        if len(batch_video1_scores)==0:
#            batch["recon1"] = torch.zeros(10,1)
#            batch["recon2"] = torch.zeros(10,1)
#            batch["video1"] = torch.zeros(10,1)
#            batch["video2"] = torch.zeros(10,1)
#            batch["video1_mask"] = torch.zeros(10,1)
#            batch["video2_mask"] = torch.zeros(10,1)
#        else:
#            batch_video1_scores = torch.stack(batch_video1_scores)
#            batch_video2_scores = torch.stack(batch_video2_scores)
#            batch_resize_smpl = torch.stack(batch_resize_smpl)
#            batch_resize_video1 = torch.stack(batch_resize_video1)
#            batch_resize_video2 = torch.stack(batch_resize_video2)
#            bs, fn, _ = batch_video1_scores.shape
#            batch_video1_scores = batch_video1_scores.reshape(bs*fn, -1)
#            batch_video2_scores = batch_video2_scores.reshape(bs*fn, -1)
#            bs, fn, ch, height, width = batch_resize_smpl.shape
#            batch_resize_smpl = batch_resize_smpl.reshape(bs*fn, ch, height, width)
#            _, reconstruction1, video1_mask = self.mae_model(batch_resize_smpl, mask_ratio=0.75, score=batch_video1_scores.detach().cpu().numpy())
#            _, reconstruction2, video2_mask = self.mae_model(batch_resize_smpl, mask_ratio=0.75, score=batch_video2_scores.detach().cpu().numpy())        
#            reconstruction1 = unpatchify(self.mae_model, reconstruction1)
#            reconstruction2 = unpatchify(self.mae_model, reconstruction2)
#            batch_resize_video1 = batch_resize_video1.reshape(bs*fn, ch, height, width)
#            batch_resize_video2 = batch_resize_video2.reshape(bs*fn, ch, height, width)
#            video1_mask = video1_mask.unsqueeze(-1).repeat(1, 1, self.mae_model.patch_embed.patch_size[0]**2 *3)  
#            video1_mask = self.mae_model.unpatchify(video1_mask)  # 1 is removing, 0 is keeping
#            video2_mask = video2_mask.unsqueeze(-1).repeat(1, 1, self.mae_model.patch_embed.patch_size[0]**2 *3)
#            video2_mask = self.mae_model.unpatchify(video2_mask)
#            batch["recon1"] = reconstruction1
#            batch["recon2"] = reconstruction2
#            batch["video1"] = batch_resize_video1
#            batch["video2"] = batch_resize_video2
#            batch["video1_mask"] = video1_mask
#            batch["video2_mask"] = video2_mask

        return batch
 
    def attention_map(self,skeleton_pointclouds,y1,y2,x1,x2,label):        
        gamma = 0.0
        beta = 1
        alpha = 2
        skeleton_pointclouds = skeleton_pointclouds.permute(2,0,1)
        skeleton_pointclouds = skeleton_pointclouds[frames]
        newarr = np.zeros((96, 96), dtype=np.float32)
        tmp_inds = np.where(newarr==0)       
        tmp_indices = np.array([tmp_inds[0],tmp_inds[1]]).transpose(1,0)
        tmp_indices = torch.from_numpy(tmp_indices).unsqueeze(0).to(skeleton_pointclouds.device)
        tmp_indices = tmp_indices.unsqueeze(0).repeat(10,1,1,1)        

        point_images = torch.matmul(skeleton_pointclouds, self.R) + self.T
        result = (point_images[:,:,:2] + 1) * (128/2.0)
        result = 128-result
        coord_points = torch.cat((result[:,:,1].unsqueeze(2), result[:,:,0].unsqueeze(2)), dim=2)
        coord_points[:,:,0] = coord_points[:,:,0] - 16
        coord_points[:,:,0] = coord_points[:,:,0] - y1
        coord_points[:,:,1] = coord_points[:,:,1] - x1
        h = y2-y1
        w = x2-x1
        row_scale = 96/h
        col_scale = 96/w
        coord_points[:,:,0] = coord_points[:,:,0]*row_scale
        coord_points[:,:,1] = coord_points[:,:,1]*col_scale
        score = self.human12_actionmask[label]
            
        coord_points = coord_points.unsqueeze(1)
        _, _, rep1, _ = tmp_indices.shape
        _, _, rep2, _ = coord_points.shape
        tmp_indices = tmp_indices.repeat(1,rep2,1,1)
        coord_points = coord_points.repeat(1,rep1,1,1)
        coord_points = coord_points.permute(0,2,1,3)
        res = torch.linalg.norm((tmp_indices-coord_points),ord=alpha,dim=3)
        score = score.unsqueeze(1).unsqueeze(0).repeat(10,1,rep1).to(res.device)
        attention_map = torch.div( (score+gamma), (0.1+beta*res) )
        attention_map = attention_map.reshape(10, rep2, 96, 96)
        
#        for index in frames:   
#            pointcloud = skeleton_pointclouds[index]
#            newarr = np.zeros((224, 224), dtype=np.float32)
#            tmp_inds = np.where(newarr==0) 
#            point_images = torch.mm(pointcloud.clone().detach(), self.R.squeeze()) + self.T
#            result = (point_images[:,:2] + 1) * (224/2.0)
#            result = 224-result
#            coord_points = torch.cat((result[:,1].unsqueeze(1), result[:,0].unsqueeze(1)), dim=1)
#            coord_points[:,0] = coord_points[:,0] - 35
#            coord_points[:,0] = coord_points[:,0] - y1
#            coord_points[:,1] = coord_points[:,1] - x1
#            h = y2-y1 
#            w = x2-x1
#            row_scale = 224/h
#            col_scale = 224/w
#            coord_points[:,0] = coord_points[:,0]*row_scale
#            coord_points[:,1] = coord_points[:,1]*col_scale
#            assert torch.where(coord_points>224)[0].shape[0]==0 
#            score = self.human12_actionmask[label]
#            tmp_indices = np.array([tmp_inds[0],tmp_inds[1]]).transpose(1,0)
#            tmp_indices = torch.from_numpy(tmp_indices).unsqueeze(0).to(coord_points.device)
#            coord_points = coord_points.unsqueeze(0)
#            _, rep1, _ = tmp_indices.shape
#            _, rep2, _ = coord_points.shape
#            tmp_indices = tmp_indices.repeat(rep2,1,1)
#            coord_points = coord_points.repeat(rep1,1,1)
#            coord_points = coord_points.permute(1,0,2)
#            res = torch.linalg.norm((tmp_indices-coord_points),ord=alpha,dim=2)
#            score = score.unsqueeze(1).repeat(1,rep1).to(res.device)
#            attention_map = torch.div( (score+gamma), (0.1+beta*res) ) 
#            attention_map = attention_map.reshape(rep2, 224, 224)

#            res = np.linalg.norm((tmp_indices-point), ord=alpha, axis=1, keepdims=True)
#            attention_map = (score + gamma ) / (0.1+beta*res)
#            coord_points = []
#            all_map = []
#            for i in range(len(pointcloud)):
#                verts = pointcloud[i].unsqueeze(0)
#                rgb = np.array([[0,0,0.9,1]])
#                rgb = np.repeat(rgb, 1, axis=0)
#                rgb = torch.Tensor(rgb).to(self.device)                
#                new_point_cloud = Pointclouds(points=[verts], features=[rgb])
#                point_images = self.renderer(new_point_cloud)
#                new_images2 = np.roll(point_images[..., :3].clone().detach().cpu().numpy(), -35, axis=1)
#                new_images3 = new_images2[:, y1:y2, x1:x2] 
#                _, h, w, _  = new_images3.shape
#                indices = np.where(new_images3[0, ..., :3]<1)
#                row_ind = indices[0]
#                col_ind = indices[1]
#                channcel_ind = indices[2]
#                channel0 = np.where(channcel_ind == 0)
#                row_ind0 = row_ind[channel0]
#                col_ind0 = col_ind[channel0]
#                row_mean = np.mean(row_ind0)
#                col_mean = np.mean(col_ind0)
#                row_scale = 224/h
#                col_scale = 224/w
#                row_mean = row_mean*row_scale
#                col_mean = col_mean*col_scale
#                coord_points.append((row_mean, col_mean))
#                newarr = np.zeros((224, 224), dtype=np.float32)
#                score = self.human12_actionmask[label][i]
#                tmp_inds = np.where(newarr==0)
#                tmp_indices = np.array([tmp_inds[0],tmp_inds[1]]).transpose(1,0)
#                point = np.array([row_mean, col_mean])
#                if np.isnan(point[0]) or np.isnan(point[1]):
#                    continue
#                res = np.linalg.norm((tmp_indices-point), ord=alpha, axis=1, keepdims=True)
#                res = res.reshape(224,224)
#                attention_map = (score + gamma ) / (0.1+beta*res)
#                all_map.append(attention_map)
#            all_map = np.stack(all_map) 
#            all_map = np.sum(all_map, axis=0)
#            all_map = torch.tensor(all_map).unsqueeze(0)
        all_map = torch.sum(attention_map, 1)
        m = torch.nn.AvgPool2d(16, stride=16)
        output = m(all_map)
        score_indices_video = output.flatten(start_dim=1)
        return score_indices_video  
         
    def crop_smpl_skeleton(self, smpl_images):
        smpl_images = smpl_images[..., :3]
        masks = ~(smpl_images > 0.96).all(-1)
        coords = np.argwhere(masks.cpu().numpy().sum(axis=0))
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        return_yx = [y1, x1, y2, x2]
        cimgs = smpl_images[:, y1:y2, x1:x2] 
#        cimgs = []
#        for cimg in smpl_images[:, y1:y2, x1:x2]:
          #  print("crop size on smpl", cimg.shape)
          #  plt.figure(figsize=(10, 10))
          #  plt.imshow(cimg.clone().detach().cpu().numpy())
#            cimgs.append(cimg)
#        cimgs = torch.stack(cimgs)    
#        print(cimgs.shape)
#        ipdb.set_trace()
#        skeleton_images = skeleton_images.clone().detach().cpu().numpy()
#        skeleton_images = np.roll(skeleton_images[..., :3], -35, axis=1)
#        cimgs2 = []
#        for cimg in skeleton_images[:, y1:y2, x1:x2]:
          #  print("crop size on skeleton", cimg.shape)
          #  plt.figure(figsize=(10, 10))
          #  plt.imshow(cimg)
#            cimgs2.append(torch.tensor(cimg))
#        cimgs2 = torch.stack(cimgs2)
        #print("all cropped skeleton image", cimgs2.shape)
        return cimgs, return_yx

    def return_latent(self, batch, seed=None):
        distrib_param = self.encoder(batch)
        batch.update(distrib_param)
        return self.reparameterize(batch, seed=seed)
