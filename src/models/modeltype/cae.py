import torch
import torch.nn as nn
from ..tools.losses import get_loss_function
from ..rotation2xyz import Rotation2xyz
import ipdb
#import src.models.mae.models_mae as models_mae
#from pytorch3d.structures import Pointclouds
#from pytorch3d.renderer import (
#    look_at_view_transform,
#    FoVOrthographicCameras,
#    PointsRasterizationSettings,
#    PointsRenderer,
#    PointsRasterizer,
#    AlphaCompositor,
#    RasterizationSettings,  
#    MeshRenderer, MeshRasterizer, 
#    HardPhongShader, PointLights, TexturesVertex,
#)
import numpy as np
#from pytorch3d.structures import Meshes
#faces = np.load(f"/scratch/work/liux17/experiments0310/exp_0320-multicards/ACTOR-master1/models/smpl/smplfaces.npy")
#faces = faces.astype(np.int64)
#faces = torch.from_numpy(faces)

#frames = np.linspace(0, 59, 10).astype(int)
import torchvision.transforms as transforms


class CAE(nn.Module):
    def __init__(self, encoder, decoder, device, lambdas, latent_dim, outputxyz, human12_actionmask, compositional_actions, 
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.outputxyz = outputxyz
        
        self.lambdas = lambdas
        
        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.device = device
        self.translation = translation
        self.jointstype = jointstype
        self.vertstrans = vertstrans
        
        self.losses = list(self.lambdas) + ["mixed"]
#        print("self.losses", self.losses)
        self.rotation2xyz = Rotation2xyz(device=self.device)
        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans}
        self.human12_actionmask = human12_actionmask
        self.compositional_actions = compositional_actions  
#        self.trans = transforms.Compose([transforms.Resize(size=(96,96))])      
#        self.R, self.T = look_at_view_transform(-20, 10, 0, up=((0, -180, 0), ), device=self.device)
#        cameras = FoVOrthographicCameras(device=self.device, R=self.R, T=self.T, znear=0.01) 
#        self.faces = torch.from_numpy(faces).to(self.device) 

#        raster_settings_phong = RasterizationSettings(image_size=128, faces_per_pixel=1)
#        rasterizer_phong = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_phong)
#        lights = PointLights(device=self.device, location=((2.0, 2.0, -2.0),))
#        self.phong_renderer = MeshRenderer(rasterizer=rasterizer_phong, shader=HardPhongShader(device=self.device, cameras=cameras, lights=lights) )        
#        self.mae_model = getattr(models_mae, 'mae_vit_large_patch16')()
#        self.mae_model.to(self.device)
#        chkpt_dir = '/scratch/work/liux17/experiments0310/exp_0320-multicards/ACTOR-master1-4/models/mae_visualize_vit_large_ganloss.pth'
#        checkpoint = torch.load(chkpt_dir, map_location=self.device)
#        self.mae_model.load_state_dict(checkpoint['model'], strict=False)
        
            
#    def skeleton_render(self, pointclouds):
#        batch_imgs = []
#        ipdb.set_trace()
#        for pointcloud in pointclouds:
#            pointcloud = pointcloud.permute(2,0,1)
#            video_imgs = []
#            for i in frames:
#                verts = pointcloud[i]
#                newverts = verts.float()
#                newrgb = np.array([[0,0,0.9,1]])
#                newrgb = np.repeat(newrgb, 24, axis=0)
#                newrgb = torch.Tensor(newrgb).to(self.device)
#                new_point_cloud = Pointclouds(points=[newverts], features=[newrgb])
#                new_images = self.renderer(new_point_cloud)
#                new_images = new_images.squeeze()[..., :3]
#                video_imgs.append(new_images)
#            video_imgs = torch.stack(video_imgs)
#            batch_imgs.append(video_imgs)
#        batch_imgs = torch.stack(batch_imgs)
#        return batch_imgs

    def smpl_render(self, meshes):
#        ipdb.set_trace()
        meshes = meshes.permute(0,3,1,2)
        meshes = meshes[:,frames]
        bn, fn, _, _ = meshes.shape
        meshes = meshes.reshape(bn*fn,6890,3)   
        num, _, _ = meshes.shape      
        meshes_rgb = torch.ones(meshes.shape)
        textures = TexturesVertex(verts_features=meshes_rgb.to(self.device))
        smpl_mesh = Meshes( verts=[meshes[i].to(self.device) for i in range(num)], faces=[self.faces.to(self.device) for i in range(num)],  textures=textures )
        batch_imgs = self.phong_renderer(meshes_world=smpl_mesh, R=self.R, T=self.T)
        _, s1, s2, c = batch_imgs.shape
        batch_imgs = batch_imgs.reshape(bn, fn, s1, s2, c)
        return batch_imgs

    def smpl_render2(self, meshes):
#        ipdb.set_trace()
        batch_imgs = []
        for mesh in meshes:
            mesh = mesh.permute(2,0,1)
            video_imgs = []
            for i in frames:
                verts = mesh[i]
                verts_rgb = torch.ones(verts.shape)[None]
                textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
                smpl_mesh = Meshes(
                        verts=[verts.to(self.device)],
                        faces=[self.faces.to(self.device)],
                        textures=textures
                        )
                img = self.phong_renderer(meshes_world=smpl_mesh, R=self.R, T=self.T)
                img = img.squeeze()[..., :3]
                video_imgs.append(img)
            video_imgs = torch.stack(video_imgs)
            batch_imgs.append(video_imgs)
        batch_imgs = torch.stack(batch_imgs)
        return batch_imgs 

    def rot2xyz(self, x, mask, **kwargs):
        kargs = self.param2xyz.copy()
        kargs.update(kwargs)
        return self.rotation2xyz(x, mask, **kargs)
    
    def forward(self, batch):
        if self.outputxyz:
            batch["x_xyz"] = self.rot2xyz(batch["x"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]
        # encode
        batch.update(self.encoder(batch))
        # decode
        batch.update(self.decoder(batch))
        # if we want to output xyz
        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch

    def compute_loss(self, batch):
#        ipdb.set_trace()
        mixed_loss = 0
        losses = {}
        for ltype, lam in self.lambdas.items():
            loss_function = get_loss_function(ltype)
            loss = loss_function(self, batch)
            mixed_loss += loss*lam
            losses[ltype] = loss.item()
        losses["mixed"] = mixed_loss.item()
        return mixed_loss, losses

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def generate_one(self, cls, duration, fact=1, xyz=False):
        y = torch.tensor([cls], dtype=int, device=self.device)[None]
        lengths = torch.tensor([duration], dtype=int, device=self.device)
        mask = self.lengths_to_mask(lengths)
        z = torch.randn(self.latent_dim, device=self.device)[None]
        
        batch = {"z": fact*z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        if not xyz:
            return batch["output"][0]
        
        output_xyz = self.rot2xyz(batch["output"], batch["mask"])

        return output_xyz[0]
            
    def generate(self, classes, durations, nspa=1,
                 noise_same_action="random", noise_diff_action="random",
                 fact=1):
        if nspa is None:
            nspa = 1
      #  nats = len(classes)
        nats = 1   
        classes = classes.unsqueeze(1)
        durations = durations[0].unsqueeze(0)
        y = classes.to(self.device).repeat(1, nspa)  # (view(nspa, nats))
        ipdb.set_trace()
        if len(durations.shape) == 1:
            lengths = durations.to(self.device).repeat(nspa)
        else:
            lengths = durations.to(self.device).reshape(y.shape)
        
        mask = self.lengths_to_mask(lengths)
        
        if noise_same_action == "random":
            if noise_diff_action == "random":
                z = torch.randn(nspa*nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_same_action = torch.randn(nspa, self.latent_dim, device=self.device)
                z = z_same_action.repeat_interleave(nats, axis=0)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
        elif noise_same_action == "interpolate":
            if noise_diff_action == "random":
                z_diff_action = torch.randn(nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(nats, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            interpolation_factors = torch.linspace(-1, 1, nspa, device=self.device)
            z = torch.einsum("ij,k->kij", z_diff_action, interpolation_factors).view(nspa*nats, -1)
        elif noise_same_action == "same":
            if noise_diff_action == "random":
                z_diff_action = torch.randn(nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(nats, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            z = z_diff_action.repeat((nspa, 1))
        else:
            raise NotImplementedError("Noise same action must be random, same or interpolate.")
#        y1 = y[0] 
#        y2 = y[1]
        alphas = torch.tensor([0.5])
        alphas = alphas.to(self.device).repeat(nspa) 
        batch = {"z": fact*z, "y": y.squeeze(), "mask": mask, "lengths": lengths, "alpha":alphas} 
   #     batch = {"z": fact*z, "y": y.squeeze(), "mask": mask, "lengths": lengths}
#        batch = {"z": fact*z, "y1":y1, "y2":y2, "mask":mask, "lengths":lengths, "alphas":alphas}
      #  z, y1, y2, mask, lengths, alphas = batch["z"], batch["y1"], batch["y2"], batch["mask"], batch["lengths"], batch["alphas"]
        batch = self.decoder(batch)
        
        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        
        return batch
    
    def return_latent(self, batch, seed=None):
        return self.encoder(batch)["z"]
