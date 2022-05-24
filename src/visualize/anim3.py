import numpy as np
import torch
import imageio
import pdb
# from action2motion
# Define a kinematic tree for the skeletal struture
humanact12_kinematic_chain = [#[0, 1, 4, 7, 10],]
        #                      [0, 2, 5, 8, 11],
                           #   [0, 3, 6, 9, 12, 15],
                           [6, 9, 12, 15]]
        #                      [9, 13, 16, 18, 20, 22],
    #                          [9, 14, 17, 19, 21, 23]]  # same as smpl

smpl_kinematic_chain = humanact12_kinematic_chain

mocap_kinematic_chain = [[0, 1, 2, 3],
                         [0, 12, 13, 14, 15],
                         [0, 16, 17, 18, 19],
                         [1, 4, 5, 6, 7],
                         [1, 8, 9, 10, 11]]

vibe_kinematic_chain = [[0, 12, 13, 14, 15],
                        [0, 9, 10, 11, 16],
                        [0, 1, 8, 17],
                        [1, 5, 6, 7],
                        [1, 2, 3, 4]]

action2motion_kinematic_chain = vibe_kinematic_chain


def add_shadow(img, shadow=15):
    img = np.copy(img)
    mask = img > shadow
    img[mask] = img[mask] - shadow
    img[~mask] = 0
    return img


def load_anim(path, timesize=None):
    data = np.array(imageio.mimread(path, memtest=False))[..., :3]
    if timesize is None:
        return data
    # take the last frame and put shadow repeat the last frame but with a little shadow
    lastframe = add_shadow(data[-1])
    alldata = np.tile(lastframe, (timesize, 1, 1, 1))

    # copy the first frames
    lenanim = data.shape[0]
    alldata[:lenanim] = data[:lenanim]
    return alldata


def plot_3d_motion(motion, length, save_path, params, title="", interval=50):
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
    from matplotlib.animation import FuncAnimation, writers  # noqa: F401
    # import mpl_toolkits.mplot3d.axes3d as p3
    matplotlib.use('Agg')
    pose_rep = params["pose_rep"]
    #from remote_pdb import set_trace
    pdb.set_trace()
    fig = plt.figure(figsize=[4.6, 4.8])
    ax = fig.add_subplot(111, projection='3d')
    # ax = p3.Axes3D(fig)
    # ax = fig.gca(projection='3d')

    def init():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_zlim(-0.7, 0.7)

        ax.view_init(azim=-90, elev=110)
        # ax.set_axis_off()
        ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)
        ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.25)

#    colors = ['red', 'magenta', 'black', 'green', 'blue']
    colors = ['black']
    if pose_rep != "xyz":
        raise ValueError("It should already be xyz.")

    if torch.is_tensor(motion):
        motion = motion.numpy()

    # invert axis
    motion[:, 1, :] = -motion[:, 1, :]
    motion[:, 2, :] = -motion[:, 2, :]

    """
    Debug: to rotate the bodies
    import src.utils.rotation_conversions as geometry
    glob_rot = [0, 1.5707963267948966, 0]
    global_orient = torch.tensor(glob_rot)
    rotmat = geometry.axis_angle_to_matrix(global_orient)
    motion = np.einsum("ikj,ko->ioj", motion, rotmat)
    """

    if motion.shape[0] == 18:
        kinematic_tree = action2motion_kinematic_chain
    elif motion.shape[0] == 24:
        kinematic_tree = smpl_kinematic_chain
    else:
        kinematic_tree = None

    def update(index):
      #  pdb.set_trace()
        ax.lines = []
        ax.collections = []
        if kinematic_tree is not None:
            for chain, color in zip(kinematic_tree, colors):
                ax.plot(motion[chain, 0, index],
                        motion[chain, 1, index],
                        motion[chain, 2, index], linewidth=4.0, color=color)
        else:
            ax.scatter(motion[1:, 0, index], motion[1:, 1, index],
                       motion[1:, 2, index], c="red")
            ax.scatter(motion[:1, 0, index], motion[:1, 1, index],
                       motion[:1, 2, index], c="blue")

    ax.set_title(title)

    ani = FuncAnimation(fig, update, frames=length, interval=interval, repeat=False, init_func=init)

    plt.tight_layout()
    # pillow have problem droping frames
    ani.save(save_path, writer='ffmpeg', fps=1000/interval)
    plt.close()


def plot_3d_motion_dico(x):
    motion, length, save_path, params, kargs = x
    plot_3d_motion(motion, length, save_path, params, **kargs)



import os

import matplotlib.pyplot as plt
import torch
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
#from .visualize import viz_epoch

import src.utils.fixseed  # noqa

plt.switch_backend('agg')


def main():
    # parse options
    params, folder, checkpointname, epoch = parser()
    pdb.set_trace()
    model, datasets = get_model_and_data(params)
    dataset = datasets["train"]

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=params["device"])
    model.load_state_dict(state_dict)
    
    # visualize_params
  #  viz_epoch(model, dataset, epoch, parameters, folder=folder, writer=None)
    model.outputxyz = True
    #print(f"Visualization of the epoch {epoch}")
    noise_same_action = params["noise_same_action"]
    noise_diff_action = params["noise_diff_action"]
    duration_mode = params["duration_mode"]
    reconstruction_mode = params["reconstruction_mode"]
    decoder_test = params["decoder_test"]

    fact = params["fact_latent"]
    figname = params["figname"].format(epoch)

    nspa = params["num_samples_per_action"]
    nats = params["num_actions_to_sample"]

    num_classes = params["num_classes"]

    # define some classes
    classes = torch.randperm(num_classes)[:nats]

    meandurations = torch.from_numpy(np.array([round(dataset.get_mean_length_label(cl.item()))
                                               for cl in classes]))

    if duration_mode == "interpolate" or decoder_test == "diffduration":
        points, step = np.linspace(-nspa, nspa, nspa, retstep=True)
        points = np.round(10*points/step).astype(int)
        gendurations = meandurations.repeat((nspa, 1)) + points[:, None]
    else:
        gendurations = meandurations.repeat((nspa, 1))

    # extract the real samples
    real_samples, mask_real, real_lengths = dataset.get_label_sample_batch(classes.numpy())
    visualization = {"x": real_samples.to(model.device),
                     "y": classes.to(model.device),
                     "mask": mask_real.to(model.device),
                     "lengths": real_lengths.to(model.device),
                     "output": real_samples.to(model.device)}

    with torch.no_grad():
        visualization["output_xyz"] = model.rot2xyz(visualization["output"], visualization["mask"])

    fps = params["fps"]
  #  params = params.copy()
    if "output_xyz" in visualization:
        outputkey = "output_xyz"
        params["pose_rep"] = "xyz"
    else:
        outputkey = "poses"

    keep = [outputkey, "lengths", "y"]
    visu = {key: visualization[key].data.cpu().numpy() for key in keep}

    finalpath = os.path.join(folder, figname + ".gif")
    tmp_path = os.path.join(folder, f"subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)

    save_path_format = os.path.join(tmp_path, "visual_{}.gif")
    iterator = ( visu[outputkey][0],
                     visu["lengths"][0],
                     save_path_format.format(0),
                     params, {"title": f"real: {dataset.label_to_action_name(visu['y'][0])}", "interval": 1000/fps})
                 
    plot_3d_motion_dico(iterator)

if __name__ == '__main__':
    main()

