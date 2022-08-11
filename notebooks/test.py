import torch
import random
import urllib.request
import imageio as imio
from einops import rearrange
import matplotlib.pyplot as plt
from IPython.display import clear_output, HTML, display

import sys
sys.path.append('..')

from models.model_utils import TrajectorySampler
from builders.builders import build_dataloader
from utils.utils import instantiate_from_config
from notebooks.walkthrough_utils import get_smooth_trajectory
from utils.camera_trajectory import rotate_n, go_forward, go_backward
import torchvision.transforms as transforms
from PIL import Image
import os

# download pretrained model (takes anywhere from a few seconds to a few minutes depending on internet speed)
# model_url = 'https://docs-assets.developer.apple.com/ml-research/models/gsn/replica_128x128.ckpt'
# checkpoint_filename = 'replica_128x128.ckpt'
# urllib.request.urlretrieve(model_url, checkpoint_filename)

# Note: adjust the data path here if data is in a different location than default
data_path = '/home/blackfoot/codes/ml-gsn/data/replica_all'
checkpoint_filename = '/home/blackfoot/codes/ml-gsn/logs/checkpoints/gsn-model-best-fid.ckpt'
checkpoint = torch.load(checkpoint_filename)
state_dict = checkpoint['state_dict']

# get rid of all the inception params which are leftover from FID metric
keys_for_deletion = []
for key in state_dict.keys():
    if 'fid' in key:
        keys_for_deletion.append(key)
for key in keys_for_deletion:
    del state_dict[key]

opt = checkpoint['opt']
opt.data_config.data_dir = data_path

# create dataloader
data_module = build_dataloader(opt.data_config, verbose=False)

# collect a set of real trajectories from the dataset
real_Rts = data_module.train_loader.dataset.get_trajectory_Rt()
trajectory_sampler = TrajectorySampler(real_Rts=real_Rts, mode=opt.model_config.params.trajectory_mode)

# initialize model and load state
gsn = instantiate_from_config(opt.model_config).cuda().eval()
gsn.set_trajectory_sampler(trajectory_sampler=trajectory_sampler)
gsn.load_state_dict(state_dict, strict=True)

# increase nerf_out_res after loading model to generate higher res samples (i.e., implicit upsampling)
gsn.generator_config.params.nerf_out_res *= 2
# trade render speed for memory by rendering smaller patches at a time
gsn.patch_size = 32

# load a batch of data so that we can use the camera parameters
real_data = next(iter(data_module.train_dataloader()))

device = 'cuda'
z_dim = opt.model_config.params.decoder_config.params.z_dim
z = torch.randn(1, z_dim, device=device)

# intrinsic camera matrix from real data
K_current = real_data['K'].to(device)
# initialize extrinsic camera matrix at the center of the scene
Rt_current = torch.eye(4, device=device).view(1, 1, 4, 4)

datapath = os.path.join(data_path, "test")
episode_path = os.path.join(datapath, '01')
i = 5
_rgb = os.path.join(episode_path, str(i).zfill(3) + '_rgb.png')
resize_transform_rgb = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
ref_rgb = resize_transform_rgb(Image.open(_rgb))[:3][None]

trajectory = {'rgb': [], 'depth': [], 'Rt': [], 'K': []}
# plt.imshow(ref_rgb.cpu().detach().numpy()[0].transpose(1,2,0), interpolation='bilinear')
plt.imshow(Image.open(_rgb))
plt.axis('off')
plt.show()
while True:
    camera_params = {'K': K_current, 'Rt': Rt_current}

    with torch.no_grad():
        fake_rgb, fake_depth, Rt, K = gsn(z, ref_rgb=ref_rgb, ref_depth=torch.zeros_like(ref_rgb),camera_params=camera_params)

    trajectory['rgb'].append(fake_rgb)
    trajectory['depth'].append(fake_depth)
    trajectory['Rt'].append(Rt)
    trajectory['K'].append(K)

    rgb_current = rearrange(fake_rgb, 'b t c h w -> (b t h) w c').cpu()

    clear_output()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(rgb_current, interpolation='bilinear')
    ax.set_title('Current observation');
    plt.axis('off')
    plt.show()

    action = input("a/w/d/s")
    if action == 'a':
        # Turn left
        Rt = rotate_n(n=-30.0).to(device)
        Rt_current = torch.bmm(Rt.unsqueeze(0), Rt_current[0]).unsqueeze(0)
    if action == 'd':
        # Turn right
        Rt = rotate_n(n=30.0).to(device)
        Rt_current = torch.bmm(Rt.unsqueeze(0), Rt_current[0]).unsqueeze(0)
    if action == 'w':
        # Go forward
        Rt_current = go_forward(Rt_current, step=opt.model_config.params.voxel_size / 0.6)
    if action == 's':
        # Go backward
        Rt_current = go_backward(Rt_current, step=opt.model_config.params.voxel_size / 0.6)
    if action == 'q':
        break

for key in trajectory.keys():
    trajectory[key] = torch.cat(trajectory[key], dim=1)

# jitter camera pose a tiny amount to make sure each pose is unique
# (to avoid problems with trajectory smoothing)
trajectory['Rt'] = trajectory['Rt'] + torch.rand_like(trajectory['Rt']) * 1e-5

# fit a smooth spline to the trajectory keypoints
n_keypoints = len(trajectory['Rt'][0])
new_Rts = get_smooth_trajectory(Rt=trajectory['Rt'][0], n_frames=5 * n_keypoints, subsample=3)
n_steps = len(new_Rts)

# render frames along new smooth trajectory
resampled_trajectory = {'rgb': [], 'depth': [], 'Rt': [], 'K': []}
for i in range(n_steps):
    if i % 10 == 0:
        print('Rendering frame {}/{}'.format(i, n_steps))

    camera_params = {'K': K_current[:1, :1], 'Rt': new_Rts[i:i + 1].to(device)}

    with torch.no_grad():
        fake_rgb, fake_depth, Rt, K = gsn(z, camera_params=camera_params)

    resampled_trajectory['rgb'].append(fake_rgb)
    resampled_trajectory['depth'].append(fake_depth)
    resampled_trajectory['Rt'].append(Rt)
    resampled_trajectory['K'].append(K)

for key in resampled_trajectory.keys():
    resampled_trajectory[key] = torch.cat(resampled_trajectory[key], dim=1)

imgs = []
for i in range(n_steps):
    im = resampled_trajectory['rgb'][0, i].permute(1, 2, 0).detach().cpu()
    imgs.append((im * 255).byte())

# save gif with random unique name
n = ''.join(random.choice('0123456789ABCDEF') for i in range(5))
animation_filename = 'camera_trajectory_{}.gif'.format(n)
print('Saving animation to {}\n'.format(animation_filename))
imio.mimsave(animation_filename, imgs, fps=30)

display(HTML('<img src={}>'.format(animation_filename)))