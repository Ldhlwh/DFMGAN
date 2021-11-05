# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--output', help='Where to save the output images', type=str, required=True, metavar='FILE')
@click.option('--cmp', help='Generate images for comparison', type=bool, metavar='BOOL', is_flag=True)
@click.option('--gen-mask', help='Generate masks along with images', type=bool, metavar='BOOL', is_flag=True)


def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    output: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    cmp: bool,
    gen_mask: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    #os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        if cmp:
            seeds = [x for x in range(10)]
        else:
            seeds = [x for x in range(100)]
        #ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    os.makedirs(output)
    if cmp:
        canvas = []
        for seed_idx, seed in tqdm(enumerate(seeds)):
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            if hasattr(G, 'transfer'):
                transfer = (G.transfer != 'none')
            else: 
                transfer = False

            if transfer:
                defect_z = torch.from_numpy(np.random.RandomState(seed + len(seeds)).randn(1, G.z_dim)).to(device)
                ws = G.mapping(z, None)
                defect_ws = G.defect_mapping(defect_z, label, truncation_psi=truncation_psi)
                if G.transfer in ['res_block', 'res_block_match_dis', 'res_block_uni_dis']:
                    img, mask = G.synthesis(ws, defect_ws, noise_mode=noise_mode, output_mask = True, fix_residual_to_zero = False)
                    good_img = G.synthesis(ws, defect_ws, noise_mode=noise_mode, output_mask = False, fix_residual_to_zero = True)
                    mask = torch.where(mask >= 0.0, 1.0, -1.0)
                    img = torch.cat([good_img, img, mask.repeat((1, 3, 1, 1))], dim = 2)
                else:
                    img = G.synthesis(ws, defect_ws, noise_mode=noise_mode, fix_residual_to_zero = False)
                    good_img = G.synthesis(ws, defect_ws, noise_mode=noise_mode, fix_residual_to_zero = True)
                    mask = torch.where(mask >= 0.0, 1.0, -1.0)
                    img = torch.cat([good_img, img], dim = 2)
            else:
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            canvas.append(img)
        img = torch.cat(canvas, dim = 3)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        if not output.endswith('.png'):
            output += '.png'
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{output}')

    else:
        for seed_idx, seed in tqdm(enumerate(seeds)):
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            if hasattr(G, 'transfer'):
                transfer = (G.transfer != 'none')
            else: 
                transfer = False
            mask = None
            if transfer:
                defect_z = torch.from_numpy(np.random.RandomState(seed + len(seeds)).randn(1, G.z_dim)).to(device)
                ws = G.mapping(z, None)
                defect_ws = G.defect_mapping(defect_z, label, truncation_psi=truncation_psi)
                if G.transfer in ['res_block', 'res_block_match_dis', 'res_block_uni_dis']:
                    img, mask = G.synthesis(ws, defect_ws, noise_mode=noise_mode, output_mask = True, fix_residual_to_zero = False)
                    mask = torch.where(mask >= 0.0, 1.0, -1.0).repeat(1, 3, 1, 1)
                else:
                    img = G.synthesis(ws, defect_ws, noise_mode=noise_mode, fix_residual_to_zero = False)
                    mask = torch.where(mask >= 0.0, 1.0, -1.0).repeat(1, 3, 1, 1)
            else:
                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)

            img = ((img.permute(0, 2, 3, 1) + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(os.path.join(output, '%d_img.png' % seed_idx))
            if gen_mask and (mask is not None):
                mask = ((mask.permute(0, 2, 3, 1) + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(mask[0].cpu().numpy(), 'RGB').save(os.path.join(output, '%d_mask.png' % seed_idx))


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
