"""Generate GIF using pretrained network pickle."""

import os

import click
import dnnlib
import numpy as np
from PIL import Image
import torch

import legacy

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed', help='Random seed', default=0, type=int)
@click.option('--num', help='Number of samples', default=5, type=int)
@click.option('--resolution', help='Resolution of the output images', default=128, type=int)
@click.option('--num-phases', help='Number of phases', default=5, type=int)
@click.option('--transition-frames', help='Number of transition frames per phase', default=10, type=int)
@click.option('--static-frames', help='Number of static frames per phase', default=5, type=int)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--output', type=str)

@click.option('--fix-content', '--fc', help='Use fixed z_object', type=click.BOOL, default=False, is_flag = True)
@click.option('--cond', help = 'conditional, set a label or "all"', type=str, default = 'none')


def generate_gif(
    network_pkl: str,
    seed: int,
    num: int,
    resolution: int,
    num_phases: int,
    transition_frames: int,
    static_frames: int,
    truncation_psi: float,
    noise_mode: str,
    output: str,
    fix_content: bool,
    cond: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    transfer = (G.transfer != 'none')
    if not transfer:
        print('Must be a transfer model.')
        exit(1)

    if output is None:
        assert network_pkl[-4:] == '.pkl'
        kimg = network_pkl[-10:-4]
        output = os.path.join(os.path.dirname(network_pkl), f'itp{kimg}.gif' if not fix_content else f'itp{kimg}_fc.gif')

    outdir = os.path.dirname(output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    np.random.seed(seed)

    output_seq = []
    if cond == 'all':
        num = G.c_dim
        
    batch_size = num
    latent_size = G.z_dim
    latents = [np.random.randn(batch_size, latent_size) if cond != 'all' else np.random.randn(1, latent_size).repeat(batch_size, 0) for _ in range(num_phases)]
    if transfer: 
        latents_defect = [np.random.randn(batch_size, latent_size) if cond != 'all' else np.random.randn(1, latent_size).repeat(batch_size, 0) for _ in range(num_phases)]

    if cond == 'all':
        num_c = G.c_dim
        cond_list = [np.diag([1 for _ in range(num_c)]) for _ in range(num_phases)]
    elif cond != 'none':
        num_c = G.c_dim
        c_label = int(cond)
        c_npy = np.zeros(num_c)
        c_npy[c_label] = 1
        cond_list = [c_npy.reshape(1, -1).repeat(batch_size, 0) for _ in range(num_phases)]


    def to_image_grid(outputs):
        canvas = []
        for output in outputs:
            output = np.reshape(output, [num, *output.shape[1:]])
            output = np.concatenate(output, axis=1)
            canvas.append(output)
        canvas = np.concatenate(canvas, axis = 0)
        return Image.fromarray(canvas).resize((resolution * num, resolution * len(outputs)), Image.ANTIALIAS)


    def transfer_generate(dlatents, defectlatents):
        images, masks = G.synthesis(dlatents, defectlatents, noise_mode=noise_mode, output_mask=True)
        masks = masks.repeat((1, 3, 1, 1))
        rounded_masks = masks.clone()
        rounded_masks[rounded_masks >= G.mask_threshold] = 1.0
        rounded_masks[rounded_masks < G.mask_threshold] = -1.0
  
        images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        masks = (masks.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        rounded_masks = (rounded_masks.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        
        return to_image_grid([images, rounded_masks, masks])

    
    for i in range(num_phases):
        dlatents0 = G.mapping(torch.from_numpy(latents[i - 1] if not fix_content else latents[0]).to(device), None)
        dlatents1 = G.mapping(torch.from_numpy(latents[i] if not fix_content else latents[0]).to(device), None)
        defectlatents0 = G.defect_mapping(torch.from_numpy(latents_defect[i - 1]).to(device), None if cond == 'none' else torch.from_numpy(cond_list[i - 1]).to(device))
        defectlatents1 = G.defect_mapping(torch.from_numpy(latents_defect[i]).to(device), None if cond == 'none' else torch.from_numpy(cond_list[i]).to(device))
        for j in range(transition_frames):
            dlatents = (dlatents0 * (transition_frames - j) + dlatents1 * j) / transition_frames
            defectlatents = (defectlatents0 * (transition_frames - j) + defectlatents1 * j) / transition_frames
            output_seq.append(transfer_generate(dlatents, defectlatents))
        output_seq.extend([transfer_generate(dlatents, defectlatents1)] * static_frames)
    
    if not output.endswith('.gif'):
        output += '.gif'
    output_seq[0].save(output, save_all=True, append_images=output_seq[1:], optimize=True, duration=100, loop=0)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_gif() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
