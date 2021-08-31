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

@click.option('--fix-content', '--fc', type=click.BOOL, default=False, is_flag = True)


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
):
    """Generate gif using pretrained network pickle.

    Examples:

    \b
    python generate_gif.py --output=obama.gif --seed=0 --num-rows=1 --num-cols=8 \\
        --network=https://hanlab.mit.edu/projects/data-efficient-gans/models/DiffAugment-stylegan2-100-shot-obama.pkl
    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    transfer = (G.transfer != 'none')
    if not transfer:
        print('Must be a transfer model.')
        exit(1)

    assert network_pkl.split('/')[0].endswith('runs')
    assert network_pkl.split('/')[2].startswith('000')
    assert network_pkl.endswith('pkl')

    if output is None:
        kimg = network_pkl.split('/')[-1].split('.')[0].split('-')[-1]
        kimg = int(kimg)
        output = os.path.join('gifs', '%s_%d%s.gif' % (network_pkl.split('/')[1], kimg, '_fc' if fix_content else ''))
        print('Will save to %s' % output)

    outdir = os.path.dirname(output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    if (output.endswith('_fc') or output.endswith('_fc.gif')) and not fix_content:
        print('Automatically setting fix_content = True.')
        fix_content = True

    np.random.seed(seed)

    output_seq = []
    batch_size = num
    latent_size = G.z_dim
    latents = [np.random.randn(batch_size, latent_size) for _ in range(num_phases)]
    if transfer: 
        latents_defect = [np.random.randn(batch_size, latent_size) for _ in range(num_phases)]

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
        defectlatents0 = G.defect_mapping(torch.from_numpy(latents_defect[i - 1]).to(device), None)
        defectlatents1 = G.defect_mapping(torch.from_numpy(latents_defect[i]).to(device), None)
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
