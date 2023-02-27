import lpips, torch
import itertools
import numpy as np
import dnnlib
from tqdm import tqdm
import copy

def compute_clpips(opts, num_gen):
    dataset_kwargs = opts.dataset_kwargs
    device = opts.device
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(device)

    with torch.no_grad():
        loss_fn_alex = lpips.LPIPS(net='alex', verbose = opts.progress.verbose).to(device) # best forward scores
        #loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

        data_list = []
        dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
        for img, _labels in torch.utils.data.DataLoader(dataset=dataset, batch_size=64, **data_loader_kwargs):
            if img.shape[1] == 1:
                img = img.repeat([1, 3, 1, 1])
            if img.shape[1] == 4:
                img = img[:, :3, :, :]
            data_list.append(img.to(device))
        data_list = torch.cat(data_list, dim = 0)

        cluster = [[] for _ in range(data_list.shape[0])]
        label = torch.zeros([1, G.c_dim], device=device)

        iterator = tqdm(range(num_gen), desc = 'Clustering') if opts.progress.verbose else range(num_gen)
        for seed in iterator:
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, defect_z = z, truncation_psi = 1, noise_mode = 'const')
            score_list = loss_fn_alex(img.repeat(data_list.shape[0], 1, 1, 1), data_list)
            #score_list = np.array([loss_fn_alex(img, data).item() for data in data_list])
            closest_index = score_list.argmin().item()
            if len(cluster[closest_index]) < 200:
                cluster[closest_index].append(img)

        cluster_lpips = []
        iterator = tqdm(cluster, desc = 'Computing clustered LPIPS') if opts.progress.verbose else cluster
        for c in iterator:
            # c_lpips = []
            # for img1, img2 in itertools.combinations(c, 2):
            #     d = loss_fn_alex(img1, img2)
            #     c_lpips.append(d.item())
            # if len(c_lpips) == 0:
            #     cluster_lpips.append(0.0)
            # else:
            #     cluster_lpips.append(sum(c_lpips) / len(c_lpips))
            if len(c) <= 1:
                cluster_lpips.append(0.0)
                continue
            c_lpips = 0.0
            img = torch.cat(c, dim = 0)
            ref_img = img.clone()
            for _ in range(img.shape[0] - 1):
                img = torch.cat([img[1:], img[0:1]], dim = 0)
                c_lpips += loss_fn_alex(img, ref_img).sum().item()
            cluster_lpips.append(c_lpips / (img.shape[0] * (img.shape[0] - 1)))

    if opts.progress.verbose:
        print('Cluster Statistics:')
        print([(len(cluster[i]), '%.4f' % cluster_lpips[i]) for i in range(len(data_list))])

    clpips = sum(cluster_lpips) / len(cluster_lpips)
    rz_sum = 0.0
    n = 0
    for score in cluster_lpips:
        if score != 0.0:
            rz_sum += score
            n += 1
    clpips_rz = rz_sum / n
    return clpips, clpips_rz
