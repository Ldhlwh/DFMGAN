import lpips, torch
import torchvision as tv
import sys, os
import itertools
from tqdm import tqdm

loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
#loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

dir = sys.argv[1]

img_list = []
for f in os.listdir(dir):
    if dir.startswith('gen_img') and f.endswith('_mask.png'):
        continue
    img = tv.io.read_image(os.path.join(dir, f))
    assert img.ndim == 3 and img.shape[0] == 3
    img = img / 127.5 - 1.0
    img = img.unsqueeze(0)
    img_list.append(img.cuda())

n = len(img_list)
#assert n == 100
n_pair = n * (n - 1) // 2
lpips_list = []

for img1, img2 in tqdm(itertools.combinations(img_list, 2)):
    d = loss_fn_alex(img1, img2)
    lpips_list.append(d.item())

print('Averaged LPIPS over %d images (%d pairs) in %s: %f (min: %f, max: %f)' % (n, n_pair, dir, sum(lpips_list) / n_pair, min(lpips_list), max(lpips_list)))



'''
import torch
img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,64,64)
d = loss_fn_alex(img0, img1)
'''