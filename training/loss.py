# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, lambda_match, lambda_ms, mode_seek, D_match = None, augment_pipe=None, G_defect_mapping = None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, 
            transfer=None):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.D_match = D_match
        if transfer == 'res_block_match_dis':
            assert self.D_match is not None
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

        self.G_defect_mapping = G_defect_mapping
        self.transfer = transfer
        self.lambda_match = lambda_match
        self.lambda_ms = lambda_ms
        self.mode_seek = mode_seek

        self.phases_printed = False

    def run_G(self, z, c, sync, defect_z = None, transfer = None, output_mask = False, mode_seek = 'none'):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

        if transfer is not None:
            with misc.ddp_sync(self.G_defect_mapping, sync):
                defect_ws = self.G_defect_mapping(defect_z, c)
                if self.style_mixing_prob > 0:
                    with torch.autograd.profiler.record_function('style_mixing'):
                        defect_cutoff = torch.empty([], dtype=torch.int64, device=defect_ws.device).random_(1, defect_ws.shape[1])
                        defect_cutoff = torch.where(torch.rand([], device=defect_ws.device) < self.style_mixing_prob, defect_cutoff, torch.full_like(defect_cutoff, defect_ws.shape[1]))
                        defect_ws[:, defect_cutoff:] = self.G_defect_mapping(torch.randn_like(defect_z), c, skip_w_avg_update=True)[:, defect_cutoff:]
        
        with misc.ddp_sync(self.G_synthesis, sync):
            input_list = [ws]
            if transfer is None:
                img = self.G_synthesis(ws)
            elif transfer == 'dual_mod':
                ws += defect_ws
                img = self.G_synthesis(ws)
            elif transfer == 'res_block':
                img = self.G_synthesis(ws, defect_ws)
                input_list.append(defect_ws)
            elif transfer == 'res_block_match_dis':
                if output_mask:
                    img, mask = self.G_synthesis(ws, defect_ws, output_mask = output_mask)
                else:
                    img = self.G_synthesis(ws, defect_ws, output_mask = output_mask)
                input_list.append(defect_ws)

                if mode_seek == 'w/mask':
                    half_batch = ws.shape[0] // 2
                    _, half_mask = self.G_synthesis(ws[:half_batch], defect_ws[half_batch:], output_mask = True)
        
        if transfer == 'res_block_match_dis' and output_mask:
            if mode_seek == 'w/mask':
                return img, mask, half_mask, input_list
            return img, mask, input_list
        else:
            return img, input_list

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def run_D_match(self, img_mask, c, sync):
        #if self.augment_pipe is not None:
        #    img = self.augment_pipe(img)
        with misc.ddp_sync(self.D_match, sync):
            logits = self.D_match(img_mask, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, gen_defect_z = None, real_mask = None, mask_threshold = 0.0):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'D_matchmain', 'D_matchreg', 'D_matchboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_D_matchmain = (phase in ['D_matchmain', 'D_matchboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        do_D_matchr1 = (phase in ['D_matchreg', 'D_matchboth']) and (self.r1_gamma != 0)
        
        # print({
        #     'do_Gmain': do_Gmain,
        #     'do_Dmain': do_Dmain,
        #     'do_D_matchmain': do_D_matchmain,
        #     'do_Gpl': do_Gpl,
        #     'do_Dr1': do_Dr1,
        #     'do_D_matchr1': do_D_matchr1,
        # })

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            if self.mode_seek != 'none':
                assert gen_z.shape[0] % 2 == 0
            with torch.autograd.profiler.record_function('Gmain_forward'):
                if self.transfer == 'res_block_match_dis':
                    if self.mode_seek == 'none':
                        gen_img, gen_mask, inputs = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl), defect_z = gen_defect_z, transfer = self.transfer, output_mask = True) # May get synced by Gpl.                       
                    elif self.mode_seek == 'w/mask':
                        gen_img, gen_mask, gen_half_mask, inputs = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl), defect_z = gen_defect_z, transfer = self.transfer, output_mask = True, mode_seek = self.mode_seek) # May get synced by Gpl.                       
                else:
                    gen_img, inputs = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl), defect_z = gen_defect_z, transfer = self.transfer) # May get synced by Gpl.

                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                if self.transfer == 'res_block_match_dis':
                    #gen_mask[gen_mask >= mask_threshold] = 1.0
                    gen_mask = torch.tanh(gen_mask)
                    #gen_mask[gen_mask < mask_threshold] = -1.0
                    gen_img_mask = torch.cat([gen_img, gen_mask], dim = 1)
                    gen_match_logits = self.run_D_match(gen_img_mask, gen_c, sync=False)
                    training_stats.report('Loss/scores/fake_match', gen_match_logits)
                    training_stats.report('Loss/signs/fake_match', gen_match_logits.sign())
                    loss_Gmain = loss_Gmain + self.lambda_match * torch.nn.functional.softplus(-gen_match_logits)

                    if self.mode_seek == 'w/mask':
                        assert len(inputs) == 2
                        assert gen_z.shape[0] % 2 == 0
                        half_batch_size = gen_z.shape[0] // 2
                        w = inputs[1]
                        w1, w2 = w[:half_batch_size], w[half_batch_size:]
                        mask1, mask2 = gen_mask[:half_batch_size], gen_half_mask
                        loss_MS = (w1 - w2).abs().mean() / (mask1 - mask2).abs().mean()
                        training_stats.report('Loss/mode_seek', loss_MS)
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain if self.mode_seek == 'none' else loss_Gmain + loss_MS).mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, input_list = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync, defect_z = gen_defect_z[:batch_size], transfer = self.transfer)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=input_list, create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        
        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False, defect_z = gen_defect_z, transfer = self.transfer)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # D_matchmain: Minimize matching logits for generated images&masks.
        loss_D_matchgen = 0
        if do_D_matchmain:
            with torch.autograd.profiler.record_function('D_matchgen_forward'):
                gen_img, gen_mask, _gen_ws = self.run_G(gen_z, gen_c, sync=False, defect_z = gen_defect_z, transfer = self.transfer, output_mask = True)
                #gen_mask[gen_mask >= mask_threshold] = 1.0
                gen_mask = torch.tanh(gen_mask)
                #gen_mask[gen_mask < mask_threshold] = -1.0
                gen_img_mask = torch.cat([gen_img, gen_mask], dim = 1)
                gen_logits = self.run_D_match(gen_img_mask, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake_match', gen_logits)
                training_stats.report('Loss/signs/fake_match', gen_logits.sign())
                loss_D_matchgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('D_matchgen_backward'):
                loss_D_matchgen.mean().mul(gain).backward()

        # D_matchmain: Maximize matching logits for real images&masks.
        # D_matchr1: Apply R1 regularization.
        if do_D_matchmain or do_D_matchr1:
            name = 'D_matchreal_Dr1' if do_D_matchmain and do_D_matchr1 else 'D_matchreal' if do_D_matchmain else 'D_matchr1'
            with torch.autograd.profiler.record_function(name + '_forward_match'):
                real_img_tmp = real_img.detach().requires_grad_(do_D_matchr1)
                real_mask_tmp = real_mask.detach().requires_grad_(do_D_matchr1)
                #assert real_mask_tmp.shape[0] % 2 == 0
                #wrong_mask_tmp = real_mask_tmp.flip(0)

                real_img_mask_tmp = torch.cat([real_img_tmp, real_mask_tmp], dim = 1)
                real_logits = self.run_D_match(real_img_mask_tmp, real_c, sync=sync)

                #wrong_img_mask_tmp = torch.cat([real_img_tmp, wrong_mask_tmp], dim = 1)
                #wrong_logits = self.run_D_match(wrong_img_mask_tmp, real_c, sync=sync)

                training_stats.report('Loss/scores/real_match', real_logits)
                training_stats.report('Loss/signs/real_match', real_logits.sign())
                #training_stats.report('Loss/scores/wrong_match', wrong_logits)
                #training_stats.report('Loss/signs/wrong_match', wrong_logits.sign())

                loss_D_matchreal = 0
                if do_D_matchmain:
                    loss_D_matchreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    #loss_D_matchreal = loss_D_matchreal + torch.nn.functional.softplus(wrong_logits)
                    training_stats.report('Loss/D_match/loss', loss_D_matchgen + loss_D_matchreal)

                loss_D_matchr1 = 0
                if do_D_matchr1:
                    with torch.autograd.profiler.record_function('r1_grads_match'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_mask_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_D_matchr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty_match', r1_penalty)
                    training_stats.report('Loss/D_match/reg', loss_D_matchr1)

            with torch.autograd.profiler.record_function(name + '_backward_match'):
                (real_logits * 0 + loss_D_matchreal + loss_D_matchr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
