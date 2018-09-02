from ..logger import info, warn
from ..metrics import eval_psnr_and_ssim
from ..utils import print_network, tensor2im
from .generators import ProSR
from bisect import bisect_left
from collections import OrderedDict

import os
import torch
import numpy as np

class SimultaneousMultiscaleTrainer(object):
    """multiscale training without curriculum scheduling"""
    def __init__(self,
                opt,
                training_dataset,
                save_dir="data/checkpoints",
                resume_from=None):
        super(SimultaneousMultiscaleTrainer, self).__init__()
        self.opt = opt
        self.save_dir = save_dir
        self.training_dataset = training_dataset
        self.start_epoch = 0
        self.progress = self.start_epoch / opt.train.epochs
        self.blend = 1
        # training variables
        self.input = torch.zeros(
            opt.train.batch_size, 3, 48, 48,
            dtype=torch.float32).cuda(non_blocking=True)
        self.label = torch.zeros_like(
            self.input, dtype=torch.float32).cuda(non_blocking=True)
        self.interpolated = torch.zeros_like(
            self.label, dtype=torch.float32).cuda(non_blocking=True)
        # for evaluation
        self.best_eval = OrderedDict(
            [('psnr_x%d' % s, 0.0) for s in opt.data.scale])
        self.eval_dict = OrderedDict(
            [('psnr_x%d' % s, []) for s in opt.data.scale])

        self.tensor2im = lambda t: tensor2im(t, mean=training_dataset.dataset.mean,
                                             stddev=training_dataset.dataset.stddev)

        opt.G.max_scale = max(opt.data.scale)

        ######### create generator and optimizer  #########
        self.net_G = ProSR(**opt.G).cuda()
        self.best_epoch = 0

        ######## Multi GPU #######
        # TODO: doesn't work for ProSRs
        if torch.cuda.device_count() > 1:
            self.net_G = torch.nn.DataParallel(self.net_G)

        self.optimizer_G = torch.optim.Adam(
            [p for p in self.net_G.parameters() if p.requires_grad],
            lr=self.opt.train.lr,
            betas=(0.9, 0.999),
            eps=1.0e-08)
        self.lr = self.opt.train.lr

        if resume_from:
            try:
                self.load_network(self.net_G, 'G', resume_from)
                self.load_optimizer(self.optimizer_G, 'G', resume_from)
                info('Set lr = %e' % self.lr)
            except Exception as e:
                warn("Error loading pretrained network. " + str(e))
                exit(0)

        ########### define loss functions  ############
        self.l1_criterion = torch.nn.L1Loss()

        # print('---------- Networks initialized -------------')
        # print_network(self.net_G)
        # print('-----------------------------------------------')

    def name(self):
        return 'SimultaneousMultiscaleTrainer'

    def set_train(self):
        self.net_G.train()
        self.isTrain = True

    def set_eval(self):
        self.net_G.eval()
        self.isTrain = False

    def set_input(self, **kwargs):
        lr = kwargs['input']
        hr = kwargs['target']
        interpolated = kwargs['bicubic']

        self.input.resize_(lr.size()).copy_(lr)
        self.label.resize_(hr.size()).copy_(hr)
        self.interpolated.resize_(interpolated.size()).copy_(interpolated)
        self.model_scale = kwargs['scale'][0].item()

    def forward(self):
        self.output = self.net_G(
            self.input, upscale_factor=self.model_scale) + self.interpolated

    def evaluate(self):
        if isinstance(self.net_G, torch.nn.DataParallel):
            # TODO: fix this silly way to pass 1 instance to multiple gpu
            # (self.net_G.module.forward wouldn't work)
            self.output = self.net_G(
                torch.cat(
                    [self.input for _ in range(torch.cuda.device_count())]),
                upscale_factor=self.model_scale,
                blend=self.blend)[:1] + self.interpolated
        else:
            self.output = self.net_G(
                self.input, upscale_factor=self.model_scale,
                blend=self.blend) + self.interpolated

        im1 = self.tensor2im(self.label)
        im2 = self.tensor2im(self.output)
        eval_res = {
            'psnr_x%d' % self.model_scale:
            eval_psnr_and_ssim(im1, im2, self.model_scale)[0]
        }
        for k, v in eval_res.items():
            self.eval_dict[k].append(v)
        return eval_res

    def backward(self):
        self.compute_loss()
        self.loss.backward()

    def compute_loss(self):
        self.loss = 0
        self.l1_loss = self.l1_criterion(self.output, self.label)
        self.loss += self.l1_loss

    def optimize_parameters(self):
        """call this function after forward()"""
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

    def get_current_visuals(self):
        disp = OrderedDict()
        disp['input'] = self.tensor2im(self.input.detach())
        disp['output'] = self.tensor2im(self.output.detach())
        disp['label'] = self.tensor2im(self.label.detach())
        return disp

    def get_current_errors(self):
        d = OrderedDict()
        for s in self.opt.data.scale:
            if self.opt.train.l1_loss_weight > 0:
                d['l1_x%d' % s] = np.nan

        if hasattr(self, 'l1_loss'):
            d['l1_x%d' % self.model_scale] = self.l1_loss.item()

        return d

    def save(self, epoch_label, epoch, lr):
        self.save_network(self.net_G, 'G', epoch_label)
        self.save_optimizer(self.optimizer_G, 'G', epoch_label, epoch, lr)

    def get_current_eval_result(self):
        eval_result = OrderedDict()
        for k, vs in self.eval_dict.items():
            eval_result[k] = 0
            if vs:
                for v in vs:
                    eval_result[k] += v
                eval_result[k] /= len(vs)
        return eval_result

    def reset_eval_result(self):
        for k in self.eval_dict:
            self.eval_dict[k].clear()

    def update_best_eval_result(self, epoch, current_eval_result=None):
        if current_eval_result is None:
            eval_result = self.get_current_eval_result()
        else:
            eval_result = current_eval_result
        is_best_sofar = any(
            [np.round(eval_result[k],2) > np.round(v,2) for k, v in self.best_eval.items()])
        if is_best_sofar:
            self.best_epoch = epoch
            self.best_eval = {
                k: max(self.best_eval[k], eval_result[k])
                for k in self.best_eval
            }

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        network = network.module if isinstance(
            network, torch.nn.DataParallel) else network
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        to_save = {
            'state_dict': network.state_dict(),
            'params': self.opt,
            'class_name': network.class_name(),
        }
        torch.save(to_save, save_path)

    def load_network(self, network, network_label, epoch_label):
        network = network.module if isinstance(
            network, torch.nn.DataParallel) else network
        save_path = '%s_net_%s.pth' % (epoch_label, network_label)
        loaded_state = torch.load(save_path)['state_dict']
        loaded_param_names = set(loaded_state.keys())

        # allow loaded states to contain keys that don't exist in current model
        # by trimming these keys;
        own_state = network.state_dict()
        extra = loaded_param_names - set(own_state.keys())
        if len(extra) > 0:
            print('Dropping ' + str(extra) + ' from loaded states')
        for k in extra:
            del loaded_state[k]

        try:
            network.load_state_dict(loaded_state)
        except KeyError as e:
            print(e)
        info('loaded network state from ' + save_path)

    def save_optimizer(self, optimizer, label, epoch_label, epoch, lr):
        save_filename = '%s_optim_%s.pth' % (epoch_label, label)
        save_path = os.path.join(self.save_dir, save_filename)

        to_save = {
            'state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'lr': lr
        }
        torch.save(to_save, save_path)

    def load_optimizer(self, optimizer, label, epoch_label):
        save_path = '%s_optim_%s.pth' % (epoch_label, label)

        data = torch.load(save_path)
        loaded_state = data['state_dict']
        optimizer.load_state_dict(loaded_state)

        # Load more params
        self.start_epoch = data['epoch']
        self.lr = data['lr']

        info('loaded optimizer state from ' + save_path)

    def set_learning_rate(self, lr, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_learning_rate(self):
        """update learning rate with exponential decay"""
        lr = self.lr * self.opt.train.lr_decay
        if lr < self.opt.train.smallest_lr:
            return
        self.set_learning_rate(lr, self.optimizer_G)
        info('update learning rate: %f -> %f' % (self.lr, lr))
        self.lr = lr


class CurriculumLearningTrainer(SimultaneousMultiscaleTrainer):
    def __init__(self,
                opt,
                training_dataset,
                save_dir="data/checkpoints",
                resume_from=None):
        super(CurriculumLearningTrainer, self).__init__(opt, training_dataset, save_dir, resume_from)
        self.reset_curriculum_for_dataloader()

    def reset_curriculum_for_dataloader(self):
        """ set data loader to load correct scales"""
        assert len(
            self.opt.train.growing_steps) == len(self.opt.data.scale) * 2 - 1
        self.current_scale_idx = (
            bisect_left(self.opt.train.growing_steps, self.progress) + 1) // 2
        self.net_G.current_scale_idx = self.current_scale_idx
        training_scales = self.opt.data.scale[:(self.current_scale_idx + 1)]
        self.training_dataset.random_vars.clear()
        for s in training_scales:
            self.training_dataset.random_vars.append(s)
        info('start training with scales: {}'.format(str(training_scales)))

    def name(self):
        return 'CurriculumLearningTrainer'

    def set_input(self, **kwargs):
        lr = kwargs['input']
        hr = kwargs['target']
        interpolated = kwargs['bicubic']

        self.input.resize_(lr.size()).copy_(lr)
        self.label.resize_(hr.size()).copy_(hr)
        self.interpolated.resize_(interpolated.size()).copy_(interpolated)
        self.model_scale = kwargs['scale'][0].item()

    def forward(self):
        if self.current_scale_idx != 0:
            lo, hi = self.opt.train.growing_steps[self.current_scale_idx * 2 -
                                                  2:self.current_scale_idx * 2]
            self.blend = min((self.progress - lo) / (hi - lo), 1)
            assert self.blend >= 0 and self.blend <= 1
        else:
            self.blend = 1
        self.output = self.net_G(
            self.input, upscale_factor=self.model_scale,
            blend=self.blend) + self.interpolated

    def optimize_parameters(self):
        """call this function after forward()"""
        super().optimize_parameters()
        # update progress
        self.increment_training_progress()

    def increment_training_progress(self):
        """increment self.progress and D, G scale_idx"""
        self.progress += 1 / len(self.training_dataset) / self.opt.train.epochs
        if self.progress > self.opt.train.growing_steps[self.current_scale_idx
                                                        * 2]:
            if self.current_scale_idx < len(self.opt.data.scale) - 1:
                self.current_scale_idx += 1
                self.net_G.current_scale_idx = self.current_scale_idx

                training_scales = [
                    self.opt.data.scale[i]
                    for i in range(self.current_scale_idx + 1)
                ]
                self.training_dataset.random_vars.clear()
                for s in training_scales:
                    self.training_dataset.random_vars.append(s)
                info('start training with scales: {}'.format(
                    str(training_scales)))
