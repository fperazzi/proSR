import os
from collections import OrderedDict

import torch

from ..config import phase
from ..logger import info
from ..misc.util import tensor2im


class BaseModel():

    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.phase == phase.TRAIN
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available(
        ) else torch.FloatTensor
        self.save_dir = os.path.join(opt.checkpoint_dir)
        if not opt.eval.scale:
            opt.eval.scale = opt.scale
        self.eval_func = OrderedDict([('psnr_x%d' % s,
                                       lambda x, y, scale=s: psnr(x, y, scale))
                                      for s in opt.eval.scale])
        self.best_eval = OrderedDict(
            [('psnr_x%d' % s, 0.0) for s in opt.eval.scale])
        self.eval_dict = OrderedDict(
            [('psnr_x%d' % s, []) for s in opt.eval.scale])
        self.train_history = []

        self.tensor2im = lambda t: tensor2im(t,
            mean=self.opt.mean_img, img_mul=self.opt.mul_img)

    def name(self):
        return 'BaseModel'

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

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
        self.best_eval = {
            k: max(self.best_eval[k], eval_result[k]) for k in self.best_eval
        }
        is_best_sofar = any(
            [eval_result[k] == v for k, v in self.best_eval.items()])
        if is_best_sofar:
            self.best_epoch = epoch

    def save(self, label):
        pass

    def evaluate(self, x, y):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.cuda()

    def save_optimizer(self, optimizer, label, epoch_label):
        save_filename = '%s_optim_%s.pth' % (epoch_label, label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, optimizer, label, epoch_label):
        save_filename = '%s_optim_%s.pth' % (epoch_label, label)
        save_path = os.path.join(self.save_dir, save_filename)
        loaded_state = torch.load(save_path)
        optimizer.load_state_dict(loaded_state)
        info('Loaded optimizer state from ' + save_path)

    def set_learning_rate(self, lr, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        loaded_state = torch.load(save_path)
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
        info('Loaded network state from ' + save_path)

    def update_learning_rate(self):
        pass

    def finetune(self, model, param_prefix):
        """finetune params with prefix defined in list param_prefix"""
        if len(param_prefix) == 0:
            return
        for p_name, p in model.named_parameters():
            freeze = all([p_name[:len(pre)] != pre for pre in param_prefix])
            if freeze:
                p.requires_grad = False
