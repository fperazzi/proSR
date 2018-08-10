from ..config import phase
from ..logger import info
from ..metrics import eval_psnr
from ..misc.util import tensor2im
from collections import OrderedDict
from easydict import EasyDict as edict

import os
import torch


class BaseModel():
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
            k: max(self.best_eval[k], eval_result[k])
            for k in self.best_eval
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
        to_save = edict({
            'state_dict': network.cpu().state_dict(),
            'params': {
                'G': self.opt.G
            }
        })
        torch.save(to_save, save_path)
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

    def update_learning_rate(self):
        """update learning rate with exponential decay"""
        lr = self.old_lr * self.opt.train.lr_decay
        if lr < self.opt.train.smallest_lr:
            return
        self.set_learning_rate(lr, self.optimizer_G)
        info('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
