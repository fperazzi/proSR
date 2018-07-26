# from torch.autograd import Variable
from bisect import bisect_left
from collections import OrderedDict

import torch

from ..config import phase
from ..logger import info, warn
from .base_model import BaseModel


class StdSuperresModel(BaseModel):

    def __init__(self, opt):
        super(StdSuperresModel, self).__init__(opt)
        self.input_t = self.Tensor(opt.train.batch_size, opt.G.num_img_channels,
                                   48, 48)
        self.blend = 1

        if opt.phase != phase.TEST:
            self.labels_t = [
                self.Tensor(opt.train.batch_size, opt.G.num_img_channels,
                            int(48 * max(opt.scale)), int(48 * max(opt.scale)))
            ]

        # load/define networks
        if opt.phase == phase.TRAIN:
            opt.G.scale = opt.scale
        else:
            opt.G.scale = opt.eval.scale

        opt.G.num_scales = len(opt.scale)

        if opt.G.output_residual:
            self.interpolated_t = [
                self.Tensor(opt.train.batch_size, opt.G.num_img_channels,
                            int(48 * max(opt.scale)), int(48 * max(opt.scale)))
            ]

        self._net_G = define_G(opt.G)
        self.finetune(self._net_G, opt.train.fine_tune)

        # load network weights
        if not opt.phase == phase.TRAIN or opt.train.resume:
            self.load_network(self._net_G, 'G', opt.which_epoch)
        retun

        # Wrap model if multiple gpus
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.net_G = torch.nn.DataParallel(self._net_G.cuda())
            else:
                self.net_G = self._net_G.cuda()
        else:
            self.net_G = self._net_G

        if opt.phase == phase.TRAIN:
            # define loss functions
            if opt.G.l1_loss_weight > 0 or opt.G.lr_loss_weight > 0:
                self.l1_criterion = torch.nn.L1Loss()

            self.has_vgg_loss = sum(opt.G.vgg_loss_weight) > 0
            if self.has_vgg_loss > 0:
                self.acquire_vgg = []
                self.acquire_vgg += self.opt.G.vgg
                self.vgg_net = Vgg16(
                    opt.mean_img,
                    opt.mul_img,
                    upto=max(self.acquire_vgg),
                    mean_pool=opt.G.vgg_mean_pool)
                if torch.cuda.is_available():
                    print('using gpu')
                    if torch.cuda.device_count() > 1:
                        self.vgg_net = torch.nn.DataParallel(
                            self.vgg_net.cuda())
                    elif torch.cuda.is_available():
                        self.vgg_net = self.vgg_net.cuda()
                self.mse_criterion = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                [p for p in self.net_G.parameters() if p.requires_grad],
                lr=self.opt.train.lr,
                betas=(opt.train.beta1, opt.train.beta2),
                eps=opt.train.epsilon)
            if opt.train.resume:
                try:
                    self.load_optimizer(self.optimizer_G, 'G', opt.which_epoch)
                    for param_group in self.optimizer_G.param_groups:
                        self.old_lr = param_group['lr']
                        break
                    info('Set lr = %e' % self.old_lr)
                except Exception as e:
                    warn(str(e))
                    self.set_learning_rate(self.old_lr, self.optimizer_G)

            # used for skip_update
            self.old_loss = float('Inf')
            self._total_loss = 0.0
            self._total_steps = 0
            self.progress = self.start_epoch / opt.train.epochs
            self.net_G.train()
        else:
            self.net_G.eval()

        # print('---------- Networks initialized -------------')
        # util.print_network(self.net_G)
        # info('Generator use residual input: ' + str(opt.G.output_residual))
        # print('-----------------------------------------------')

    def name(self):
        return 'SuperresModel'

    def set_train(self):
        self.net_G.train()
        self.isTrain = True

    def set_eval(self):
        self.net_G.eval()
        self.isTrain = False

    def set_model_scale(self, scale):
        self.model_scale = scale[0]

    def set_input(self, **kwargs):
        lr = hrs = interpolated = None
        if 'lr' in kwargs:
            lr = kwargs['lr']
            lr = lr.narrow(1, 0, self.opt.G.num_img_channels)
        if 'hr' in kwargs:
            hrs = kwargs['hr']
        if 'interp' in kwargs:
            interpolated = kwargs['interp']

        if self.opt.G.which_model_netG.lower() == 'downsample':
            input_t = hrs[0] if hrs is not None else None
            if input_t is None:
                raise ValueError('downsample network must have hr input')
            labels_t = [lr] if lr is not None else None
            interpolated_t = interpolated
        else:
            input_t = lr
            labels_t = hrs
            interpolated_t = interpolated

        self.input_t.resize_(input_t.size()).copy_(input_t)

        if self.isTrain:
            if labels_t is None:
                raise ValueError('train phase must have label image')
            self.labels = []
            for i in range(len(labels_t)):
                assert isinstance(labels_t[i], torch.Tensor)
                labels_t[i] = labels_t[i].narrow(1, 0,
                                                 self.opt.G.num_img_channels)
                self.labels_t[i].resize_(labels_t[i].size()).copy_(labels_t[i])
                # self.labels += [Variable(self.label_t[i], volatile=(not self.isTrain))]
            if interpolated_t is not None:
                self.interpolated = []
                for i in range(len(interpolated)):
                    assert isinstance(interpolated[i], torch.Tensor)
                    interpolated[i] = interpolated[i].narrow(
                        1, 0, self.opt.G.num_img_channels)
                    self.interpolated_t[i].resize_(
                        interpolated[i].size()).copy_(interpolated[i])
                    # self.interpolated += [Variable(self.interpolated_t[i],
                    #                                volatile=(not self.isTrain))]
        else:
            if labels_t is not None:
                # self.labels = Variable(labels_t[0], volatile=True)
                self.labels_t = labels_t[0]
            if self.opt.G.output_residual:
                # self.interpolated = Variable(interpolated[0], volatile=True)
                self.interpolated = interpolated[0]

    def forward(self):
        self.outputs = self.net_G(
            self.input_t, scale=self.model_scale, blend=self.blend)
        for i in range(len(self.outputs)):
            if self.opt.G.output_residual:
                self.outputs[i] += self.interpolated[i]
        assert len(self.outputs) == len(self.labels_t)

    def test(self):
        b, c, h, w = self.input_t.size()
        if (h * w) < self.opt.eval.chop_size:
            self.outputs = self._net_G(
                self.input_t, scale=self.model_scale, blend=self.blend).cpu()
            if self.opt.G.output_residual:
                self.outputs += self.interpolated
        else:
            self.outputs = self._chop_test(self.input_t)
            if self.opt.G.output_residual:
                self.outputs += self.interpolated

    def evaluate(self):
        im1 = self.tensor2im(self.labels_t)
        im2 = self.tensor2im(self.outputs)
        eval_res = {
            name: func(im1, im2)
            for name, func in self.eval_func.items()
            if 'x%d' % self.model_scale in name
        }
        for k, v in eval_res.items():
            self.eval_dict[k].append(v)
        return eval_res

    def chop_tensor(self, tensor, patch_size=None):
        b, c, h, w = tensor.size()
        if patch_size is None:
            hHalf, wHalf = h // 2, w // 2
            hc = hHalf + self.opt.eval.chop_shave
            wc = wHalf + self.opt.eval.chop_shave
        else:
            hc, wc = patch_size
        # x1, y1 --- x2, y1
        # x1, y2 --- x2, y2
        # bnd = {'x1': 0, 'x2': w - wc, 'y1': 0, 'y2': h - hc}
        ret = [
            tensor.narrow(2, 0, hc).narrow(3, 0, wc),
            tensor.narrow(2, 0, hc).narrow(3, w - wc, wc),
            tensor.narrow(2, h - hc, hc).narrow(3, 0, wc),
            tensor.narrow(2, h - hc, hc).narrow(3, w - wc, wc),
        ]
        return ret

    def _chop_test(self, tensor, base_img=None):
        scale = self.model_scale
        b, c, h, w = tensor.size()
        nGPU = torch.cuda.device_count()

        input_patch = self.chop_tensor(tensor)

        if base_img is not None:
            # dict of list
            _, _, hc, wc = input_patch[0].size()
            base_img = self.chop_tensor(base_img, patch_size=(hc * 2, wc * 2))

        hc, wc = input_patch[0].size()[2:]
        outputPatch = []
        if (wc * hc) < self.opt.eval.chop_size:
            for i in range(0, 4, nGPU):
                inputBatch_var = torch.cat(input_patch[i:i + nGPU], dim=0)
                if base_img is not None:
                    outBatch_var = self.net_G(
                        inputBatch_var,
                        scale=self.model_scale,
                        blend=self.blend,
                        base_img=torch.cat(base_img[i:i + nGPU], dim=0)).cpu()
                else:
                    outBatch_var = self.net_G(
                        inputBatch_var,
                        scale=self.model_scale,
                        blend=self.blend).cpu()
                outputPatch.append(outBatch_var.data.clone())
                del inputBatch_var

            outputPatch = torch.cat(outputPatch, dim=0)
        else:
            for i in range(4):
                if base_img is not None:
                    outputPatch.append(
                        self._chop_test(input_patch[i], base_img[i]))
                else:
                    outputPatch.append(self._chop_test(input_patch[i]))
            outputPatch = torch.cat(outputPatch, dim=0)

        hnew = int(scale * h)
        wnew = int(scale * w)
        wHalf = w // 2
        hHalf = h // 2
        ret = torch.FloatTensor(b, c, hnew, wnew)
        w, wHalf, wc = wnew, int(scale * wHalf), int(scale * wc)
        h, hHalf, hc = hnew, int(scale * hHalf), int(scale * hc)
        # bndR = {'x1': (0, wHalf), 'x2': (wHalf, w), 'y1': (0, hHalf), 'y2': (hHalf, h)}
        # bndO = {'x2': (wc - w + wHalf, wc), 'y2': (hc - h + hHalf, hc)}
        ret[:, :, 0:hHalf, 0:wHalf].copy_(outputPatch[0][:, 0:hHalf, 0:wHalf])
        ret[:, :, 0:hHalf, wHalf:w].copy_(
            outputPatch[1][:, 0:hHalf, wc - w + wHalf:wc])
        ret[:, :, hHalf:h, 0:wHalf].copy_(
            outputPatch[2][:, hc - h + hHalf:hc, 0:wHalf])
        ret[:, :, hHalf:h, wHalf:w].copy_(
            outputPatch[3][:, hc - h + hHalf:hc, wc - w + wHalf:wc])
        del outputPatch
        return ret

    def backward(self):
        self.compute_loss()
        self.loss.backward()

    def compute_loss(self):
        self.loss = 0
        if self.opt.G.l1_loss_weight > 0:
            self.l1_loss = 0
            for o, t in zip(self.outputs, self.labels_t):
                self.l1_loss += self.l1_criterion(o,
                                                  t) * self.opt.G.l1_loss_weight
            self.loss += self.l1_loss

        if self.has_vgg_loss > 0:
            self.vgg_loss = 0
            for o, t in zip(self.outputs, self.labels_t):
                vgg_real = self.vgg_net(t, acquire=self.acquire_vgg)
                vgg_fake = self.vgg_net(o, acquire=self.acquire_vgg)
                if self.has_vgg_loss:
                    for i in range(len(self.opt.G.vgg)):
                        self.vgg_loss += self.mse_criterion(vgg_fake[i], vgg_real[i].detach()) * \
                                         self.opt.G.vgg_loss_weight[i]
            self.loss += self.vgg_loss

    def optimize_parameters(self):
        """call this function after forward()"""
        self.optimizer_G.zero_grad()
        self.backward()

    def increment_training_progress(self):
        self.progress += 1 / len(self.training_dataset) / self.opt.train.epochs

    def get_current_errors(self):
        d = OrderedDict()
        for s in self.opt.scale:
            if self.opt.G.l1_loss_weight > 0:
                d['l1_x%d' % s] = 0.0
            if self.has_vgg_loss:
                d['vgg_x%d' % s] = 0.0
            if self.opt.G.lr_loss_weight > 0:
                d['lr_x%d' % s] = 0.0

        if hasattr(self, 'l1_loss'):
            d['l1_x%d' % self.model_scale] = self.l1_loss.data[0]
        if hasattr(self, 'vgg_loss'):
            d['vgg_x%d' % self.model_scale] = self.vgg_loss.data[0]

        return d

    def get_current_visuals(self):
        disp = OrderedDict()
        disp['input'] = self.tensor2im(self.input_t)
        cnt = 0
        for output_t in self.outputs:
            disp['output_%d' % cnt] = self.tensor2im(output_t)
            cnt += 1
        if hasattr(self, 'labels'):
            cnt = 0
            for label in self.labels_t:
                # label = label.data if isinstance(label, Variable) else label
                disp['label_%d' % cnt] = self.tensor2im(label)
                cnt += 1
        return disp

    def save(self, epoch):
        self.save_network(self._net_G, 'G', epoch)
        self.save_optimizer(self.optimizer_G, 'G', epoch)

    def update_learning_rate(self):
        """update learning rate with exponential decay"""
        lr = self.old_lr * self.opt.train.lr_decay
        if lr < self.opt.train.smallest_lr:
            return
        self.set_learning_rate(lr, self.optimizer_G)
        info('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class GrowingGeneratorModel(StdSuperresModel):
    """docstring for GrowingGeneratorModel"""

    def __init__(self, opt):
        super().__init__(opt)
        if self.isTrain:
            # set data loader to load correct scales
            assert len(self.opt.train.growing_steps) == len(opt.scale) * 2 - 1
            i = bisect_left(self.opt.train.growing_steps, self.progress)
            self.max_scale_idx = (i + 1) // 2
            self._net_G.max_scale_idx = self.max_scale_idx
            training_scales = [
                opt.scale[i] for i in range(self.max_scale_idx + 1)
            ]
            self.training_dataset.random_vars.clear()
            for s in training_scales:
                self.training_dataset.random_vars.append(s)
            info('start training with scales: {}'.format(str(training_scales)))

    def name(self):
        return 'GrowingGeneratorModel'

    def increment_max_scale(self):
        if self.progress > self.opt.train.growing_steps[self.max_scale_idx * 2]:
            if self.max_scale_idx < len(self.opt.scale) - 1:
                self.max_scale_idx += 1
                self._net_G.max_scale_idx = self.max_scale_idx

                training_scales = [
                    self.opt.scale[i] for i in range(self.max_scale_idx + 1)
                ]
                self.training_dataset.random_vars.clear()
                for s in training_scales:
                    self.training_dataset.random_vars.append(s)
                info('start training with scales: {}'.format(
                    str(training_scales)))

    def optimize_parameters(self):
        result = super().optimize_parameters()
        self.increment_max_scale()
        return result

    def forward(self):
        # progress is curr_step/epochs * steps_per_epoch
        if self.max_scale_idx != 0:
            lo, hi = self.opt.train.growing_steps[self.max_scale_idx * 2 -
                                                  2:self.max_scale_idx * 2]
            if lo == hi:
                self.blend = 1
            else:
                self.blend = min((self.progress - lo) / (hi - lo), 1)
            assert self.blend >= 0 and self.blend <= 1
        else:
            self.blend = 1
        super().forward()
