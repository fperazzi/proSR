# import visdom

import numpy as np

import torch
from prosr.dataset import DataLoader, Dataset, pil_loader, tensor2im
from prosr.models import EDSR
from torch.autograd import Variable


def set_seed(rng_seed):
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(rng_seed)
    else:
        torch.cuda.manual_seed_all(rng_seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)


def clip_gradients(model, threshold):
    for param in model.parameters():
        param.grad.data.clamp_(-threshold, threshold)


if __name__ == '__main__':

    # Seed training
    # vis = visdom.Visdom()

    mul_img = 255
    mean_img = [0.4488, 0.4371, 0.4040]
    std = tuple([1 / mul_img] * 3)
    dataset = Dataset(
        './data/datasets/DIV2K_train_LR_bicubic/X8',
        './data/datasets/DIV2K_train_HR',
        'png',
        crop_size=48)

    batch_size = 8
    loss_avg20 = 0

    n_epochs = 150
    model = EDSR(8, 36)
    model.cuda()

    beta1 = 0.9  # momentum term of adam
    beta2 = 0.999  # momentum term of adam
    epsilon = 1e-8
    lr = 0.0001  # initial learning rate for adam

    loss_fn = torch.nn.L1Loss()
    clip = -1

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(beta1, beta2),
        eps=epsilon)

    data_loader = DataLoader(None, dataset, batch_size)
    for epoch in range(n_epochs):
        __import__('pdb').set_trace()
        for i, data in enumerate(data_loader):
            inputs = Variable(data['input']).cuda()
            outputs = model(inputs)

            loss = 0
            for output, label in zip(outputs, data['target']):
                loss += loss_fn(output, label.cuda())

            optimizer.zero_grad()  #reset gradients

            if clip > 0:
                clip_gradients_fn(model, clip / lr)

            loss.backward()
            optimizer.step()
            outputs = outputs.cpu().data
            targets = data['target']

            output_images = [
                tensor2im(o / 255.0, mean_img, 1, False) for o in outputs
            ]
            target_images = [
                tensor2im(t / 255.0, mean_img, 1, False) for t in targets
            ]

            # vis.images(output_images,win='Output')
            # vis.images(target_images,win='GT')
            loss_avg20 += loss.cpu().data[0]

            if i % 20 == 0:
                print("Epoch: %.5d - loss: %.5f" % (i, loss_avg20))
                loss_avg20 = 20

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, "checkpoint_%.4d.pth" % (epoch + 1))
    # ret_data = dataset.get(0)
    # # from torchsummary import summary
    # print(ret_data['input'].shape)
    # output = edsr(ret_data['input'])
    # edsr.cuda()
    # summary(edsr,(3,128,120))
