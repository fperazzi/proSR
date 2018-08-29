from .utils import html

import ntpath
import numpy as np
import os
import skimage.io as io


class Visualizer():
    def __init__(self, name, port=8067, use_visdom=True,use_incoming_socket=False):
        self.win_size = 192
        self.name = name
        self.plot_data = {}
        if use_visdom:
            import visdom
            self.vis = visdom.Visdom(port=port, env=self.name,
                                     use_incoming_socket=use_incoming_socket)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        for label, item in visuals.items():
            self.vis.image(
                np.transpose(item, (2, 0, 1)),
                opts=dict(title=label),
                win=label)

    # errors: dictionary of error labels and values
    def plot(self, data, epoch, display_id, ylabel='value'):
        if display_id not in self.plot_data:
            self.plot_data[display_id] = {
                'X': [],
                'Y': [],
                'legend': list(data.keys())
            }
        mdata = self.plot_data[display_id]
        mdata['X'].append(epoch)
        mdata['Y'].append(
            [data[k] for k in self.plot_data[display_id]['legend']])
        self.vis.line(
            X=np.stack([np.array(mdata['X'])] * len(mdata['legend']), 1),
            Y=np.array(self.plot_data[display_id]['Y']),
            opts={
                'title': display_id,
                'ytickmax': 1e-4,
                'legend': mdata['legend'],
                'xlabel': 'epoch',
                'ylabel': ylabel
            },
            win=(display_id))

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            io.imsave(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
