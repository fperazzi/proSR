import numpy as np
import os
import ntpath
import skimage.io as io
from .utils import html

class Visualizer():
    def __init__(self, name, port=8067, use_html=False, use_visdom=True):
        self.use_html = use_html
        self.win_size = 192
        self.name = name
        self.plot_data = {}
        if use_visdom:
            import visdom
            self.vis = visdom.Visdom(port=port, env=self.name)
            print('start visom python -m visdom.server -port {}'.format(port))

        if self.use_html:
            self.web_dir = os.path.join('data/checkpoints', self.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            os.makedirs(self.web_dir)
            os.makedirs(self.img_dir)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, iter=0, display_id=1):
        idx = 1
        for label, item in visuals.items():
            if 'pc' in label:
                self.vis.scatter(np.transpose(item),
                                 Y=None,
                                 opts=dict(title=label, markersize=0.5),
                                 win=display_id + idx)
            elif 'img' in label:
                # the transpose: HxWxC -> CxHxW
                self.vis.image(np.transpose(item, (2,0,1)), opts=dict(title=label),
                               win=display_id + idx)
            idx += 1

        if self.use_html:  # save images to a html file
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d-%d_%s.png' % (epoch, iter, label))
                io.imsave(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d-%d_%s.png' % (n, iter, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot(self, data, epoch, display_id):
        if display_id not in self.plot_data:
            self.plot_data[display_id] = {'X': [], 'Y': [], 'legend': list(data.keys())}
        mdata = self.plot_data[display_id]
        mdata['X'].append(epoch)
        mdata['Y'].append([data[k] for k in self.plot_data[display_id]['legend']])
        self.vis.line(
            X=np.stack([np.array(mdata['X'])] * len(mdata['legend']), 1),
            Y=np.array(self.plot_data[display_id]['Y']),
            opts={
                'title': ' + '.join(mdata['legend']),
                'legend': mdata['legend'],
                'xlabel': 'epoch',
                'ylabel': 'value'},
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
