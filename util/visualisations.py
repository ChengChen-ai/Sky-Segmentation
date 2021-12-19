import numpy as np
import matplotlib.pyplot as plt


def show(img):
    npimg = img.numpy()
    plt.imshow(npimg.transpose(1, 2, 0))


class Visualizer(object):
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.title = 'training loss {}'.format(opt.backward_type)
        if opt.display_id > 0:
            import visdom
            self.vis = visdom.Visdom()

    def display_images(self, visuals, epoch):
        idx = 1
        for label, image_numpy in visuals.iteritems():
            self.vis.image()

    def plot_errors(self, errors, epoch, fraction_passed):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + fraction_passed)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([self.plot_data['X']] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.title,
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id
        )
