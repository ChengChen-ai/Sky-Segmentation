# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import argparse
import time
import os
import torch.utils.data
import random

import torchvision.utils as vutils

from visdom import Visdom
from Models.model import SSMNet
from dataset.pairs_dataset import UnalignedDataLoader
from util.logger import Logger
from collections import OrderedDict
from util.visualisations import Visualizer

parser = argparse.ArgumentParser('OTSUNet train')
parser.add_argument('--data_root', default='./data', type=str)
parser.add_argument('--width', default=512, type=int)
parser.add_argument('--height', default=512, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--beta1', default=0.5, type=float)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--input_nc', default=3, type=int)
parser.add_argument('--output_nc', default=3, type=int)
parser.add_argument('--train', default=True, type=bool)
parser.add_argument('--flag', default=True, type=bool)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--cuda', default=True, action='store_true')
parser.add_argument('--num_gpu', default=0, type=int)
parser.add_argument('--save_path', default='./results', type=str)
parser.add_argument('--save_interval', default=100, type=int)
parser.add_argument('--vis_interval', default=10, type=int)
parser.add_argument('--backward_type', default='separate')
parser.add_argument('--log_interval', default=10, type=int)
parser.add_argument('--display_id', default=1, type=int)
parser.add_argument('--phase', default='train', type=str)
parser.add_argument('--num_filter', default=64, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--nums_layer', default=6, type=int)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--continue_epoch', default=-1, type=int)
parser.add_argument('--prior_channels', default=512, type=int)
parser.add_argument('--prior_size', default=16, type=int)
parser.add_argument('--in_channels', default=2048, type=int)
parser.add_argument('--channels', default=512, type=int)
parser.add_argument('--am_kernel_size', default=11, type=int)
parser.add_argument('--num_classes', default=19, type=int)


args = parser.parse_args()

for k,v in vars(args).items():
    print('{:20} = {}'.format(k,v))


try:
    os.makedirs(args.save_path)
except OSError as e:
    pass

def set_random_seed(seed):
    random.seed(seed)
    torch.cuda.manual_seed(seed)

set_random_seed(args.seed)
Vis = Visdom(env='demo')

batch_size = args.batch_size
length = 0

data_loader = UnalignedDataLoader(args)
dataset = data_loader.load_data()
print("Successful Loader Data!!!!!!!")
visualizer = Visualizer(args)
logger = Logger(args.epochs, len(data_loader))

def train():
    use_gpu = torch.cuda.is_available()

    model = SSMNet(params=args, use_gpu=use_gpu)
    model.train()

    if args.continue_epoch > -1:
        model.load_parameters(args.continue_epoch)

    for e in range(args.epochs):
        e_begin = time.time()
        for batch_idx, inputs in enumerate(dataset):
            model.set_inputs(inputs)
            model.optimize_parameters()
            e_fraction_passed = batch_idx * args.batch_size / len(dataset.data_loader_A)
            images = model.get_AB_images_triple()
            errors = model.get_errors()
            errors = list(errors.values())

            logger.log(OrderedDict(
                [('lossTotal', errors[0]),('loss_dec', errors[1]),('loss_enh', errors[2])
                    ,('loss_G', errors[3]), ('loss_D', errors[4]),('loss_V',errors[5])]),
                images = {'real_A' : images[0*batch_size +length], 'R_a' : images[1*batch_size+length], 'real_B' : images[2*batch_size+length],
                          'R_b' : images[3*batch_size+length], 'feature_map' : images[4*batch_size*batch_size+length], 'fake_sky' : images[5*batch_size+length],
                          }
            )

            if batch_idx % args.log_interval == 0:
                err = model.get_errors()
                visualizer.plot_errors(err, e, e_fraction_passed)
                desc = model.get_errors_string()
                print('Epoch:[{}/{}] Batch:[{:10d}/{}] '.format(e, args.epochs,
                                                                batch_idx * args.batch_size,
                                                                len(dataset.data_loader_A)), desc)
            if batch_idx % args.vis_interval == 0:
                imAB_gen_file = os.path.join(args.save_path, 'imAB_gen_{}_{}.jpg'.format(e, batch_idx))
                vutils.save_image(model.get_AB_images_triple(), imAB_gen_file, normalize=True)
            if batch_idx % args.save_interval == 0:
                model.save_parameters(e)
        e_end = time.time()
        e_time = e_end - e_begin
        print('End of epoch [{}/{}] Time taken: {:.4f} sec.'.format(e, args.epochs, e_time))

    print('saving final model paramaters')
    model.save_parameters(args.epochs)


if __name__ == '__main__':
    train()
