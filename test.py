import argparse
import os
import torch

import torchvision.utils as vutils

from Models.model import SSMNet
from dataset.pairs_dataset_test import UnalignedDataLoader

parser = argparse.ArgumentParser('OTSUNet models test')
parser.add_argument('--data_root', default='./data/test', type=str)
parser.add_argument('--width', default=512, type=int)
parser.add_argument('--height', default=512, type=int)
parser.add_argument('--save_path', default='./results', type=str)
parser.add_argument('--test_save_path', default='./results_test', type=str)
parser.add_argument('--shuffle', default=False, action='store_true')
parser.add_argument('--load_epoch', default=10, type=int)
parser.add_argument('--cuda', default=True, action='store_true')
parser.add_argument('--num_gpu', default=0, type=int)
parser.add_argument('--use_lsgan', default=True, type=bool)
parser.add_argument('--backward_type', default='separate')
parser.add_argument('--train', default=False, type=bool)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--input_nc', default=3, type=int)
parser.add_argument('--output_nc', default=3, type=int)
parser.add_argument('--identity', default=0, type=int)
parser.add_argument('--num_test_iterations', default=100, type=int)
parser.add_argument('--phase', default='test', type=str)
parser.add_argument('--nums_layer', default=6, type=int)
parser.add_argument('--num_filter', default=64, type=int)
parser.add_argument('--prior_channels', default=512, type=int)
parser.add_argument('--prior_size', default=16, type=int)
parser.add_argument('--in_channels', default=2048, type=int)
parser.add_argument('--channels', default=512, type=int)
parser.add_argument('--am_kernel_size', default=11, type=int)
parser.add_argument('--num_classes', default=19, type=int)
args = parser.parse_args()

for k,v in vars(args).items():
    print('{} = {}'.format(k,v))

use_gpu = torch.cuda.is_available()

def test():
    test_dir = os.path.join(args.test_save_path,'test')
    if not os.path.exists(args.save_path):
        return 0
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    model = SSMNet(args, use_gpu)
    model.load_parameters(args.load_epoch)
    # model.print_model_desription()

    data_loader = UnalignedDataLoader(args)
    dataset = data_loader.load_data()

    for batch_idx, inputs in enumerate(dataset):
        print(batch_idx)
        if batch_idx >= args.num_test_iterations:
            print("here")
            break
        model.set_inputs_test(inputs)
        model.test_model()

        imAB_gen_file = os.path.join(test_dir, 'imAB_gen_{}_{}_{}_test.jpg'.format(batch_idx, args.height, args.width))
        vutils.save_image(model.get_AB_images_triple_test(), imAB_gen_file, normalize=True)
        print('processed item with idx: {}'.format(batch_idx))
        torch.cuda.empty_cache()

if __name__ == '__main__':
    test()

