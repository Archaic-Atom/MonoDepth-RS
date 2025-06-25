from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import os, sys, errno
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import post_process_depth, flip_lr
from new_networks.NewCRFDepth import NewCRFDepth
from dataloaders.anywhu_dataloader import NewDataLoader

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='IEBins PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='iebins')
parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07', default='large07')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--min_depth', type=float, help='maximum depth in estimation', default=0.01)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')



if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()




model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = NewDataLoader(args, 'test')

    model = NewCRFDepth(encoder='vitl', inv_depth=False, max_depth=args.max_depth,min_depth=args.min_depth)
    print("ok")
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []
    start_time = time.time()
    with torch.no_grad():
        for step, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
           

            # Predict

            pred_depths_r_list, _, _ = model(image)
            post_process = True
            if post_process:
                image_flipped = flip_lr(image)
                pred_depths_r_list_flipped, _, _ = model(image_flipped)
                pred_depth = post_process_depth(pred_depths_r_list[-1], pred_depths_r_list_flipped[-1])

            pred_depth = pred_depth.cpu().numpy().squeeze()


            pred_depth=(pred_depth-pred_depth.min())/(pred_depth.max()-pred_depth.min()) *255.0

            pred_depths.append(pred_depth)


    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')

    save_name = 'result_' + args.model_name

    print('Saving result pngs..')
    if not os.path.exists(save_name):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/2')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    for s in tqdm(range(num_test_samples)):
        filename_pred_png = save_name + '/2/' +lines[s].split()[0].split('/')[-1]

        pred_depth = pred_depths[s]
        pred_depth_scaled = pred_depth*100
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        plt.imsave(filename_pred_png, pred_depth_scaled, cmap='viridis')

    return


if __name__ == '__main__':
    test(args)
