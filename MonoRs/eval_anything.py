import torch
import torch.backends.cudnn as cudnn

import os, sys
import argparse
import numpy as np
from tqdm import tqdm

from utils import post_process_depth, flip_lr, compute_errors
from networks.NewCRFDepth import NewCRFDepth

from dataloaders.anywhu_dataloader import NewDataLoader

from depth_anything_v2.dpt import DepthAnythingV2

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='IEBins PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='iebins')
parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07, tiny07', default='large07')
parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')

# Dataset
parser.add_argument('--dataset', type=str, help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)

# Preprocessing
parser.add_argument('--do_random_rotate', help='if set, will perform random rotation for augmentation',
                    action='store_true')
parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                    action='store_true')

# Eval
parser.add_argument('--data_path_eval', type=str, help='path to the data for evaluation', required=False)
parser.add_argument('--gt_path_eval', type=str, help='path to the groundtruth data for evaluation', required=False)
parser.add_argument('--filenames_file_eval', type=str, help='path to the filenames text file for evaluation',
                    required=False)
parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=300)
parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--input-size', type=int, default=518)



model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}




if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()




def eval(model, dataloader_eval, post_process=False):
    eval_measures = torch.zeros(10).cuda()

    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'])
            image=image.squeeze()
            gt_depth = eval_sample_batched['depth']
            image=image.numpy().transpose(1,2,0)
            padding_height, padding_width = 420, 840
            top_pad, left_pad = padding_height - image.shape[0], padding_width - image.shape[1]
            image = np.lib.pad(image, ((top_pad, 0), (0, left_pad), (0, 0)),
                             mode='constant', constant_values=0)



            pred_depths_r_list = model.infer_image(image,args.input_size)

            if post_process:

                image = image.transpose(2,0,1)

                image = torch.from_numpy(image)
                image = image.unsqueeze(0)


                image_flipped = flip_lr(image)
                image_flipped = image_flipped.cuda()
                pred_depths_flipped = model(image_flipped)
                image = image.cuda()
                pred_depth = model(image)

                pred_depth = post_process_depth(pred_depth.unsqueeze(1), pred_depths_flipped.unsqueeze(1))

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
            pred_depth = pred_depth[top_pad:, : -left_pad]



        
        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)


        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                 'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                 'd3'))
    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.4f}'.format(eval_measures_cpu[8]))
    return eval_measures_cpu


def main_worker(args):
    # CRF model
    #加载depthanything
    #model = NewCRFDepth(version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=None)
    model=DepthAnythingV2(**model_configs['vitl'])
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    # model = torch.nn.DataParallel(model)
    model.cuda()

    print("== Model Initialized")

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            # checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            # model.load_state_dict(checkpoint['model'])
           
            model.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            # del checkpoint
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))

    cudnn.benchmark = True

    dataloader_eval = NewDataLoader(args, 'online_eval')
    # print("123")
    # ===== Evaluation ======
    model.eval()
    with torch.no_grad():
        eval_measures = eval(model, dataloader_eval, post_process=True)


def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    main_worker(args)


if __name__ == '__main__':
    main()
