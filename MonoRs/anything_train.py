import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import os, sys, time
from telnetlib import IP
import argparse
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils import post_process_depth, flip_lr, silog_loss, compute_errors, eval_metrics, entropy_loss, colormap, \
    block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args, colormap_magma
from new_networks.NewCRFDepth import NewCRFDepth
from new_networks.depth_update import *
from datetime import datetime
from sum_depth import Sum_depth
from networks.losses import *

parser = argparse.ArgumentParser(description='IEBins PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--model_name', type=str, help='model name', default='iebins')
parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07, tiny07', default='large07')
parser.add_argument('--pretrain', type=str, help='path of pretrained encoder', default=None)

# Dataset
parser.add_argument('--dataset', type=str, help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--gt_path', type=str, help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=0.1)

# Log and save
parser.add_argument('--log_directory', type=str, help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq', type=int, help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq', type=int, help='Checkpoint saving frequency in global steps', default=5000)

# Training
parser.add_argument('--weight_decay', type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--retrain', help='if used with checkpoint_path, will restart training from step zero',
                    action='store_true')
parser.add_argument('--adam_eps', type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size', type=int, help='batch size', default=4)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=50)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate', type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus', type=float,
                    help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error',
                    default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate', help='if set, will perform random rotation for augmentation',
                    action='store_true')
parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                    action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size', type=int, help='number of nodes for distributed training', default=1)
parser.add_argument('--rank', type=int, help='node rank for distributed training', default=0)
parser.add_argument('--dist_url', type=str, help='url used to set up distributed training',
                    default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend', type=str, help='distributed backend', default='nccl')
parser.add_argument('--gpu', type=int, help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed', help='Use multi-processing distributed training to launch '
                                                          'N processes per node, which has N GPUs. This is the '
                                                          'fastest way to use PyTorch for either single node or '
                                                          'multi node data parallel training', action='store_true', )
# Online eval
parser.add_argument('--do_online_eval', help='if set, perform online eval in every eval_freq steps',
                    action='store_true')
parser.add_argument('--data_path_eval', type=str, help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval', type=str, help='path to the groundtruth data for online evaluation',
                    required=False)
parser.add_argument('--filenames_file_eval', type=str, help='path to the filenames text file for online evaluation',
                    required=False)
parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=10)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq', type=int, help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory', type=str, help='output directory for eval summary,'
                                                               'if empty outputs to checkpoint folder', default='')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


from dataloaders.anywhu_dataloader import NewDataLoader


def online_eval(model, dataloader_eval, gpu, epoch, ngpus, group, post_process=False):
    eval_measures = torch.zeros(13).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']

            pred_depths_r_list, _, _ = model(image)

            if post_process:
                image_flipped = flip_lr(image)
                pred_depths_r_list_flipped, _, _ = model(image_flipped)
                pred_depth = post_process_depth(pred_depths_r_list[-1], pred_depths_r_list_flipped[-1])

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()



        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval


        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:12] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[12] += 1

    if args.multiprocessing_distributed:

        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[12].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3','d1_new','d2_new','d3_new'))
        for i in range(11):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.4f}'.format(eval_measures_cpu[11]))
        return eval_measures_cpu

    return None


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("== Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,rank=args.rank)

    # model
    # model = NewCRFDepth(encoder=args.encoder, inv_depth=False,max_depth=args.max_depth,  pretrained=args.pretrain)
    model = NewCRFDepth(encoder=args.encoder, inv_depth=False,max_depth=args.max_depth)
    if args.pretrain:
        model.load_state_dict({k: v for k, v in torch.load(args.pretrain, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    model.train()

    #冻结backbone
    for name,param in model.named_parameters():
        if 'pretrained' in name:
            param.requires_grad=False

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))




    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.distributed:
        print("== Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("== Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Training parameters
    optimizer = torch.optim.Adam([{'params': model.module.parameters()}],
                                 lr=args.learning_rate)

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.retrain:
                try:
                    global_step = checkpoint['global_step']
                    best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                    best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                    best_eval_steps = checkpoint['best_eval_steps']
                except KeyError:
                    print("Could not load values for online evaluation")

            print("== Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True
        del checkpoint

    cudnn.benchmark = True

    dataloader = NewDataLoader(args, 'train')
    dataloader_eval = NewDataLoader(args, 'online_eval')



    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    silog_criterion = silog_loss(variance_focus=args.variance_focus)
    sum_localdepth = Sum_depth().cuda(args.gpu)

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum / var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    group = dist.new_group([i for i in range(ngpus_per_node)])

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()
            si_loss = 0
            ad_loss=0



            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))


            pred_depths_r_list, pred_depths_c_list, uncertainty_maps_list = model(image, epoch, step)

            mask = depth_gt > 1.0
            max_tree_depth = len(pred_depths_r_list)

            for curr_tree_depth in range(max_tree_depth):


                si_loss += silog_criterion.forward(pred_depths_r_list[curr_tree_depth], depth_gt, mask.to(torch.bool))
                ad_loss+=Adaptive_Multi_Modal_Cross_Entropy_Loss(pred_depths_r_list[curr_tree_depth],depth_gt,mask.to(torch.bool),maxdepth=args.max_depth,m=1,n=9,top_k=9,epsilon=3,min_samples=1)

            #loss = si_loss
            loss = 0.5*si_loss+0.5*ad_loss

            loss.backward()  # 不同308-315
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (
                            1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            if not args.multiprocessing_distributed or (
                    args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step,steps_per_epoch,global_step,current_lr, loss))
                # if np.isnan(loss.cpu().item()):
                #     print('NaN in loss occurred. Aborting training.')
                #     return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (
                        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item() / var_cnt,
                                          time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('silog_loss', si_loss, global_step)
                    # writer.add_scalar('var_loss', var_loss, global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar('var average', var_sum.item() / var_cnt, global_step)

                    writer.flush()

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                model.eval()
                with torch.no_grad():
                    eval_measures = online_eval(model, dataloader_eval, gpu, epoch, ngpus_per_node, group,post_process=True)
                if eval_measures is not None:
                    exp_name = '%s' % (datetime.now().strftime('%m%d'))
                    log_txt = os.path.join(args.log_directory + '/' + args.model_name, exp_name + '_logs.txt')
                    with open(log_txt, 'a') as txtfile:
                        txtfile.write(
                            ">>>>>>>>>>>>>>>>>>>>>>>>>Step:%d>>>>>>>>>>>>>>>>>>>>>>>>>\n" % (int(global_step)))
                        txtfile.write("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}\n".format('silog',
                                        'abs_rel','log10','rms','sq_rel','log_rms','d1','d2','d3','d1_new','d2_new','d3_new'))
                        txtfile.write("depth estimation\n")
                        line = ''
                        for i in range(12):
                            line += '{:7.4f}, '.format(eval_measures[i])
                        txtfile.write(line + '\n')  # 356-367额外添加

                    for i in range(9):
                        eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i - 6]:
                            old_best = best_eval_measures_higher_better[i - 6].item()
                            best_eval_measures_higher_better[i - 6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    eval_summary_writer.flush()
                model.train()
                block_print()
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer.close()
        if args.do_online_eval:
            eval_summary_writer.close()


def main():
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1

    exp_name = '%s' % (datetime.now().strftime('%m%d'))
    args.log_directory = os.path.join(args.log_directory, exp_name)
    command = 'mkdir ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)

    args_out_path = os.path.join(args.log_directory, args.model_name)
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    save_files = True
    if save_files:
        aux_out_path = os.path.join(args.log_directory, args.model_name)
        networks_savepath = os.path.join(aux_out_path, 'networks')
        dataloaders_savepath = os.path.join(aux_out_path, 'dataloaders')
        command = 'cp /train.py ' + aux_out_path
        os.system(command)
        command = 'mkdir -p ' + networks_savepath + ' && cp /networks/*.py ' + networks_savepath
        os.system(command)
        command = 'mkdir -p ' + dataloaders_savepath + ' && cp /dataloaders/*.py ' + dataloaders_savepath
        os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print(
            "This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
