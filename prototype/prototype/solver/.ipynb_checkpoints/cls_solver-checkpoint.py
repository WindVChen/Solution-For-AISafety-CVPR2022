import sys
sys.path.append('/code/')

import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import time
import datetime
import torch
import random
import json
import prototype.spring.linklink as link
import torch.nn.functional as F
from collections import OrderedDict
from prototype.prototype.solver.base_solver import BaseSolver
from prototype.prototype.utils.dist import link_dist, DistModule, broadcast_object
from prototype.prototype.utils.misc import makedir, create_logger, get_logger, count_params, count_flops, \
    param_group_all, AverageMeter, accuracy, load_state_model, load_state_optimizer, mixup_data, \
    mix_criterion, modify_state, cutmix_data, parse_config
from prototype.prototype.utils.ema import EMA
from prototype.prototype.model import model_entry
from prototype.prototype.optimizer import optim_entry
from prototype.prototype.lr_scheduler import scheduler_entry
from prototype.prototype.data import build_imagenet_train_dataloader, build_imagenet_test_dataloader
from prototype.prototype.data import build_custom_dataloader
from prototype.prototype.loss_functions import LabelSmoothCELoss
from torchcontrib.optim.swa import SWA

import numpy as np

dict = [0, 1, 12, 23, 34, 45, 56, 67, 78, 89, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84,
        85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

def normalize(x, mode='normal', typ=False):
    mean = torch.tensor(np.array([0.485, 0.456, 0.406]), dtype=x.dtype)[np.newaxis, :, np.newaxis, np.newaxis].cuda()
    var = torch.tensor(np.array([0.229, 0.224, 0.225]), dtype=x.dtype)[np.newaxis, :, np.newaxis, np.newaxis].cuda()
    if typ:
        mean = mean.half()
        var = var.half()
    if mode == 'normal':
        return (x - mean) / var
    elif mode == 'inv':
        return x * var + mean

class ClsSolver(BaseSolver):

    def __init__(self, config_file):
        self.config_file = config_file
        self.prototype_info = EasyDict()
        self.config = parse_config(config_file)
        self.setup_env()
        self.build_model()
        self.build_optimizer()
        self.build_data()
        self.build_lr_scheduler()

    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        self.prototype_info.world_size = self.dist.world_size
        # directories
        self.path = EasyDict()
        totalResult = os.path.join("/model",
                                   time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        os.mkdir(totalResult)
        self.path.root_path = totalResult
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, 'results')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)
        # tb_logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)
        # logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.info(f'config: {pprint.pformat(self.config)}')
        if 'SLURM_NODELIST' in os.environ:
            self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")
        # load pretrain checkpoint
        if hasattr(self.config.saver, 'pretrain'):
            self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
            self.logger.info(f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
            if hasattr(self.config.saver.pretrain, 'ignore'):
                self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
        else:
            self.state = {}
            self.state['last_iter'] = 0
        # others
        torch.backends.cudnn.benchmark = True

    def build_model(self):
        if hasattr(self.config, 'lms'):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.info('Enable large model support, limit of {}G!'.format(
                    self.config.lms.kwargs.limit))

        self.model = model_entry(self.config.model)
        self.prototype_info.model = self.config.model.type
        self.model.cuda()

        count_params(self.model)
        count_flops(self.model, input_shape=[
                    1, 3, self.config.data.input_size, self.config.data.input_size])


        self.model = DistModule(self.model, self.config.dist.sync)

        new_state_dict = OrderedDict()
        for k, v in self.state['model'].items():
            name = "module."+k[:]  # remove `module.`
            new_state_dict[name] = v
#         self.state['last_iter'] = 0
        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])

    def build_optimizer(self):

        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        # make param_groups
        pconfig = {}

        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}

        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        param_group, type2num = param_group_all(self.model, pconfig)

        opt_config.kwargs.params = param_group

        self.optimizer = optim_entry(opt_config)

        self.optimizer = SWA(self.optimizer, swa_start=37350,
                        swa_freq=996, swa_lr=0.0001)

        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])

        # EMA
        if self.config.ema.enable:
            self.config.ema.kwargs.model = self.model
            self.ema = EMA(**self.config.ema.kwargs)
        else:
            self.ema = None

        if 'ema' in self.state and self.ema:
            self.ema.load_state_dict(self.state['ema'])

    def build_lr_scheduler(self):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def build_data(self):
        self.config.data.last_iter = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.train_data = build_imagenet_train_dataloader(self.config.data)
        else:
            self.train_data = build_custom_dataloader('train', self.config.data)

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.val_data = build_imagenet_test_dataloader(self.config.data)
        else:
            self.val_data = build_custom_dataloader('test', self.config.data)

    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        self.model.train()

        label_smooth = self.config.get('label_smooth', 0.0)
        self.num_classes = self.config.model.kwargs.get('num_classes', 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes
        if label_smooth > 0:
            self.logger.info('using label_smooth: {}'.format(label_smooth))
            self.criterion = LabelSmoothCELoss(label_smooth, self.num_classes)
        else:
            weights = [1.0, 1.0]
            self.logger.info('Using weight: {}'.format(weights))
            class_weights = torch.FloatTensor(weights).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.mixup = self.config.get('mixup', 1.0)
        self.cutmix = self.config.get('cutmix', 0.0)
        self.switch_prob = 0.0
        if self.mixup < 1.0:
            self.logger.info('using mixup with alpha of: {}'.format(self.mixup))
        if self.cutmix > 0.0:
            self.logger.info('using cutmix with alpha of: {}'.format(self.cutmix))
        if self.mixup < 1.0 and self.cutmix > 0.0:
            # the probability of switching mixup to cutmix if both are activated
            self.switch_prob = self.config.get('switch_prob', 0.5)
            self.logger.info('switching between mixup and cutmix with probility of: {}'.format(self.switch_prob))

    def train(self):
        init_top1 = 0.

        self.pre_train()
        total_step = len(self.train_data['loader'])
        start_step = self.state['last_iter'] + 1
        end = time.time()
        for i, batch in enumerate(self.train_data['loader']):
            input = batch['image']
            target = batch['label']
            curr_step = start_step + i
            self.lr_scheduler.step(curr_step)
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]
            # measure data loading time
            self.meters.data_time.update(time.time() - end)
            # transfer input to gpu
            target = target.squeeze().cuda().long()
            input = input.cuda()
            # mixup
            if self.mixup < 1.0 and random.uniform(0, 1) > self.switch_prob:
                input, target_a, target_b, lam = mixup_data(input, target, self.mixup)
            # cutmix
            elif self.cutmix > 0.0:
                input, target_a, target_b, lam = cutmix_data(input, target, self.cutmix)
            # forward
            logits = self.model(input)
            logits = logits[:, dict]

            # mixup
            if self.mixup < 1.0 or self.cutmix > 0.0:
                loss = mix_criterion(self.criterion, logits, target_a, target_b, lam)
                loss /= self.dist.world_size
            else:
                loss = self.criterion(logits, target) / self.dist.world_size
            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, target, topk=(1, self.topk))

            reduced_loss = loss.clone()
            reduced_prec1 = prec1.clone() / self.dist.world_size
            reduced_prec5 = prec5.clone() / self.dist.world_size

            self.meters.losses.reduce_update(reduced_loss)
            self.meters.top1.reduce_update(reduced_prec1)
            self.meters.top5.reduce_update(reduced_prec5)

            # compute and update gradient
            self.optimizer.zero_grad()
            loss.backward()
            self.model.sync_gradients()
            self.optimizer.step()

            # EMA
            if self.ema is not None:
                self.ema.step(self.model, curr_step=curr_step)
            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)

            # training logger
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar('acc5_train', self.meters.top5.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()+remain_secs))
                log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                    f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                    f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                    f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                    f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                    f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                    f'LR {current_lr:.4f}\t' \
                    f'Remaining Time {remain_time} ({finish_time})'
                self.logger.info(log_msg)

            # testing during training
            if curr_step > 0 and curr_step % self.config.saver.val_freq == 0:
                if curr_step >= 37350:
                    print("Evaluate with SWA!")
                    self.optimizer.swap_swa_sgd()
                    self.optimizer.bn_update(self.train_data['loader'], self.model, device='cuda')

                metrics = self.evaluate()
                # testing logger
                if self.dist.rank == 0 and self.config.data.test.evaluator.type == 'imagenet':
                    metric_key = 'top{}'.format(self.topk)
                    self.tb_logger.add_scalar('acc1_val', metrics.metric['top1'], curr_step)
                    self.tb_logger.add_scalar('acc5_val', metrics.metric[metric_key], curr_step)

                # save ckpt
                if self.dist.rank == 0:
                    if self.config.saver.save_many:
                        ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                    else:
                        if metrics.metric['top1'] > init_top1:
                            ckpt_name = f'{self.path.save_path}/ckpt_best.pth.tar'
                            init_top1 = metrics.metric['top1']
                            self.logger.info("Get Best!")
                        else:
                            ckpt_name = f'{self.path.save_path}/ckpt_last.pth.tar'
                    self.state['model'] = self.model.state_dict()
                    self.state['optimizer'] = self.optimizer.state_dict()
                    self.state['last_iter'] = curr_step
                    if self.ema is not None:
                        self.state['ema'] = self.ema.state_dict()
                    torch.save(self.state, ckpt_name)

                if curr_step >= 37350:
                    self.optimizer.swap_swa_sgd()

            end = time.time()

        self.optimizer.swap_swa_sgd()
        self.optimizer.bn_update(self.train_data['loader'], self.model, device='cuda')
        metrics = self.evaluate()
        # testing logger
        metric_key = 'top{}'.format(self.topk)
        self.tb_logger.add_scalar('acc1_val', metrics.metric['top1'], curr_step)
        self.tb_logger.add_scalar('acc5_val', metrics.metric[metric_key], curr_step)
        if self.config.saver.save_many:
            ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
        else:
            if metrics.metric['top1'] > init_top1:
                ckpt_name = f'{self.path.save_path}/ckpt_best.pth.tar'
                init_top1 = metrics.metric['top1']
                self.logger.info("Get Best!")
            else:
                ckpt_name = f'{self.path.save_path}/ckpt_last.pth.tar'
        self.state['model'] = self.model.state_dict()
        self.state['optimizer'] = self.optimizer.state_dict()
        self.state['last_iter'] = curr_step
        if self.ema is not None:
            self.state['ema'] = self.ema.state_dict()
        torch.save(self.state, ckpt_name)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        imagenetc_flag = self.config.data.test.get("imagenet_c", False)
        if imagenetc_flag:

            noise_list = []

            writer = {'noise': {'gaussian_noise': {}, 'shot_noise': {}, 'impulse_noise': {}},
                 'blur': {'defocus_blur': {},
                          'glass_blur': {},
                          'motion_blur': {},
                          'zoom_blur': {}},
                 'weather': {'snow': {}, 'frost': {}, 'fog': {}, 'brightness': {}},
                 'digital': {'contrast': {},
                             'elastic_transform': {},
                             'pixelate': {},
                             'jpeg_compression': {}},
                 'extra': {'speckle_noise': {},
                           'spatter': {},
                           'gaussian_blur': {},
                           'saturate': {}}}
            for noise in writer:
                for noise_type in writer[noise]:
                    for i in range(1, 6):
                        res_file = os.path.join(self.path.result_path,
                                                f'{noise}-{noise_type}-{i}-results.txt.rank{self.dist.rank}')
                        writer[noise][noise_type][i] = open(res_file, 'w')
                        noise_list.append(os.path.join(self.path.result_path,
                                                       f'{noise}-{noise_type}-{i}-results.txt.rank'))
            noise_list = sorted(noise_list)
        else:
            res_file = os.path.join(self.path.result_path, f'results.txt.rank{self.dist.rank}')
            writer = open(res_file, 'w')

        bn_lst = []
        for batch_idx, batch in enumerate(self.val_data['loader']):
            if batch_idx % 10 == 0:
                info_str = f"[{batch_idx}/{len(self.val_data['loader'])}] ";
                info_str += f"{batch_idx * 100 / len(self.val_data['loader']):.6f}%"
                self.logger.info(info_str)


            input = batch['image']
            label = batch['label']
            input = input.cuda()
            label = label.squeeze().view(-1).cuda().long()
            # compute output
            logits= self.model(input)
            logits = logits[:, dict]

            # bn_lst.append(bn)
            # if batch_idx % 10 == 0:
            #     torch.save(bn_lst, "./bn_clean.pth")
            #     exit(0)
            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            # update batch information
            batch.update({'prediction': preds})
            batch.update({'score': scores})
            # save prediction information
            self.val_data['loader'].dataset.dump(writer, batch)
        if imagenetc_flag:
            for noise in writer:
                for noise_type in writer[noise]:
                    for i in range(1, 6):
                        writer[noise][noise_type][i].close()
        else:
            writer.close()
        link.barrier()
        if imagenetc_flag:
            for idx, file_prefix in enumerate(noise_list):
                if idx % self.dist.world_size == self.dist.rank:
                    # print(f"idx: {idx}, rank: {self.dist.rank}, {file_prefix}")
                    self.val_data['loader'].dataset.evaluate(file_prefix)
            link.barrier()
            if self.dist.rank == 0:
                self.val_data['loader'].dataset.merge_eval_res(self.path.result_path)
            metrics = {}
        else:
            if self.dist.rank == 0:
                metrics = self.val_data['loader'].dataset.evaluate(res_file)
                self.logger.info(json.dumps(metrics.metric, indent=2))
            else:
                metrics = {}
        link.barrier()

        # broadcast metrics to other process
        metrics = broadcast_object(metrics)
        self.model.train()
        return metrics

    @torch.no_grad()
    def test(self):
        self.model.eval()
        res_file = os.path.join(self.path.result_path, f'resultsTest.txt')
        writer = open(res_file, 'w')
        for batch_idx, batch in enumerate(self.val_data['loader']):
            input = batch['image']
            input = input.cuda()
            # compute output
            logits = self.model(input)
            logits = logits[:, dict]

            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            tmp = {}
            tmp.update({'prediction': preds})
            tmp.update({'filename': batch['filename']})
            # save prediction information
            self.val_data['loader'].dataset.dump(writer, tmp)

        writer.close()

    @torch.no_grad()
    def test_TTA(self):
        import ttach as tta
        transforms = tta.Compose(
            [
                tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.FiveCrops(192, 192),
                tta.Resize([(224, 224)])
            ]
        )

        tta_model = tta.ClassificationTTAWrapper(self.model, transforms)

        self.model.cuda().eval()
        res_file = os.path.join(self.path.result_path, f'resultsTest.txt')
        writer = open(res_file, 'w')
        for batch in self.val_data['loader']:
            image = batch['image']
            pred = tta_model(image.cuda())
            preds = torch.argmax(pred, dim=1).detach()
            preds = preds.view(-1)

            tmp = {}
            tmp.update({'prediction': preds})
            tmp.update({'filename': batch['filename']})
            # save prediction information
            self.val_data['loader'].dataset.dump(writer, tmp)

        writer.close()

    @torch.no_grad()
    def test_TTA_custom(self):
        from torchvision import transforms

        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.)),
            transforms.RandomOrder([
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.4),
                transforms.RandomApply([transforms.RandomRotation(degrees=[90, 90])], p=0.4),
            ]),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model.eval()
        res_file = os.path.join(self.path.result_path, f'resultsTest.txt')
        writer = open(res_file, 'w')
        for batch in self.val_data['loader']:
            image = batch['image'].cuda()
            image = normalize(image, 'inv')
            tmp_log = None
            for i in range(32):
                aug_image = augmentation(image)
                logits = self.model(aug_image)
                if i ==0:
                    tmp_log = logits
                else:
                    tmp_log += logits
            _, preds = tmp_log.data.topk(k=1, dim=1)
            preds = preds.view(-1)

            tmp = {}
            tmp.update({'prediction': preds})
            tmp.update({'filename': batch['filename']})
            # save prediction information
            self.val_data['loader'].dataset.dump(writer, tmp)

        writer.close()

    @torch.no_grad()
    def test_label(self):
        self.model.eval()
        res_file = os.path.join(self.path.result_path, f'labelTest.txt')
        writer = open(res_file, 'w')
        for batch_idx, batch in enumerate(self.val_data['loader']):
            input = batch['image']
            input = input.cuda()
            # compute output
            logits = self.model(input)
            preds = F.softmax(logits, dim=1)
            # compute prediction
            scores, cates = preds.data.topk(k=1, dim=1)
            scores = scores.view(-1).detach().cpu().numpy()
            cates = cates.view(-1).detach().cpu().numpy()
            for i in range(len(scores)):
                if scores[i]>0.85:
                    writer.write("{{\"filename\": \"{}\", \"label\": {}, \"label_name\": \"{}\"}}\n".format(os.path.basename(batch['filename'][i]), cates[i], cates[i]))
            # save prediction information

        writer.close()

    def test_generate_adv(self):
        import foolbox as fb
        import cv2

        def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
            """
            将tensor保存为cv2格式
            :param input_tensor: 要保存的tensor
            :param filename: 保存的文件名
            """
            assert len(input_tensor.shape) == 3
            # 复制一份
            input_tensor = input_tensor.clone().detach()
            # 到cpu
            input_tensor = input_tensor.to(torch.device('cpu'))
            # 反归一化
            # input_tensor = unnormalize(input_tensor)
            # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
            input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            # RGB转BRG
            input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, input_tensor)

        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        self.model.eval()
        f_model = fb.PyTorchModel(self.model, bounds=(0, 1), device="cuda:" + str(int(self.dist.rank % 8)),
                                  preprocessing=preprocessing)
        res_file = os.path.join(self.path.result_path, f'labelAdv.txt')
        writer = open(res_file, 'w')
        for batch_idx, batch in enumerate(self.val_data['loader']):
            input = batch['image']
            target = batch['label']

            target = target.squeeze().cuda().long()
            input = input.cuda()

            adv_input_01 = normalize(input, 'inv')
            pgdlinf_att = fb.attacks.LinfProjectedGradientDescentAttack(rel_stepsize=1 / 15, steps=20)
            adv_fbpgd_linf, _, success = pgdlinf_att(f_model, adv_input_01, target, epsilons=16 / 255)

            for i in range(len(adv_fbpgd_linf)):
                filename = os.path.basename(batch['filename'][i]).replace(".JPEG", "_adv.JPEG")
                filePath = os.path.join("/media/disk/zyl/Competition/Data/adv", filename)
                save_image_tensor2cv2(adv_fbpgd_linf[i], filePath)
                writer.write("{{\"filename\": \"{}\", \"label\": {}, \"label_name\": \"{}\"}}\n".format(
                    filename, batch['label'][i], batch['label'][i]))
            # save prediction information
        writer.close()

@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', type=str, default="/code/exprs/nips_benchmark/pgd_adv_train/vit_base_patch16_224/customconfigSwin.yaml")
    parser.add_argument('--evaluate', action='store_true', default=False)

    args = parser.parse_args()
    # build solver
    solver = ClsSolver(args.config)
    # evaluate or train
    if args.evaluate:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn('Evaluating without resuming any solver checkpoints.')
        solver.test_generate_adv()
        if solver.ema is not None:
            solver.ema.load_ema(solver.model)
            solver.test_generate_adv()
    else:
        if solver.config.data.last_iter < solver.config.data.max_iter:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
