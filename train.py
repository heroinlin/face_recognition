import argparse
import os
import sys
import time
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from tensorboardX import SummaryWriter

from backbones import init_backbone
from heads import init_head
from loss import init_loss
from config import Cfg
from datasets import init_database
from utils.avgmeter import AverageMeter
from utils.eval import accuracy
from evaluate import Evaluation, buffer_val
from utils.logger import Logger
from utils.random_seed import worker_init_fn
from utils.combine_conv_bn import fuse_module
from utils.separate_bn_paras import apply_weight_decay


def loss_forward(criterions, feature, outputs, targets):
    losses = list()
    for index, criterion_name in enumerate(criterions.keys()):
        if criterion_name in ['FocalLoss', 'Softmax']:
            losses.append(criterions[criterion_name](outputs, targets))
        else:
            losses.append(criterions[criterion_name](feature, targets))
    return losses


class NetTrain:
    def __init__(self, backbone, head, train_data_loader, val_dataset,
                 criterions=None,
                 loss_weights=None,
                 optimizer=None,
                 backbone_name='backbone',
                 head_name='head',
                 lowest_train_loss=5,
                 use_cuda=True, gpu_list=None):
        self.train_data_loader = train_data_loader
        self.backbone = backbone
        self.head = head
        self.backbone_name = backbone_name
        self.head_name = head_name
        self.loss_forward = loss_forward
        self.evaluation = Evaluation(val_dataset, 'lfw',self.batch_inference)
        self.criterions = criterions
        self.loss_weights = loss_weights
        self.optimizer = optimizer
        self.lowest_train_loss = lowest_train_loss
        self.use_cuda = use_cuda
        self.gpu_list = gpu_list
        self.writer = SummaryWriter()
        sys.stdout = Logger()
        self.epoch = 0
        self.max_epoch = 400
        self.combine_conv_bn_epoch = 150
        self.init_meter()

        if self.criterions is None:
            self.criterions = {'xent': torch.nn.CrossEntropyLoss()}
        if self.loss_weights is None:
            self.loss_weights = torch.as_tensor([1.0]*len(self.criterions))
        if self.use_cuda:
            self.backbone.cuda()
            self.head.cuda()
            self.loss_weights = self.loss_weights.cuda()
        if self.gpu_list is None:
            self.gpu_list = range(torch.cuda.device_count())

    def init_meter(self):
        self.accuracy_top_1 = AverageMeter()
        self.accuracy_top_5 = AverageMeter()
        self.total_losses_meter = AverageMeter()
        self.loss_meters = list()
        for index, criterion_name in enumerate(self.criterions.keys()):
            self.loss_meters.append(AverageMeter())

    def reset_meter(self):
        self.accuracy_top_1.reset()
        self.accuracy_top_5.reset()
        self.total_losses_meter.reset()
        for index, criterion_name in enumerate(self.criterions.keys()):
            self.loss_meters[index].reset()

    def load_checkpoint(self, check_point, finetune=False, pretrained=False):
        check_point = torch.load(check_point)
        if pretrained:
            self.backbone.load_state_dict(check_point)
            return
        if finetune:
            # 导入特征提取部分网络参数
            mapped_state_dict = self.backbone.state_dict()
            for key, value in check_point['backbone'].items():
                mapped_state_dict[key] = value
            self.backbone.load_state_dict(mapped_state_dict)
            # 导入特征提取部分优化子参数
            optimizer_state_dict = self.optimizer.state_dict()
            param_len = len(optimizer_state_dict['param_groups'][0]['params'])
            for index in range(param_len):
                optimizer_state_dict['state'].update({
                    optimizer_state_dict['param_groups'][0]['params'][index]:
                        check_point['optimizer']['state'].get(
                            check_point['optimizer']['param_groups'][0]['params'][index])})
            self.optimizer.load_state_dict(optimizer_state_dict)
        else:
            self.lowest_train_loss = check_point['loss']
            self.epoch = check_point['epoch']
            if self.epoch > 150:
                fuse_module(self.backbone)
                fuse_module(self.head)
            print("lowest_train_loss: ", self.lowest_train_loss)
            mapped_state_dict = self.backbone.state_dict()
            for key, value in check_point['backbone'].items():
                mapped_state_dict[key] = value
            self.backbone.load_state_dict(mapped_state_dict)

            mapped_state_dict = self.head.state_dict()
            for key, value in check_point['head'].items():
                mapped_state_dict[key] = value
            self.head.load_state_dict(mapped_state_dict)
            self.optimizer.load_state_dict(check_point['optimizer'])

    def finetune_model(self):
        if isinstance(self.backbone, torch.nn.DataParallel):
            backbone_named_children = self.backbone.module.named_children()
        else:
            backbone_named_children = self.backbone.named_children()
        if isinstance(self.head, torch.nn.DataParallel):
            head_named_children = self.head.module.named_children()
        else:
            head_named_children = self.head.named_children()
        for name, module in backbone_named_children:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        for name, module in head_named_children:
            module.train()
            for p in module.parameters():
                p.requires_grad = True

    def set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def adjust_lr_exp(self, optimizer, ep, total_ep, start_decay_at_ep):
        """Decay exponentially in the later phase of training. All parameters in the
        optimizer share the same learning rate.

        Args:
          optimizer: a pytorch `Optimizer` object
          base_lr: starting learning rate
          ep: current epoch, ep >= 1
          total_ep: total number of epochs to train
          start_decay_at_ep: start decaying at the BEGINNING of this epoch

        Example:
          base_lr = 2e-4
          total_ep = 300
          start_decay_at_ep = 201
          It means the learning rate starts at 2e-4 and begins decaying after 200
          epochs. And training stops after 300 epochs.

        NOTE:
          It is meant to be called at the BEGINNING of an epoch.
        """
        assert ep >= 1, "Current epoch number should be >= 1"
        if ep < start_decay_at_ep:  # warm-up
            for g in optimizer.param_groups:
                g['lr'] = (g['initial_lr'] * 0.1 * (10 ** (float(ep) / start_decay_at_ep)))
                print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))
        else:
            for g in optimizer.param_groups:
                g['lr'] = (g['initial_lr'] * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                                        / (total_ep + 1 - start_decay_at_ep))))
                print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))

    def eval(self):
        self.backbone.eval()
        self.head.eval()
        accuracy, best_thresholds, roc_curve_tensor = self.evaluation.evaluate()
        buffer_val(self.writer, 'lfw', accuracy, best_thresholds, roc_curve_tensor, self.epoch)
        # self.evaluation.eval_rerank()

    def train(self, epoches=10, save_flag=True, finetune=False):
        if len(self.gpu_list) > 1:
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=self.gpu_list)
            self.head = torch.nn.DataParallel(self.head, device_ids=self.gpu_list)
            cudnn.benchmark = True
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1, last_epoch=self.epoch)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60, 100],
        #                                                  gamma=0.1, last_epoch=self.epoch)
        while self.epoch < epoches:
            print("Epoch: ", self.epoch)
            self.adjust_lr_exp(self.optimizer, self.epoch + 1, epoches, int(finetune) * 10 + 10)
            if self.epoch % 10 == 0:
                print(self.optimizer)
            self.reset_meter()
            if finetune and self.epoch < 10:
                self.finetune_model()
            else:
                self.backbone.train()
                self.head.train()
            if self.epoch == self.combine_conv_bn_epoch:
                # 冻结BN参数更新
                # self.model.apply(self.set_bn_eval)
                # 融合conv+bn
                fuse_module(self.backbone)
                fuse_module(self.head)
            self.train_epoch()
            # scheduler.step()
            # if (self.epoch + 1) % 10 == 0:
            #     self.eval()
            if save_flag:
                self.save_model()
                self.save_model(False, False)
            self.epoch += 1
        torch.cuda.empty_cache()
        self.writer.close()
        print("Finished training.")

    def search_learn_rate(self):
        if self.use_cuda:
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=self.gpu_list)
            self.head = torch.nn.DataParallel(self.head, device_ids=self.gpu_list)
            cudnn.benchmark = True
        print(self.optimizer)
        lr_mult = (1 / 1e-5) ** (1 / self.train_data_loader.__len__())
        self.backbone.train()
        self.head.train()
        self.reset_meter()
        train_data_batch = self.train_data_loader.__len__()
        batch_iterator = iter(self.train_data_loader)
        for step in range(train_data_batch):
            start = time.time()

            images, targets = next(batch_iterator)
            self.batch_inference(images, targets)

            end = time.time()
            batch_time = end - start
            eta = int(batch_time * ((train_data_batch - step) + (self.max_epoch - self.epoch) * train_data_batch))
            for g in self.optimizer.param_groups:
                g['lr'] = (g['lr'] * lr_mult)
            if (step + 1) % 10 == 0:
                print_infos = 'Epoch:{}/{} || Epochiter: {}/{} || Batchtime: {:.4f} s || ETA: {} || ' \
                    .format(self.epoch, self.max_epoch, step, train_data_batch,
                            batch_time, str(datetime.timedelta(seconds=eta)))
                print_infos += 'acc_top1: {:>.4f}, acc_top5: {:>.4f}, total_loss: {:>.4f}( {:>.4f})'.format(
                    self.accuracy_top_1.avg, self.accuracy_top_5.avg,
                    self.total_losses_meter.val, self.total_losses_meter.avg)
                for index, criterion_name in enumerate(self.criterions.keys()):
                    print_infos = print_infos + f", {criterion_name}: {self.loss_meter[index].val:>.4f}" \
                                                f"({self.loss_meter[index].avg:>.4f})"
                print(print_infos)
                self.writer.add_scalar('loss/loss', self.total_losses_meter.val, step)
                self.writer.add_scalar('loss/total_loss', self.total_losses_meter.val, step)
                for index, criterion_name in enumerate(self.criterions.keys()):
                    self.writer.add_scalar(f'loss/{criterion_name}', self.loss_meter[index].val,
                                           step)
                self.writer.add_scalar('acc/acc_top1', self.accuracy_top_1.val, step)
                self.writer.add_scalar('acc/acc_top5', self.accuracy_top_5.val, step)
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], step)
            if (step + 1) % 100 == 0:
                print(self.optimizer)
        torch.cuda.empty_cache()
        self.writer.close()
        print("Finished training.")

    def train_epoch(self):
        train_data_batch = self.train_data_loader.__len__()
        batch_iterator = iter(self.train_data_loader)
        for step in range(train_data_batch):
            start = time.time()

            images, targets = next(batch_iterator)
            self.batch_inference(images, targets)

            end = time.time()
            batch_time = end - start
            eta = int(batch_time * ((train_data_batch - step) + (self.max_epoch - self.epoch) * train_data_batch))
            if step % 20 == 0:
                print_infos = 'Epoch:{}/{} || Epochiter: {}/{} || Batchtime: {:.4f} s || ETA: {} || ' \
                    .format(self.epoch, self.max_epoch, step, train_data_batch,
                            batch_time, str(datetime.timedelta(seconds=eta)))
                print_infos += ' acc_top1: {:>.4f}, acc_top5: {:>.4f}, total_loss: {:>.4f}( {:>.4f})'.format(
                    self.accuracy_top_1.avg, self.accuracy_top_5.avg,
                    self.total_losses_meter.val, self.total_losses_meter.avg)
                for index, criterion_name in enumerate(self.criterions.keys()):
                    print_infos = print_infos + f", {criterion_name}: {self.loss_meters[index].val:>.4f}" \
                                                f"({self.loss_meters[index].avg:>.4f})"
                print(print_infos)
            if step % 100 == 0:
                # Window
                # self.writer.add_image('Image', images, step + self.epoch * train_data_batch)
                # Linux
                # self.writer.add_image('Image', image, step + self.epoch * train_data_batch, dataformats='NCHW')
                for name, param in self.backbone.named_parameters():
                    self.writer.add_histogram(name, param.clone().cpu().data.numpy(),
                                              step + self.epoch * train_data_batch)
                for name, param in self.head.named_parameters():
                    self.writer.add_histogram(
                        name,
                        param.clone().cpu().data.numpy(),
                        step + self.epoch * train_data_batch)
                self.writer.add_scalar('loss/total_loss', self.total_losses_meter.val, step + self.epoch * train_data_batch)
                for index, criterion_name in enumerate(self.criterions.keys()):
                    self.writer.add_scalar(f'loss/{criterion_name}', self.loss_meters[index].val,
                                           step + self.epoch * train_data_batch)
                self.writer.add_scalar('acc/acc_top1', self.accuracy_top_1.val, step + self.epoch * train_data_batch)
                self.writer.add_scalar('acc/acc_top5', self.accuracy_top_5.val, step + self.epoch * train_data_batch)
        print("Total train loss:", self.total_losses_meter.avg)

    def save_model(self, save_head=True, save_optimizer=True):
        if self.total_losses_meter.avg < self.lowest_train_loss or self.total_losses_meter.avg < 2.0:
            state = {
                'backbone': self.backbone.module.state_dict() if self.use_cuda else self.backbone.state_dict(),
                'loss': self.total_losses_meter.avg,
                'epoch': self.epoch + 1
            }
            if save_optimizer:
                state.update({'optimizer': self.optimizer.state_dict()})
            model_name = self.backbone_name
            if save_head:
                state.update({'head': self.head.module.state_dict() if self.use_cuda else self.head.state_dict()})
                model_name = '_'.join([self.backbone_name, self.head_name])
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            save_path = './checkpoints/{}_{}_{:.04f}.pth'.format(model_name, self.epoch,
                                                                 self.total_losses_meter.avg)
            torch.save(state, save_path)
        if self.total_losses_meter.avg < self.lowest_train_loss:
            self.lowest_train_loss = self.total_losses_meter.avg

    def batch_inference(self, images, targets, backward=True):
        if torch.__version__ < "0.4.0":
            images = torch.autograd.Variable(images)
            targets = torch.autograd.Variable(targets)
        if self.use_cuda:
            images = images.cuda()
            targets = targets.cuda()
        if not self.backbone.training and not self.head.training:
            features = self.backbone(images)
            return features
        features = self.backbone(images)
        outputs = self.head(features, targets.long())
        total_loss = 0

        losses = self.loss_forward(self.criterions, features, outputs, targets)
        accuracy_top_1, accuracy_top_5 = accuracy(outputs, targets, (1, 5))
        total_loss = torch.stack(losses).mul(self.loss_weights).sum()

        if backward:
            self.optimizer.zero_grad()
            total_loss.backward()
            apply_weight_decay(self.backbone)
            apply_weight_decay(self.head)
            self.optimizer.step()

        losses_value = []
        for index, criterion_name in enumerate(self.criterions.keys()):
            losses_value.append(losses[index].item())
        total_loss_value = total_loss.item()
        accuracy_top_1_value = accuracy_top_1.item()
        accuracy_top_5_value = accuracy_top_5.item()

        for index, criterion_name in enumerate(self.criterions.keys()):
            self.loss_meters[index].update(losses_value[index], targets.size(0))
        self.total_losses_meter.update(total_loss_value, targets.size(0))
        self.accuracy_top_1.update(accuracy_top_1_value, targets.size(0))
        self.accuracy_top_5.update(accuracy_top_5_value, targets.size(0))
        return outputs


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true", default=Cfg.Train.train)
    parser.add_argument("--search_learn_rate", dest="search_learn_rate", action="store_true",
                        default=Cfg.Train.search_learn_rate)
    parser.add_argument('--backbone_name', type=str, default=Cfg.Net.backbone_name,
                        help='backbone name')
    parser.add_argument('--head_name', type=str, default=Cfg.Net.head_name,
                        help='head name')
    parser.add_argument("--test", dest="test", action="store_true", default=Cfg.Train.test)
    parser.add_argument('-p', '--pretrained', dest='pretrained', action="store_true", default=Cfg.Net.pretrained)
    parser.add_argument("--resume", dest="resume", action="store_true", default=Cfg.Train.resume)
    parser.add_argument("--finetune", dest="finetune", action="store_true", default=Cfg.Train.finetune)
    parser.add_argument('--checkpoint', type=str, default=Cfg.Train.checkpoint,
                        help='weights file path')
    parser.add_argument('--feature_size', type=int, default=Cfg.Net.feature_size,
                        help='model feature size')
    parser.add_argument('--margin', type=float, default=Cfg.Loss.margin,
                        help='margin')
    parser.add_argument('--gpu_list', default=Cfg.Train.gpu_list, nargs='+', type=int,
                        help='gpu device list')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    print(args)
    gpu_list = args.gpu_list
    if gpu_list is None:
        gpu_list = range(torch.cuda.device_count())
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(gpu_device) for gpu_device in gpu_list])
    use_cuda = torch.cuda.is_available()
    train_dataset = init_database(Cfg.Database.name, Cfg.Database.image_root, Cfg.Database.pickle_folder)
    val_dataset = Cfg.Database.val_image_root

    train_data_loader = data.DataLoader(dataset=train_dataset,
                                        batch_size=Cfg.Sample.batch_size,
                                        shuffle=False,
                                        num_workers=Cfg.Sample.num_workers,
                                        worker_init_fn=worker_init_fn,
                                        drop_last=True)
    criterions = {}
    for loss_name in Cfg.Loss.list:
        criterions.update({loss_name: init_loss(loss_name)})
    loss_weights = torch.as_tensor(Cfg.Loss.weight_list)
    backbone = init_backbone(name=args.backbone_name, input_size=Cfg.Database.image_size )
    head = init_head(name=args.head_name,
                     in_features=512,
                     out_features=512,
                     device_id=gpu_list)
    if args.finetune:
        base_params = list(map(id, backbone.parameters()))
        classfier_params = list(map(id, head.parameters()))
        optimizer = torch.optim.Adam([{'params': base_params,
                                       'initial_lr': 0.1 * Cfg.Train.initial_lr,
                                       'lr': 0.1 * Cfg.Train.lr},
                                      {'params': head.parameters(),
                                       'initial_lr': Cfg.Train.initial_lr}],
                                     lr=Cfg.Train.lr,
                                     amsgrad=True,
                                     weight_decay=Cfg.Train.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': backbone.parameters(),
                                       'initial_lr': Cfg.Train.initial_lr},
                                      {'params': head.parameters(),
                                       'initial_lr': Cfg.Train.initial_lr}],
                                     lr=Cfg.Train.lr,
                                     amsgrad=True,
                                     weight_decay=Cfg.Train.weight_decay)
        # optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': Cfg.Train.initial_lr}],
        #                             lr=Cfg.Train.lr, momentum=0.9, weight_decay=0.0005)
    print(train_dataset.__len__(), train_dataset.class_num())
    fine_tuner = NetTrain(backbone, head, train_data_loader, val_dataset,
                          criterions=criterions,
                          loss_weights=loss_weights,
                          optimizer=optimizer,
                          backbone_name=args.backbone_name,
                          head_name=args.head_name,
                          lowest_train_loss=10)
    fine_tuner.max_epoch = Cfg.Train.epochs
    if args.pretrained:
        print("loading pretrained model...")
        fine_tuner.load_checkpoint(args.checkpoint, pretrained=True)
    if args.resume:
        print("continue to train...")
        fine_tuner.load_checkpoint(args.checkpoint, finetune=args.finetune)
        # fine_tuner.load_checkpoint(args.checkpoint, finetune=False)
        fine_tuner.epoch = 1
    if args.train:
        fine_tuner.train(epoches=Cfg.Train.epochs, finetune=args.finetune)
    if args.test:
        fine_tuner.eval()
    if args.search_learn_rate:
        fine_tuner.search_learn_rate()
