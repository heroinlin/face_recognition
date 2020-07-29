import argparse
import os
import sys
import time
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from tensorboardX import SummaryWriter

import models
from config import Cfg
from datasets import init_database
from loss import CrossEntropyLoss, AngularPenaltySMLoss
from loss import TripletLoss as TripletLoss
from utils.avgmeter import AverageMeter
from utils.eval import accuracy, Evaluation
from utils.logger import Logger
from utils.random_seed import worker_init_fn
from utils.combine_conv_bn import fuse_module


class NetTrain:
    def __init__(self, model, train_data_loader, val_dataset,
                 criterion_xent=None, criterion_htri=None,
                 optimizer=None, 
                 model_name='model', lowest_train_loss=5, 
                 use_cuda=True, gpu_list=None):
        self.train_data_loader = train_data_loader
        self.model = model
        self.model_name = model_name
        self.evaluation = Evaluation(val_dataset.data_list, self.batch_inference)
        self.criterion_xent = criterion_xent
        self.criterion_htri = criterion_htri
        self.optimizer = optimizer
        self.optimizer_center = optimizer_center
        self.lowest_train_loss = lowest_train_loss
        self.use_cuda = use_cuda
        self.gpu_list = gpu_list
        self.writer = SummaryWriter()
        sys.stdout = Logger()
        self.epoch = 0
        self.max_epoch = 400
        self.weight_cent = weight_cent
        self.xent_losses_meter = AverageMeter()
        self.htri_losses_meter = AverageMeter()
        self.total_losses_meter = AverageMeter()
        self.accuracy_top_1 = AverageMeter()
        self.accuracy_top_5 = AverageMeter()

        if self.criterion_xent is None:
            self.criterion_xent = torch.nn.CrossEntropyLoss()
        if self.use_cuda:
            self.model.cuda()
        if self.gpu_list is None:
            self.gpu_list = range(torch.cuda.device_count())

    def load_checkpoint(self, check_point, finetune=False):
        check_point = torch.load(check_point)
        if finetune:
            # 导入特征提取部分网络参数
            mapped_state_dict = self.model.state_dict()
            for key, value in check_point['net'].items():
                if 'classifier' in key:
                    continue
                mapped_state_dict[key] = value
            self.model.load_state_dict(mapped_state_dict)
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
            self.lowest_train_loss = check_point['val_acc']
            self.epoch = check_point['epoch']
            if self.epoch > 150:
                fuse_module(self.model)
            print("lowest_train_loss: ", self.lowest_train_loss)
            mapped_state_dict = self.model.state_dict()
            for key, value in check_point['net'].items():
                mapped_state_dict[key] = value
            # mapped_state_dict["bnneck.weight"] = self.model.bnneck.weight.data
            # mapped_state_dict["bnneck.bias"] = self.model.bnneck.bias.data
            # mapped_state_dict["bnneck.running_mean"] = self.model.bnneck.running_mean.data
            # mapped_state_dict["bnneck.running_var"] = self.model.bnneck.running_var.data
            self.model.load_state_dict(mapped_state_dict)

            # self.model.load_state_dict(check_point['net'])
            self.optimizer.load_state_dict(check_point['optimizer'])
            if Cfg.Loss.criterion_cent:
                self.criterion_cent.load_state_dict(check_point['centers'])
                self.optimizer_center.load_state_dict(check_point['optimizer_center'])
            # optimizer_state_dict = self.optimizer.state_dict()
            # param_len = len(optimizer_state_dict['param_groups'][0]['params'])
            # for index in range(param_len):
            #     optimizer_state_dict['state'].update({
            #         optimizer_state_dict['param_groups'][0]['params'][index]:
            #             check_point['optimizer']['state'].get(
            #                 check_point['optimizer']['param_groups'][0]['params'][index])})
            # self.optimizer.load_state_dict(optimizer_state_dict)
            # param_len = len(optimizer_state_dict['param_groups'][2]['params'])
            # for index in range(param_len):
            #     optimizer_state_dict['state'].update({
            #         optimizer_state_dict['param_groups'][2]['params'][index]:
            #             check_point['optimizer']['state'].get(
            #                 check_point['optimizer']['param_groups'][1]['params'][index])})

    def finetune_model(self, open_layers=['classifier']):
        if isinstance(self.model, torch.nn.DataParallel):
            named_children = self.model.module.named_children()
        else:
            named_children = self.model.named_children()

        for layer in open_layers:
            assert hasattr(model,
                           layer), '"{}" is not an attribute of the model, please provide the correct name'.format(
                layer)
        for name, module in named_children:
            if name in open_layers:
                module.train()
                for p in module.parameters():
                    p.requires_grad = True
            else:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

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
        if ep < start_decay_at_ep:
            for g in optimizer.param_groups:
                g['lr'] = (g['initial_lr'] * 0.1 * (10 ** (float(ep) / start_decay_at_ep)))
                print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))
        else:
            for g in optimizer.param_groups:
                g['lr'] = (g['initial_lr'] * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                                        / (total_ep + 1 - start_decay_at_ep))))
                print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))

    def eval(self):
        self.model.eval()
        self.evaluation.eval()
        # self.evaluation.eval_rerank()

    def train(self, epoches=10, save_flag=True, finetune=False):
        if self.use_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_list)
            cudnn.benchmark = True
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1, last_epoch=self.epoch)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60, 100],
        #                                                  gamma=0.1, last_epoch=self.epoch)
        while self.epoch < epoches:
            print("Epoch: ", self.epoch)
            self.adjust_lr_exp(self.optimizer, self.epoch + 1, epoches, int(finetune) * 10 + 10)
            if self.epoch % 10 == 0:
                print(self.optimizer)
            self.total_losses_meter.reset()
            self.xent_losses_meter.reset()
            self.htri_losses_meter.reset()
            self.accuracy_top_1.reset()
            self.accuracy_top_5.reset()
            if finetune and self.epoch < 10:
                self.finetune_model()
            else:
                self.model.train()
            # self.model.eval()
            if self.epoch > 150:
                # 冻结BN参数更新
                # self.model.apply(self.set_bn_eval)
                # 融合conv+bn
                fuse_module(self.model)
            self.train_epoch()
            # scheduler.step()
            if (self.epoch + 1) % 10 == 0:
                self.eval()
            if save_flag:
                self.save_model()
            self.epoch += 1
        torch.cuda.empty_cache()
        # self.writer.close()
        print("Finished training.")

    def search_learn_rate(self):
        if self.use_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_list)
            cudnn.benchmark = True
        print(self.optimizer)
        lr_mult = (1 / 1e-5) ** (1 / self.train_data_loader.__len__())
        self.model.train()
        self.total_losses_meter.reset()
        self.xent_losses_meter.reset()
        self.htri_losses_meter.reset()
        self.accuracy_top_1.reset()
        self.accuracy_top_5.reset()
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
                if Cfg.Loss.criterion_xent:
                    print_infos = print_infos + f", xent_loss: {self.xent_losses_meter.val:>.4f}" \
                                                f"({self.xent_losses_meter.avg:>.4f})"
                if Cfg.Loss.criterion_htri:
                    print_infos = print_infos + f", tri_loss: {self.htri_losses_meter.val:>.4f}" \
                                                f"({self.htri_losses_meter.avg:>.4f})"
                if Cfg.Loss.criterion_cent:
                    print_infos = print_infos + f", cent_loss: {self.cent_losses_meter.val:>.4f}" \
                                                f"({self.cent_losses_meter.avg:>.4f})"
                print(print_infos)
                self.writer.add_scalar('loss/loss', self.total_losses_meter.val, step)
                if Cfg.Loss.criterion_xent:
                    self.writer.add_scalar('loss/xent_loss', self.xent_losses_meter.val,
                                           step)
                if Cfg.Loss.criterion_htri:
                    self.writer.add_scalar('loss/htri_loss', self.htri_losses_meter.val,
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
                if Cfg.Loss.criterion_xent:
                    print_infos = print_infos + f", xent_loss: {self.xent_losses_meter.val:>.4f}" \
                                                f"({self.xent_losses_meter.avg:>.4f})"
                if Cfg.Loss.criterion_htri:
                    print_infos = print_infos + f", tri_loss: {self.htri_losses_meter.val:>.4f}" \
                                                f"({self.htri_losses_meter.avg:>.4f})"
                print(print_infos)
            if step % 100 == 0:
                # Window
                # self.writer.add_image('Image', images, step + self.epoch * train_data_batch)
                # Linux
                # self.writer.add_image('Image', image, step + self.epoch * train_data_batch, dataformats='NCHW')
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(name, param.clone().cpu().data.numpy(),
                                              step + self.epoch * train_data_batch)
                self.writer.add_scalar('loss/loss', self.total_losses_meter.val, step + self.epoch * train_data_batch)
                if Cfg.Loss.criterion_xent:
                    self.writer.add_scalar('loss/xent_loss', self.xent_losses_meter.val,
                                           step + self.epoch * train_data_batch)
                if Cfg.Loss.criterion_htri:
                    self.writer.add_scalar('loss/htri_loss', self.htri_losses_meter.val,
                                           step + self.epoch * train_data_batch)
                self.writer.add_scalar('acc/acc_top1', self.accuracy_top_1.val, step + self.epoch * train_data_batch)
                self.writer.add_scalar('acc/acc_top5', self.accuracy_top_5.val, step + self.epoch * train_data_batch)
        print("Total train loss:", self.total_losses_meter.avg)

    def save_model(self):
        if self.total_losses_meter.avg < self.lowest_train_loss or self.total_losses_meter.avg < 2.0:
            state = {
                'net': self.model.module.state_dict() if self.use_cuda else self.model.state_dict(),
                'val_acc': self.total_losses_meter.avg,
                'epoch': self.epoch + 1,
                'optimizer': self.optimizer.state_dict(),
            }
            if Cfg.Loss.criterion_cent:
                state.update({'centers': self.criterion_cent.state_dict()})
                state.update({'optimizer_center': self.optimizer_center.state_dict()})
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            save_path = './checkpoints/{}_{}_{:.04f}.pth'.format(self.model_name, self.epoch,
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
        if not self.model.training:
            features = self.model(images)
            return features
        features, outputs = self.model(images)
        total_loss = 0
        if Cfg.Loss.criterion_xent:
            xent_loss = self.criterion_xent(outputs, targets)
            # xent_loss = torch.mean(
            #     torch.cat([self.criterion_xent(output, targets).view(1) for output in outputs]))
            total_loss += xent_loss
        if Cfg.Loss.criterion_htri:
            tri_loss = self.criterion_htri(features, targets)
            total_loss += tri_loss
        accuracy_top_1, accuracy_top_5 = accuracy(outputs, targets, (1, 5))
        # accuracy_top_1, accuracy_top_5 = accuracy(outputs[0], targets, (1, 5))
        # print(self.model.module.features.conv1.weight)
        if backward:
            self.optimizer.zero_grad()
            if Cfg.Loss.criterion_cent:
                self.optimizer_center.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if Cfg.Loss.criterion_cent:
                # by doing so, self.weight_cent would not impact on the learning of centers
                for param in self.criterion_cent.parameters():
                    param.grad.data *= (1. / self.weight_cent)
                self.optimizer_center.step()
        if torch.__version__ < "0.4.0":
            if Cfg.Loss.criterion_xent:
                xent_loss_value = xent_loss.data.cpu().numpy()[0]
            if Cfg.Loss.criterion_htri:
                tri_loss_value = tri_loss.data.cpu().numpy()[0]
            if Cfg.Loss.criterion_cent:
                cent_loss_value = cent_loss.data.cpu().numpy()[0]
            total_loss_value = total_loss.data.cpu().numpy()[0]
            accuracy_top_1_value = accuracy_top_1.data.cpu().numpy()[0]
            accuracy_top_5_value = accuracy_top_5.data.cpu().numpy()[0]
        else:
            if Cfg.Loss.criterion_xent:
                xent_loss_value = xent_loss.item()
            if Cfg.Loss.criterion_htri:
                tri_loss_value = tri_loss.item()
            if Cfg.Loss.criterion_cent:
                cent_loss_value = cent_loss.item()
            total_loss_value = total_loss.item()
            accuracy_top_1_value = accuracy_top_1.item()
            accuracy_top_5_value = accuracy_top_5.item()
        if Cfg.Loss.criterion_xent:
            self.xent_losses_meter.update(xent_loss_value, targets.size(0))
        if Cfg.Loss.criterion_htri:
            self.htri_losses_meter.update(tri_loss_value, targets.size(0))
        if Cfg.Loss.criterion_cent:
            self.cent_losses_meter.update(cent_loss_value, targets.size(0))
        self.total_losses_meter.update(total_loss_value, targets.size(0))
        self.accuracy_top_1.update(accuracy_top_1_value, targets.size(0))
        self.accuracy_top_5.update(accuracy_top_5_value, targets.size(0))
        return outputs


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true", default=Cfg.Train.train)
    parser.add_argument("--search_learn_rate", dest="search_learn_rate", action="store_true",
                        default=Cfg.Train.search_learn_rate)
    parser.add_argument('-n', '--model_name', type=str, default=Cfg.Net.name,
                        help='model name')
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
    val_dataset = init_database(Cfg.Database.name, Cfg.Database.val_image_root, Cfg.Database.val_pickle_folder,
                                phase="test")
    train_data_loader = data.DataLoader(dataset=train_dataset,
                                        batch_size=Cfg.Sample.batch_size,
                                        shuffle=False,
                                        num_workers=Cfg.Sample.num_workers,
                                        worker_init_fn=worker_init_fn,
                                        drop_last=True)
    criterion_xent = CrossEntropyLoss(num_classes=train_dataset.class_num(), use_gpu=use_cuda, label_smooth=True)
criterion_htri = TripletLoss(margin=args.margin)
    model = models.init_model(name=args.model_name, pretrained=args.pretrained, num_classes=train_dataset.class_num())
    if args.finetune:
        classfier_params = list(map(id, model.classifier.parameters()))
        # bnneck_params = list(map(id, model.bnneck.parameters()))
        # base_params = filter(lambda p: id(p) not in classfier_params and id(p) not in bnneck_params, model.parameters())
        base_params = filter(lambda p: id(p) not in classfier_params, model.parameters())
        optimizer = torch.optim.Adam([{'params': base_params,
                                       'initial_lr': 0.1 * Cfg.Train.initial_lr,
                                       'lr': 0.1 * Cfg.Train.lr},
                                      # {'params': model.bnneck.parameters(),
                                      #  'initial_lr': Cfg.Train.initial_lr},
                                      {'params': model.classifier.parameters(),
                                       'initial_lr': Cfg.Train.initial_lr}],
                                     lr=Cfg.Train.lr,
                                     amsgrad=True,
                                     weight_decay=Cfg.Train.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': Cfg.Train.initial_lr}],
                                     lr=Cfg.Train.lr,
                                     amsgrad=True,
                                     weight_decay=Cfg.Train.weight_decay)
        # optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': Cfg.Train.initial_lr}],
        #                             lr=Cfg.Train.lr, momentum=0.9, weight_decay=0.0005)
    print(train_dataset.__len__(), train_dataset.class_num())
    fine_tuner = NetTrain(model, train_data_loader, val_dataset,
                          criterion_xent=criterion_xent, criterion_htri=criterion_htri, 
                          optimizer=optimizer,
                          model_name=args.model_name, lowest_train_loss=10)
    fine_tuner.max_epoch = Cfg.Train.epochs
    if args.resume:
        print("continue to train...")
        fine_tuner.load_checkpoint(args.checkpoint, finetune=args.finetune)
        # fine_tuner.load_checkpoint(args.checkpoint, finetune=False)
        fine_tuner.epoch = 1
    if args.train:
        fine_tuner.train(epoches=Cfg.Train.epochs, finetune=args.finetune)
    if args.export:
        fine_tuner.export_net(args.checkpoint, args.checkpoint + '.onnx')
    if args.test:
        fine_tuner.eval()
    if args.search_learn_rate:
        fine_tuner.search_learn_rate()
