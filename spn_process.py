import utils
from SPN.demo import run
from center.centerloss import CenterLoss
from cluster_code.pairwise import PairEnum
from cluster_code import criterion
from cluster_code.demo import prepare_task_target
from contrastive.component import Projector
from prototype_code.core import NShotTaskSampler
from tllib.utils.analysis import collect_feature, tsne, a_distance
from tllib.utils.logger import CompleteLogger
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.metric import accuracy
from tllib.utils.data import ForeverDataIterator
from tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier
from tllib.modules.domain_discriminator import DomainDiscriminator

import os
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp

import timm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F

import models
from SPN.metric import Confusion
from SPN.similarity import Learner_DensePairSimilarity

sys.path.append('../../..')


sys.path.append('.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SPN_Net(nn.Module):
    def __init__(self, backbone, backbone_out_feature=512):
        super(SPN_Net, self).__init__()
        self.backbone = backbone.features
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_out_feature, backbone_out_feature),
            nn.BatchNorm1d(backbone_out_feature),
            nn.ReLU()
        )
        n_feat = backbone_out_feature

        self.last = nn.Sequential(
            nn.Linear(n_feat * 2, n_feat * 4),
            nn.BatchNorm1d(n_feat * 4),
            # inplace = True 
            nn.ReLU(inplace=True),
            nn.Linear(n_feat * 4, 2)
        )

    def forward(self, x):  # [8, 3, 224, 224]
        x = self.backbone(x)  # [8, 512, 7, 7]
        x = self.bottleneck(x)  # [8, 512]
        feat1, feat2 = PairEnum(x)
        featcat = torch.cat([feat1, feat2], 1)
        out = self.last(featcat)  # [64, 2]
        return out


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.data == 'Office31':
        domain_list = ['A', 'W', 'D']
        domain_list.remove(args.target[0])
        domain_list.remove(args.source[0])

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(
        args.train_resizing,
        random_horizontal_flip=not args.no_hflip,
        random_color_jitter=True,  
        resize_size=args.resize_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std)
    val_transform = utils.get_val_transform(
        args.val_resizing,
        resize_size=args.resize_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)

    train_source_dataset_list = []
    train_source_dataset_list.append(train_source_dataset)
    for domain in domain_list:
        train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
            utils.get_dataset(args.data, args.root, domain, args.target, train_transform, val_transform)
        train_source_dataset_list.append(train_source_dataset)

    train_dataset = ConcatDataset(train_source_dataset_list)

    # train_source_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True)
    # train_target_loader
    eval_loader = DataLoader(
        train_target_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True)

    # create model
    print("=> using model '{}'".format(args.arch))
    # args.scratch
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    spn = SPN_Net(backbone=backbone, backbone_out_feature=512)

    criterion = nn.CrossEntropyLoss()

    # define optimizer and lr scheduler
    optim_args = {'lr': args.lr, 'momentum': 0.9}
    optimizer = torch.optim.SGD(spn.parameters(), **optim_args)
    lr_scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    # to device
    spn = spn.to(device)
    criterion = criterion.to(device)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(
            logger.get_checkpoint_path('best'),
            map_location='cpu')
        spn.load_state_dict(checkpoint)

    if args.phase == 'test':
        acc1 = evaluate(eval_loader, spn, args)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(
            train_loader,
            spn,
            criterion,
            optimizer,
            lr_scheduler,
            epoch,
            args)

        # evaluate on validation set
        acc1 = evaluate(eval_loader, spn, args)

        # remember best acc@1 and save checkpoint
        torch.save(
            spn.state_dict(),
            logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(
                logger.get_checkpoint_path('latest'),
                logger.get_checkpoint_path('best'))

        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    spn.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = evaluate(eval_loader, spn, args)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def evaluate(eval_loader, model, args):
    # Initialize all meters
    confusion = Confusion(args.out_dim)

    print('---- Evaluation ----')
    model.eval()
    for i, (input, target) in enumerate(eval_loader):

        # Prepare the inputs
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
        _, eval_target = prepare_task_target(input, target, args)

        # Inference
        output = model(input)

        # Update the performance meter
        output = output.detach()
        confusion.add(output, eval_target)

    # Loss-specific information
    KPI = 0
    confusion.show(
        width=15,
        row_labels=[
            'GT_dis-simi',
            'GT_simi'],
        column_labels=[
            'Pred_dis-simi',
            'Pred_simi'])
    KPI = confusion.f1score(1)
    # f1-score for similar pair (label:1)
    print('[Test] similar pair f1-score:', KPI)
    print('[Test] dissimilar pair f1-score:', confusion.f1score(0))
    return KPI


def train(
        train_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        epoch,
        args):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Sim Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()
        train_target, eval_target = prepare_task_target(input, target, args)

        # measure data loading time。。
        data_time.update(time.time() - end)

        # compute output
        output = model(input)

        loss = criterion(output, train_target)

        cls_acc = accuracy(output, train_target)[0]

        losses.update(loss.item(), input.size(0))
        cls_accs.update(cls_acc.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


