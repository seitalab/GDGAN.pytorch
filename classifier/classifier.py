import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import os
from datetime import datetime
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from classifier import networks

from logger import Logger

import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

global_iter = 0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='chest-xray',
                        choices=['chest-xray-with-synth',
                                 'chest-xray-with-synth-acgan',
                                 'over-chest-xray',
                                 'under-chest-xray',
                                 'chest-xray'],
                        help='name of dataset')
    parser.add_argument('--root_path', type=str,
                        default="./data/nih-chest-xrays/images")
    parser.add_argument('--label_path', type=str,
                        default="./data/nih-chest-xrays/"
                                "labels/train_val_list2.txt")
    parser.add_argument('--test_label_path', type=str,
                        default="./data/nih-chest-xrays/"
                                "labels/test_list2.txt")
    parser.add_argument('--model', type=str, default='vgg11',
                        choices=['vgg11',
                                 'vgg11_bn',
                                 'vgg13',
                                 'vgg13_bn',
                                 'vgg16',
                                 'vgg16_bn',
                                 'vgg19',
                                 'vgg19_bn',
                                 'resnet18',
                                 'resnet34',
                                 'resnet50',
                                 'resnet101',
                                 'resnet152',
                                 'densenet121',
                                 'densenet169',
                                 'densenet201'],
                        help='name of model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--synth_root', type=str, default="",
                        help='root of the synthetic images')
    parser.add_argument('--image_size', type=int, default=64,
                        help='image size')
    return parser.parse_args()


def train(model, criterion, optimizer, dataloader, class_indices, logger):
    model.train()
    for i, (x, y) in enumerate(dataloader):
        global global_iter
        global_iter += 1
        x, y = x.to(device), y[:, class_indices].to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        logger.add_scalar('data/train_loss',
                          loss.item(),
                          global_iter)


def valid(model, criterion, dataloader, class_indices, epoch, logger):
    roc_scores, ys, cs, loss = test_base(model,
                                         criterion,
                                         dataloader,
                                         class_indices)
    logger.add_scalar('data/valid_loss', loss, epoch+1)
    scalars_dic = {}
    for i, ind in enumerate(class_indices):
        scalars_dic['label%d' % ind] = roc_scores[i]
        logger.add_pr_curve('data/valid_pr/label%d' % ind,
                            ys[:, i],
                            cs[:, i],
                            epoch+1)
    logger.add_scalars('data/valid_roc', scalars_dic, epoch+1)
    return loss


def test(model, criterion, dataloader, class_indices, epoch, logger):
    roc_scores, ys, cs, loss = test_base(model,
                                         criterion,
                                         dataloader,
                                         class_indices)
    logger.add_scalar('data/test_loss', loss, epoch+1)
    scalars_dic = {}
    for i, ind in enumerate(class_indices):
        scalars_dic['label%d' % ind] = roc_scores[i]
        logger.add_pr_curve('data/test_pr/label%d' % ind,
                            ys[:, i],
                            cs[:, i],
                            epoch+1)
    logger.add_scalars('data/test_roc', scalars_dic, epoch+1)


def test_base(model, criterion, dataloader, class_indices):
    model.eval()
    if dataloader.sampler is not None:
        size = (len(dataloader.sampler), len(class_indices))
    else:
        size = (len(dataloader.dataset), len(class_indices))
    ys = np.empty(size)
    cs = np.empty(size)
    start, end = 0, 0
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            start = end
            end = start + x.size(0)
            x = x.to(device)
            y_tensor = y[:, class_indices].to(device)
            y = y[:, class_indices].numpy()
            out = model(x)
            loss = criterion(out, y_tensor)
            total_loss += loss.item()
            n += 1
            out = F.sigmoid(out)
            out = out.detach().cpu().numpy()

            ys[start:end] = y
            cs[start:end] = out
    roc_scores = roc_auc_score(ys, cs, average=None)

    return roc_scores, ys, cs, total_loss / n


def save_model(model, epoch, save_dir):
    torch.save(model.state_dict(),
               os.path.join(save_dir, 'epoch%03d.pkl' % (epoch+1)))


def main():
    args = parse_args()
    args.date_str = datetime.now().strftime('%Y_%m_%d_%H_%M')
    summary_dir = os.path.join('classifier', 'summary', args.dataset)
    result_dir = os.path.join('classifier',
                              'result',
                              args.dataset,
                              args.date_str)
    model_save_dir = os.path.join(result_dir, 'models')
    logger = Logger(args, summary_dir, result_dir)

    if 'chest-xray' in args.dataset:
        class_indices = list(range(14))
        class_num = len(class_indices)

        train_transform = \
            transforms.Compose([transforms.Resize(args.image_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
        valid_test_transform = \
            transforms.Compose([transforms.Resize(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
        train_dataset = \
            datasets.ChestXrayDataset(root=args.root_path,
                                      image_list_file=args.label_path,
                                      train=True,
                                      transform=train_transform,
                                      synth_root=args.synth_root,
                                      oversample=('over' in args.dataset),
                                      undersample=('under' in args.dataset),
                                      class_indices=class_indices)
        valid_dataset = \
            datasets.ChestXrayDataset(root=args.root_path,
                                      image_list_file=args.label_path,
                                      train=True,
                                      transform=valid_test_transform)
        dataset_indices = list(range(len(train_dataset)))
        train_sampler = \
            (torch.utils.data.sampler
             .SubsetRandomSampler(dataset_indices[:-train_dataset.val_num]))
        valid_sampler = \
            (torch.utils.data.sampler
             .SubsetRandomSampler(dataset_indices[-valid_dataset.val_num:]))
        train_dataloader = \
            torch.utils.data.DataLoader(train_dataset,
                                        sampler=train_sampler,
                                        batch_size=args.batch_size,
                                        num_workers=4)
        valid_dataloader = \
            torch.utils.data.DataLoader(valid_dataset,
                                        sampler=valid_sampler,
                                        batch_size=args.batch_size,
                                        num_workers=4)

        test_dataset = \
            datasets.ChestXrayDataset(root=args.root_path,
                                      image_list_file=args.test_label_path,
                                      train=False,
                                      transform=valid_test_transform)
        test_dataloader = \
            torch.utils.data.DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=4)

    # model loading
    if args.model == 'vgg11':
        model = networks.vgg.VGG11(num_classes=class_num)
    elif args.model == 'vgg11_bn':
        model = networks.vgg.VGG11_bn(num_classes=class_num)
    elif args.model == 'vgg13':
        model = networks.vgg.VGG13(num_classes=class_num)
    elif args.model == 'vgg13_bn':
        model = networks.vgg.VGG13_bn(num_classes=class_num)
    elif args.model == 'vgg16':
        model = networks.vgg.VGG16(num_classes=class_num)
    elif args.model == 'vgg16_bn':
        model = networks.vgg.VGG16_bn(num_classes=class_num)
    elif args.model == 'vgg19':
        model = networks.vgg.VGG19(num_classes=class_num)
    elif args.model == 'vgg19_bn':
        model = networks.vgg.VGG19_bn(num_classes=class_num)
    elif args.model == 'resnet18':
        model = networks.resnet.ResNet18(num_classes=class_num)
    elif args.model == 'resnet34':
        model = networks.resnet.ResNet34(num_classes=class_num)
    elif args.model == 'resnet50':
        model = networks.resnet.ResNet50(num_classes=class_num)
    elif args.model == 'resnet101':
        model = networks.resnet.ResNet101(num_classes=class_num)
    elif args.model == 'resnet152':
        model = networks.resnet.ResNet152(num_classes=class_num)
    elif args.model == 'densenet121':
        model = networks.DenseNet121(num_classes=class_num)
    elif args.model == 'densenet169':
        model = networks.DenseNet169(num_classes=class_num)
    elif args.model == 'densenet201':
        model = networks.DenseNet201(num_classes=class_num)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           betas=(args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     patience=1)

    best_loss = 10000000000
    last_updated_epoch = 0
    patience = 20
    for epoch in range(args.epochs):
        print('epoch:', epoch + 1)
        print('train')
        train(model,
              criterion,
              optimizer,
              train_dataloader,
              class_indices,
              logger)

        print('valid')
        valid_loss = valid(model,
                           criterion,
                           valid_dataloader,
                           class_indices,
                           epoch,
                           logger)

        scheduler.step(valid_loss)

        print('test')
        test(model, criterion, test_dataloader, class_indices, epoch, logger)

        if valid_loss < best_loss:
            print("best loss updated")
            save_model(model, epoch, model_save_dir)
            best_loss = valid_loss
            last_updated_epoch = epoch

        if last_updated_epoch + patience <= epoch:
            print("early stopping came into effect, finishing train...")
            break

    print('finish')
    logger.save_history()


if __name__ == '__main__':
    main()
