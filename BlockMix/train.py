# coding=utf-8
import argparse
import torch
import os
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from Datasets import train_dataset, test_dataset
from Datasets.miniImageNet import miniImageNet
from Datasets.tiered_imagenet import Tiered_ImageNet
from Datasets.CUB import CUB_200_2011
from torchvision import transforms
from Models import ResNet12,ResNet18
from tqdm import tqdm
from utils import Timer,check_dir,rand_bbox
import torchvision.utils as vutils

parser = argparse.ArgumentParser(description='Train image model with 2 cross entropy loss')
parser.add_argument('--method', default='PN_global_loss_Jigsaw_aug', type=str)
parser.add_argument('--network', type=str, default='ResNet12',
                    help='choose which embedding network to use')
parser.add_argument('--weight1', type=float, default=0.5,
                    help='the weight of local loss')
parser.add_argument('--weight2', type=float, default=1.0,
                    help='the weight of global loss')
parser.add_argument('--max_replace_block_num', default=9, type=int)
# ************************************************************
parser.add_argument('--gpu_devices', default='3', type=str)
parser.add_argument('--dataset', type=str, default='tiered_imagenet', help='miniImageNet,CUB_200_2011, tiered_imagenet, Caltech256')
# ************************************************************

parser.add_argument('--lr', default=0.05, type=float,
                    help="initial learning rate")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--max-epoch', default=100, type=int,
                    help="maximum epochs to run")

parser.add_argument('--train-batch', default=1, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int,
                    help="test batch size")

parser.add_argument('--num_classes', type=int, default=64)
parser.add_argument('--save_dir', type=str, default='./result')

parser.add_argument('--nKnovel', type=int, default=5,
                    help='number of novel categories')
parser.add_argument('--nExemplars', type=int, default=5,
                    help='number of training(support) examples per novel category.')
parser.add_argument('--test_nExemplars', type=int, default=1,
                    help='number of testing(support) examples per novel category.')
# 6 and 15 query samples are used for training and 1shot.py respectively
parser.add_argument('--train_nTestNovel', type=int, default=5 * 5,
                    help='number of test(query) examples for all the novel category when training')
parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                    help='number of test examples for all the novel category')

parser.add_argument('--train_epoch_size', type=int, default=1000,
                    help='number of batches per epoch when training')
parser.add_argument('--epoch_size', type=int, default=600,
                    help='number of batches(tasks) per epoch')

parser.add_argument('--seed', type=int, default=1)

params = parser.parse_args()
torch.manual_seed(params.seed)
# 指定GPU运行代码
os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu_devices
print("Currently using GPU {}".format(params.gpu_devices))
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(params.seed)

save_path = os.path.join(params.save_dir, params.dataset, params.method, params.network, str(params.max_replace_block_num),str(params.weight2),str(params.weight1))
check_dir(save_path)

print('Initializing Dataset and Dataloder')
transform_train = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    transforms.RandomCrop(224, padding=8),
    transforms.RandomHorizontalFlip(),
    lambda x: np.asarray(x),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.RandomErasing(0.5)
            ])
transform_test = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
if params.dataset in 'miniImageNet':
    Dataset = miniImageNet()
    params.num_classes = 64
elif params.dataset in 'tiered_imagenet':
    Dataset = Tiered_ImageNet()
    params.num_classes = 351
elif params.dataset in 'CUB_200_2011':
    params.num_classes = 200
    Dataset = CUB_200_2011()

Dataset_train = train_dataset.FewShotDataset_train(
                 dataset=Dataset.train, # dataset of [(img_path, cats), ...].
                 labels2inds=Dataset.train_labels2inds, # labels of index {(cats: index1, index2, ...)}.
                 labelIds= Dataset.train_labelIds, # train labels [0, 1, 2, 3, ...,].
                 nKnovel=params.nKnovel, # number of novel categories.
                 nExemplars=params.nExemplars, # number of training examples per novel category.
                 nTestNovel=params.train_nTestNovel, # number of test examples for all the novel categories.
                 epoch_size=params.train_epoch_size, # number of tasks per eooch.
                 transform = transform_train
                 )
Dataloder_train = DataLoader(dataset=Dataset_train,
                             batch_size=params.train_batch,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=True)

Dataset_val = test_dataset.FewShotDataset_test(
                dataset=Dataset.test,
                labels2inds=Dataset.test_labels2inds,
                labelIds=Dataset.test_labelIds,
                nKnovel=params.nKnovel,
                nExemplars=params.test_nExemplars,
                nTestNovel=params.nTestNovel,
                epoch_size=params.epoch_size,
                transform=transform_test,
)
Dataloder_val = DataLoader(dataset=Dataset_val,
                             batch_size=params.test_batch,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=False)

print('Initializing Model and Optimizer')

if params.network in ['ResNet12']:
    model = ResNet12.Net(num_classes=params.num_classes).cuda()
    model_output_dim=1024
elif params.network in ['ResNet18']:
    model = ResNet18.Net(num_classes=params.num_classes).cuda()
    model_output_dim = 512

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD([{'params': model.parameters()}],
        lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay, nesterov=True)

# lambda_epoch = lambda e: 1.0 if e < 20 else (0.1 if e < 40 else (0.06 if e < 60 else (0.012 if e < 70 else (0.0024))))
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80,90,100], gamma=0.4)

print('Start Training')
timer = Timer()
max_val_acc = 0.0
max_val_epoch = 0.0
for epoch in range(1, params.max_epoch+1):

    lr = optimizer.param_groups[0]['lr']
    print('\nEpoch: [%d | %d] LR: %f' % (epoch, params.max_epoch, lr))
    model.train()
    avg_loss = 0

    iteration = len(Dataloder_train)
    for batch_idx, (images_train, labels_train, pids_train, images_test, labels_test, pids_test) in enumerate(tqdm(Dataloder_train),1):

        # images_train:[batch_size,25,3,84,84] images_train_aug:[batch_size,25,2,3,84,84]
        images_train = images_train.cuda().squeeze(0)#, labels_train.cuda()
        images_test, labels_test = images_test.cuda().squeeze(0), labels_test.cuda().squeeze(0)  # images_test:[batch_size,30,3,84,84]
        pids_train = pids_train.cuda().squeeze(0) # [batch_size,30]
        pids_test = pids_test.cuda().squeeze(0)

        weight = torch.zeros((params.nKnovel, model_output_dim), requires_grad=True).cuda()
        aug_images = images_test.clone()
        # L2归一化后的特征

        support_feature, support_score= model(images_train.view(-1,3,224,224)) #[25,1024]
        query_feature,query_score = model(images_test.view(-1,3,224,224)) #[30,1024]


        for i in range(params.nKnovel):
            weight_point = torch.zeros(params.nExemplars, model_output_dim)
            for j in range(params.nExemplars):
                features = support_feature[i*params.nExemplars+j].unsqueeze(0)# [1,1024]
                weight_point[j:(j+1)] = features
            weight[i] = torch.mean(weight_point, 0)
        weight = model.l2_norm(weight) # [5,1024]

        predict = torch.matmul(query_feature, torch.transpose(weight, 0, 1)) * model.s
        loss1 = criterion(predict, labels_test)


        # lam = np.random.beta(1, 1)
        rand_index = torch.randperm(aug_images.size()[0]).cuda()
        target_a = pids_test
        target_b = pids_train[rand_index]

        if params.max_replace_block_num == 0:
            replace_block_num = 0
            replaced_indexs = []
        else:
            replace_block_num = np.random.randint(1, params.max_replace_block_num+1) # 从[1,max]中随机选择替换block的数量
            replaced_indexs = np.random.choice(9, replace_block_num, replace=False)

        # BlockMix
        patch_xl = np.array([0, 0, 0, 74, 74, 74, 148, 148, 148])
        patch_xr = np.array([74, 74, 74, 148, 148, 148, 224, 224, 224])
        patch_yl = np.array([0, 74, 148, 0, 74, 148, 0, 74, 148])
        patch_yr = np.array([74, 148, 224, 74, 148, 224, 74, 148, 224])
        for l in range(replace_block_num):
            replaced_index = replaced_indexs[l]
            aug_images[:, :, patch_xl[replaced_index]:patch_xr[replaced_index],
            patch_yl[replaced_index]:patch_yr[replaced_index]] = images_train[rand_index, :,
                                                                 patch_xl[replaced_index]:patch_xr[replaced_index],
                                                                 patch_yl[replaced_index]:patch_yr[replaced_index]]
        lam = 1 - float( replace_block_num/ 9 )
        # compute output
        _,output = model(aug_images.cuda())
        loss2 = lam*criterion(output, target_a)+(1-lam)*criterion(output,target_b)

        loss = loss1*params.weight1 + loss2*params.weight2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.weight_norm()
        avg_loss = avg_loss + loss.item()
    print('Epoch {:d} | Batch {:d} | Loss {:f}'.format(
            epoch, iteration, avg_loss / float(iteration)))

    lr_scheduler.step()

    if epoch == 1 or  epoch > 60:
        with torch.no_grad():
            test_accuracies = []
            model.eval()
            for batch_idx , (images_train, labels_train, images_test, labels_test) in enumerate(tqdm(Dataloder_val),1):

                images_train = images_train.cuda().squeeze(0) # [25,3,224,224]
                images_test, labels_test = images_test.cuda().squeeze(0), labels_test.cuda().squeeze(0) # [75,3,224,224] [75]

                weight = torch.zeros((5, model_output_dim), requires_grad=True).cuda()

                support_feature, _ = model(images_train.view(-1, 3, 224, 224))  # [25,1024]
                query_feature, _ = model(images_test.view(-1, 3, 224, 224))  # [75,1024]

                for i in range(5):
                    weight_point = torch.zeros(params.test_nExemplars, model_output_dim)
                    for j in range(params.test_nExemplars):
                        features = support_feature[i*params.test_nExemplars+j].unsqueeze(0)
                        weight_point[j:(j+1)]=features
                    weight[i] = torch.mean(weight_point, 0)

                weight = model.l2_norm(weight)

                predict = torch.matmul(query_feature, torch.transpose(weight, 0, 1)) * model.s # [75,5]
                acc = (predict.topk(1)[1].view(-1) == labels_test).float().sum(0) / labels_test.shape[0] * 100.
                test_accuracies.append(acc.item())

            val_acc_novel = np.mean(np.array(test_accuracies))
            val_acc_novel_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(params.epoch_size)

            if val_acc_novel > max_val_acc:
                max_val_acc = val_acc_novel
                max_val_epoch = epoch
                torch.save({'Model': model.state_dict()}, os.path.join(save_path, 'best_model.pth'))
                print('Best model saving!!!')
                print(
                    'Val_Epoch: [{}/{}]\tAccuracyNovel: {:.2f} +- {:.2f} %'.format(
                        epoch, params.max_epoch,
                        val_acc_novel, val_acc_novel_ci95))
            else:
                print(
                    'Val_Epoch: [{}/{}]\tAccuracyNovel: {:.2f} +- {:.2f} %'.format(
                        epoch, params.max_epoch,
                        val_acc_novel, val_acc_novel_ci95))

            torch.save({'Model': model.state_dict()}, os.path.join(save_path, 'last_model.pth'))
    print('Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(params.max_epoch))))