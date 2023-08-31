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
# from 1shot.py import block_mix_inference,block_mix_sorted_inference, block_mix_all_inference,block_mix_5shot_sorted_inference
# from 1shot.py import block_mix_inference,block_mix_sorted_inference, block_mix_all_inference, inference_wo_aug,inference_wo_aug_att
from inference import   inference_wo_aug, block_mix_all_inference, block_mix_sorted_inference, block_mix_inference,inference_wo_aug_att
from utils import Timer,check_dir
# from cutmix_inference import cut_mix_inference
# from mixup_inference import mix_up_inference
import torchvision.utils as vutils

parser = argparse.ArgumentParser(description='Test image model with cross entropy loss')
parser.add_argument('--method', default='PN_global_loss_Jigsaw_mix', type=str)
parser.add_argument('--network', type=str, default='ResNet18',help='choose which embedding network to use')
parser.add_argument('--weight1', type=float, default=0.5,
                    help='the weight of local loss')
parser.add_argument('--weight2', type=float, default=0.5,
                    help='the weight of global loss')
parser.add_argument('--max_replace_block_num', default=8, type=int)
parser.add_argument('--num_inference', default=3, type=int)
# ************************************************************
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('-d', '--dataset', type=str, default='tiered_imagenet', help='miniImageNet')
# ************************************************************

parser.add_argument('--test-batch', default=1, type=int,
                    help="test batch size")
parser.add_argument('--num_classes', type=int, default=64)
parser.add_argument('--save_dir', type=str, default='./result')

parser.add_argument('--nKnovel', type=int, default=5,
                    help='number of novel categories')
parser.add_argument('--nExemplars', type=int, default=5,
                    help='number of training(support) examples per novel category.')
# 6 and 15 query samples are used for training and 1shot.py respectively

parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                    help='number of test examples for all the novel category')

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
# check_dir(save_path)

print('Initializing Dataset and Dataloder')

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

Dataset_test_1shot = test_dataset.FewShotDataset_test(
                 dataset=Dataset.test, # dataset of [(img_path, cats), ...].
                 labels2inds=Dataset.test_labels2inds, # labels of index {(cats: index1, index2, ...)}.
                 labelIds= Dataset.test_labelIds, # train labels [0, 1, 2, 3, ...,].
                 nKnovel=params.nKnovel, # number of novel categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=params.nTestNovel, # number of test examples for all the novel categories.
                 epoch_size=params.epoch_size, # number of tasks per eooch.
                 transform = transform_test
                 )
Dataloder_test_1shot = DataLoader(dataset=Dataset_test_1shot,
                             batch_size=params.test_batch,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=False)

Dataset_test_5shot = test_dataset.FewShotDataset_test(
                dataset=Dataset.test,
                labels2inds=Dataset.test_labels2inds,
                labelIds=Dataset.test_labelIds,
                nKnovel=params.nKnovel,
                nExemplars=5,
                nTestNovel=params.nTestNovel,
                epoch_size=params.epoch_size,
                transform=transform_test,
)
Dataloder_test_5shot = DataLoader(dataset=Dataset_test_5shot,
                             batch_size=params.test_batch,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=False)

print('Initializing Model and Optimizer')

if params.network in ['ResNet12']:
    model = ResNet12.Net(num_classes=params.num_classes).cuda()
    model_output_dim = 1024
elif params.network in ['ResNet18']:
    model = ResNet18.Net(num_classes=params.num_classes).cuda()
    model_output_dim = 512

checkpoint = torch.load(os.path.join(save_path,'best_model.pth'))
print(os.path.join(save_path,'best_model.pth'))
model.load_state_dict(checkpoint['Model'])

print('Start Testing')

for dataloder in [Dataloder_test_1shot,Dataloder_test_5shot]:
# for dataloder in [Dataloder_test_5shot]:
    if dataloder in [Dataloder_test_1shot]:
        params.nExemplars=1
    else:
        params.nExemplars = 5

    with torch.no_grad():
        test_accuracies = []
        model.eval()

        for batch_idx , (images_train, labels_train, images_test, labels_test) in enumerate(tqdm(dataloder),1):

            images_train = images_train.cuda().squeeze(0) # [25,3,224,224]
            images_test, labels_test = images_test.cuda().squeeze(0), labels_test.cuda().squeeze(0) # [75,3,224,224] [75]

            weight = torch.zeros((params.nKnovel, model_output_dim), requires_grad=True).cuda()

            support_feature, _ = model(images_train.view(-1, 3, 224, 224))  # [25,1024]
            query_feature, _ = model(images_test.view(-1, 3, 224, 224))  # [75,1024]

            for i in range(params.nKnovel):
                weight_point = torch.zeros(params.nExemplars, model_output_dim)
                for j in range(params.nExemplars):
                    features = support_feature[i*params.nExemplars+j].unsqueeze(0)
                    weight_point[j:(j+1)]=features
                weight[i] = torch.mean(weight_point, 0)
            weight = model.l2_norm(weight)

            predict = torch.matmul(query_feature, torch.transpose(weight, 0, 1)) * model.s # [75,5]

            num = 0

            while num < params.num_inference:
                weight = block_mix_inference(predict,images_train,images_test,model,weight,params)
                predict = torch.matmul(query_feature, torch.transpose(weight, 0, 1)) * model.s
                num += 1


            acc = (predict.topk(1)[1].view(-1) == labels_test).float().sum(0) / labels_test.shape[0] * 100.
            test_accuracies.append(acc.item())

        val_acc_novel = np.mean(np.array(test_accuracies))
        val_acc_novel_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(params.epoch_size)


        print('AccuracyNovel: {:.2f} +- {:.2f} %'.format(val_acc_novel, val_acc_novel_ci95))


