# coding=utf-8
import torch
import numpy as np
import torch.nn.functional as F
patch_xl = np.array([0,0,0,74,74,74,148,148,148])
patch_xr = np.array([74,74,74,148,148,148,224,224,224])
patch_yl = np.array([0,74,148,0,74,148,0,74,148])
patch_yr = np.array([74,148,224,74,148,224,74,148,224])


def jigsaw_aug(support_images, auxiliary_images, support_features, confidence_scores,model):

    weight_point = torch.zeros(support_images.size(0), 1024)

    for i in range(support_images.size(0)):
        aug_images = []
        confidence=confidence_scores

        if auxiliary_images.size(0) == 0:
            prototype = support_features[i]
        # elif auxiliary_images.size(0) == 1:
        else:
            for j in range(auxiliary_images.size(0)):
                replace_block_num = int(confidence[j]*9)
                replaced_indexs = np.random.choice(9, replace_block_num, replace=False)
                aug_image = support_images[i,:,:,:].clone()
                aux_img = auxiliary_images[j,:,:,:]
                for l in range(replace_block_num):
                    replaced_index = int(replaced_indexs[l])
                    aug_image[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]] = aux_img[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]]
                aug_images.append(aug_image.cuda())
            aug_images.append(support_images[i,:,:,:])
            confidence = torch.cat((confidence,torch.ones(1, 1).cuda()),0)

            aug_images=torch.stack(aug_images,dim=0)
            aug_features,_ = model(aug_images.view(-1,3,224,224))
            features = aug_features.mul(confidence)

            attention = torch.sum(confidence, dim=0)

            prototype=torch.div(torch.sum(features, dim=0), attention)


        weight_point[i] = prototype

    return weight_point

def jigsaw_aug_round(support_images, auxiliary_images, confidence_scores,model,params):
    if params.network in ['ResNet18']:
        weight_point = torch.zeros(support_images.size(0), 512)
    elif params.network in ['ResNet12']:
        weight_point = torch.zeros(support_images.size(0), 1024)

    for i in range(support_images.size(0)):
        aug_images = []
        confidence=confidence_scores

        if auxiliary_images.size(0) == 0:
            aug_features, _ = model(support_images[i].view(-1, 3, 224, 224))
            prototype = aug_features
        # elif auxiliary_images.size(0) == 1:
        else:
            for j in range(auxiliary_images.size(0)):
                replace_block_num = int(torch.round(confidence[j]*9).item())
                # print(replace_block_num)
                replaced_indexs = np.random.choice(9, replace_block_num, replace=False)
                aug_image = support_images[i,:,:,:].clone()

                aux_img = auxiliary_images[j,:,:,:]
                for l in range(replace_block_num):
                    replaced_index = int(replaced_indexs[l])
                    aug_image[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]] = aux_img[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]]
                aug_images.append(aug_image.cuda())
            aug_images.append(support_images[i, :, :, :])
            confidence = torch.cat((confidence,torch.ones(1, 1).cuda()),0)
            aug_images=torch.stack(aug_images,dim=0)

            aug_features,_ = model(aug_images.view(-1,3,224,224))
            features = aug_features.mul(confidence)
            attention = torch.sum(confidence, dim=0)
            prototype=torch.div(torch.sum(features, dim=0), attention)

        weight_point[i] = prototype

    return weight_point

# def jigsaw_aug_5shot_round(support_images, auxiliary_images, confidence_scores,model,params):
#     # if params.network in ['ResNet18']:
#     #     weight_point = torch.zeros(support_images.size(0), 512)
#     # elif params.network in ['ResNet12']:
#     #     weight_point = torch.zeros(support_images.size(0), 1024)
#
#     # for i in range(support_images.size(0)):
#     aug_images = []
#     #     confidence=confidence_scores
#
#     if auxiliary_images.size(0) == 0:
#         aug_features, _ = model(support_images.view(-1, 3, 224, 224))
#         prototype = aug_features
#         # elif auxiliary_images.size(0) == 1:
#     else:
#         for j in range(auxiliary_images.size(0)):
#             replace_block_num = int(torch.round(confidence_scores[j]*9).item())
#             # print(replace_block_num)
#             replaced_indexs = np.random.choice(9, replace_block_num, replace=False)
#
#             aug_image_index = np.random.choice(5)
#             aug_image = support_images[aug_image_index,:,:,:].clone()
#
#             aux_img = auxiliary_images[j,:,:,:]
#             for l in range(replace_block_num):
#                 replaced_index = int(replaced_indexs[l])
#                 aug_image[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]] = aux_img[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]]
#             aug_images.append(aug_image.cuda())
#         aug_images = torch.stack(aug_images, dim=0)
#         aug_images = torch.cat((aug_images,support_images),dim=0)
#         confidence = torch.cat((confidence_scores,torch.ones(5, 1).cuda()),0)
#         # import torchvision.utils as vutils
#         # vutils.save_image(auxiliary_images.view(-1, 3, 224, 224), './vis/query.png', normalize=True)
#         # vutils.save_image(aug_images.view(-1, 3, 224, 224), './vis/aug.png',normalize=True)
#         # exit(0)
#
#         aug_features,_ = model(aug_images.view(-1,3,224,224))
#
#         features = aug_features.mul(confidence)
#         attention = torch.sum(confidence, dim=0)
#         prototype=torch.div(torch.sum(features, dim=0), attention)
#
#     # weight_point[i] = prototype
#
#     return prototype



def block_mix_inference(predict,image_train,image_test,model,weight,params):
    confidence_scores = torch.nn.Softmax(dim=-1)(predict)
    topk_scores, topk_labels = confidence_scores.data.topk(1, 1, True, True)

    # sorted_topk_scores, sorted_idx = topk_scores.view(-1).sort(descending=True)
    # sorted_topk_scores = sorted_topk_scores.view(-1, 1)
    # sorted_topk_labels = topk_labels[sorted_idx].view(-1, 1)
    # sorted_images_test = images_test[sorted_idx]

    for i in range(params.nKnovel):
        query_idx = torch.nonzero(topk_labels.view(-1) == i).view(-1)
        query_aux_imgs = image_test[query_idx]  # [num,3,224,224]
        support_imgs = image_train[i * params.nExemplars:(i + 1) * params.nExemplars]  # [5,3,224,224]
        confidence_aux = topk_scores[query_idx]  # [num,1]
        weight_point = jigsaw_aug_round(support_imgs, query_aux_imgs, confidence_aux, model,params)
        weight[i] = torch.mean(weight_point, 0)
    weight = model.l2_norm(weight)

    return weight

def block_mix_sorted_inference(predict,image_train,image_test,model,weight,params):
    confidence_scores = torch.nn.Softmax(dim=-1)(predict)
    topk_scores, topk_labels = confidence_scores.data.topk(1, 1, True, True)

    sorted_topk_scores, sorted_idx = topk_scores.view(-1).sort(descending=True)
    sorted_topk_scores = sorted_topk_scores.view(-1, 1)
    sorted_topk_labels = topk_labels[sorted_idx].view(-1, 1)
    sorted_images_test = image_test[sorted_idx]

    for i in range(params.nKnovel):
        query_idx = torch.nonzero(sorted_topk_labels.view(-1) == i).view(-1)
        query_aux_imgs = sorted_images_test[query_idx]  # [num,3,224,224]
        support_imgs = image_train[i * params.nExemplars:(i + 1) * params.nExemplars]  # [5,3,224,224]
        confidence_aux = sorted_topk_scores[query_idx]  # [num,1]
        weight_point = jigsaw_aug_round(support_imgs, query_aux_imgs, confidence_aux, model,params)
        weight[i] = torch.mean(weight_point, 0)
    weight = model.l2_norm(weight)

    return weight

def jigsaw_aug_all_round(support_images, auxiliary_images, confidence_scores,model):

    weight_point = torch.zeros(support_images.size(0), 1024)

    for i in range(support_images.size(0)):
        aug_images = []
        confidences = []
        confidence=confidence_scores
        # print(confidence.shape)
        # exit(0)
        for j in range(auxiliary_images.size(0)):
            replace_block_num = int(torch.round(confidence[j]*9).item())

            if replace_block_num == 0:
                pass
            else:
                replaced_indexs = np.random.choice(9, replace_block_num, replace=False)

                aug_image = support_images[i,:,:,:].clone()
                aux_img = auxiliary_images[j,:,:,:]
                for l in range(replace_block_num):
                    replaced_index = int(replaced_indexs[l])
                    aug_image[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]] = aux_img[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]]
                aug_images.append(aug_image.cuda())
                confidences.append(confidence[j].view(-1,1))
        aug_images.append(support_images[i, :, :, :])
        # confidence = torch.cat((confidence.view(-1,1),torch.ones(1, 1).cuda()),0)
        confidences.append(torch.ones(1,1).cuda())
        aug_images=torch.stack(aug_images,dim=0)
        confidences = torch.cat(confidences,dim=0)

        aug_features,_ = model(aug_images.view(-1,3,224,224))
        features = aug_features.mul(confidences)
        attention = torch.sum(confidences, dim=0)
        prototype=torch.div(torch.sum(features, dim=0), attention)

        weight_point[i] = prototype

    return weight_point



def block_mix_all_inference(predict,image_train,image_test,model,weight,params):
    confidence_scores = torch.nn.Softmax(dim=-1)(predict)

    for i in range(params.nKnovel):
        support_imgs = image_train[i * params.nExemplars:(i + 1) * params.nExemplars]  # [5,3,224,224]
        weight_point = jigsaw_aug_all_round(support_imgs, image_test, confidence_scores[:,i], model)
        weight[i] = torch.mean(weight_point, 0)
    weight = model.l2_norm(weight)

    return weight

def inference_wo_aug(predict,support_feature,query_feature,model,weight,params):

    confidence_scores = torch.nn.Softmax(dim=-1)(predict)  # [75,5]
    topk_scores, topk_labels = confidence_scores.data.topk(1, 1, True, True)
    for i in range(params.nKnovel):
        query_idx = torch.nonzero(topk_labels.view(-1) == i).view(-1)
        support = support_feature[i * params.nExemplars:(i + 1) * params.nExemplars]  # [1,1024]
        query = query_feature[query_idx].mul(topk_scores[query_idx])
        #
        support_attention = torch.sum(torch.ones(support.size(0), 1).cuda(), dim=0)
        query_attention = torch.sum(topk_scores[query_idx], dim=0)
        attention = support_attention + query_attention
        weight[i] = torch.div(torch.sum(torch.cat((support, query), dim=0), dim=0), attention)
        # weight[i] = torch.sum(torch.cat((support, query), dim=0), dim=0)

    weight = model.l2_norm(weight)

    return weight





# def block_mix_5shot_sorted_inference(predict,image_train,image_test,model,weight,params,per):
#     confidence_scores = torch.nn.Softmax(dim=-1)(predict)
#     topk_scores, topk_labels = confidence_scores.data.topk(1, 1, True, True)
#
#     sorted_topk_scores, sorted_idx = topk_scores.view(-1).sort(descending=True)
#     sorted_topk_scores = sorted_topk_scores.view(-1, 1)
#     sorted_topk_labels = topk_labels[sorted_idx].view(-1, 1)
#     sorted_images_test = image_test[sorted_idx]
#
#     for i in range(params.nKnovel):
#         query_idx = torch.nonzero(sorted_topk_labels.view(-1) == i).view(-1)
#         num = round(len(query_idx)*per)
#         query_idx = query_idx[:num]
#         query_aux_imgs = sorted_images_test[query_idx]  # [num,3,224,224]
#         support_imgs = image_train[i * params.nExemplars:(i + 1) * params.nExemplars]  # [5,3,224,224]
#         confidence_aux = sorted_topk_scores[query_idx]  # [num,1]
#         weight_point = jigsaw_aug_5shot_round(support_imgs, query_aux_imgs, confidence_aux, model,params)
#         weight[i] = weight_point
#     weight = model.l2_norm(weight)
#
#     return weight


# def jigsaw_finetune_round(support_images, auxiliary_images, confidence_scores):
#
#
#     aug_images = []
#     for i in range(support_images.size(0)):
#         # aug_images = []
#         confidence=confidence_scores
#
#         if auxiliary_images.size(0) == 0:
#             aug_images.append(support_images[i, :, :, :])
#             # pass
#         # elif auxiliary_images.size(0) == 1:
#         else:
#             # print(auxiliary_images.size(0))
#             for j in range(auxiliary_images.size(0)):
#                 replace_block_num = int(torch.round(confidence[j]*9).item())
#                 replaced_indexs = np.random.choice(9, replace_block_num, replace=False)
#                 aug_image = support_images[i,:,:,:].clone()
#
#                 aux_img = auxiliary_images[j,:,:,:]
#                 for l in range(replace_block_num):
#                     replaced_index = int(replaced_indexs[l])
#                     aug_image[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]] = aux_img[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]]
#                 aug_images.append(aug_image.cuda())
#             # aug_images.append(support_images[i, :, :, :])
#     aug_images=torch.stack(aug_images,dim=0)
#     # print(aug_images.shape)
#     return aug_images

def inference_wo_aug_att(predict,support_feature,query_feature,model,weight,params):

    confidence_scores = torch.nn.Softmax(dim=-1)(predict)  # [75,5]
    topk_scores, topk_labels = confidence_scores.data.topk(1, 1, True, True)
    for i in range(params.nKnovel):
        query_idx = torch.nonzero(topk_labels.view(-1) == i).view(-1)
        support = support_feature[i * params.nExemplars:(i + 1) * params.nExemplars]  # [1,1024]
        query = query_feature[query_idx]


        weight[i] = torch.sum(torch.cat((support, query), dim=0), dim=0)

    weight = model.l2_norm(weight)

    return weight

def jigsaw_aug_round_wo_att(support_images, auxiliary_images, confidence_scores,model,params):
    if params.network in ['ResNet18']:
        weight_point = torch.zeros(support_images.size(0), 512)
    elif params.network in ['ResNet12']:
        weight_point = torch.zeros(support_images.size(0), 1024)

    for i in range(support_images.size(0)):
        aug_images = []
        confidence=confidence_scores

        if auxiliary_images.size(0) == 0:
            aug_features, _ = model(support_images[i].view(-1, 3, 224, 224))
            prototype = aug_features
        # elif auxiliary_images.size(0) == 1:
        else:
            for j in range(auxiliary_images.size(0)):
                replace_block_num = int(torch.round(confidence[j]*9).item())
                # print(replace_block_num)
                replaced_indexs = np.random.choice(9, replace_block_num, replace=False)
                aug_image = support_images[i,:,:,:].clone()

                aux_img = auxiliary_images[j,:,:,:]
                for l in range(replace_block_num):
                    replaced_index = int(replaced_indexs[l])
                    aug_image[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]] = aux_img[:, patch_xl[replaced_index]:patch_xr[replaced_index],patch_yl[replaced_index]:patch_yr[replaced_index]]
                aug_images.append(aug_image.cuda())
            aug_images.append(support_images[i, :, :, :])
            # confidence = torch.cat((confidence,torch.ones(1, 1).cuda()),0)
            aug_images=torch.stack(aug_images,dim=0)

            aug_features,_ = model(aug_images.view(-1,3,224,224))
            # features = aug_features.mul(confidence)
            # attention = torch.sum(confidence, dim=0)
            prototype=torch.sum(aug_features, dim=0)

        weight_point[i] = prototype

    return weight_point