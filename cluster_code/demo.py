import torch

from cluster_code.pairwise import Class2Simi


# def prepare_task_target(input, target, args):
#     """
#     如果是分类的话，输出的两个变量都是类别标签
#     如果是聚类的话，输出的第一个是相似度标签，第二个是类别标签
#     如果是训练相似度预测网络的话，输出的两个变量都是相似度标签
#     :param input:
#     :param target:
#     :param args:
#     :return:
#     """
#     if args.use_SPN:
#         if args.SPN is None:
#             # print("# from original class label prepare labels")
#             train_target = eval_target = Class2Simi(target, mode='cls')
#         else:
#             # print("# from predicted class label prepare labels")
#             _, train_target = args.SPN(input).max(1)  # Binaries the predictions
#             train_target = train_target.float()
#             train_target[train_target == 0] = -1  # Simi:1, Dissimi:-1
#             eval_target = target
#     else:
#         # print("# from original class label prepare labels")
#         # Convert class labels to pairwise similarity
#         train_target = Class2Simi(target, mode='hinge')  # batch_size²
#         eval_target = target
#
#     return train_target.detach(), eval_target.detach()  # Make sure no gradients


def prepare_task_target(input, target, args):
    """
    如果是分类的话，输出的两个变量都是类别标签
    如果是聚类的话，输出的第一个是相似度标签，第二个是类别标签
    如果是训练相似度预测网络的话，输出的两个变量都是相似度标签
    :param input:
    :param target:
    :param args:
    :param SPN2CPU:
    :return:
    """
    input4spn = input.clone()

    if args.use_SPN:
        # 如果使用SPN train_target是预测得到的相似与否标签, eval_target是正确的相似与否的标签
        if args.SPN is None:
            print("Please provide a pretrained SPN model!")
        else:
            if args.SPN2CPU:
                input4spn = input4spn.cpu()
                args.SPN = args.cpu()

            train_target_predict = args.SPN(input4spn)
            _, train_target = train_target_predict.max(1)  # Binaries the predictions
            train_target = train_target.float()
            train_target[train_target == 0] = -1  # Simi:1, Dissimi:-1

            if args.SPN2CPU:
                # 如果SPN之前在CPU上运算的，还将相似度标签放回GPU
                train_target = train_target.cuda()

            eval_target = Class2Simi(target, mode='hinge')  # batch_size²
    else:
        # 如果不使用SPN那得到的train_target和eval_target都是标准的相似与否的标签
        train_target = Class2Simi(target, mode='hinge')  # batch_size²
        eval_target = train_target

    return train_target.detach(), eval_target.detach()  # Make sure no gradients
