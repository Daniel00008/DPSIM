# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import os
import pdb
import pickle
import pprint
import sys
import time

import _init_paths
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.faster_rcnn.resnet      import resnet
from model.faster_rcnn.vgg16       import vgg16
from model.roi_layers              import nms
from model.rpn.bbox_transform      import bbox_transform_inv, clip_boxes
from model.utils.config            import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils         import save_net, load_net, vis_detections
from parse                         import parse_args_test
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb          import combined_roidb
from torch.autograd                import Variable
import ipdb
import mmcv
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

COLOR_LIST = ['Red', 'Green', 'Blue', 'Cyan', 'Yellow', 'Purple', 'Orange', 'magenta', 'Blueviolet', 'Brown',
              'Burlywood', 'Cadetblue', 'Chartruse', 'Chocolate', 'Coral', 'Cornflowerblue', 'Crimson', 'Darkblue',
              'Darkcyan', 'Darkgoldenrod', 'Darkkhaki', 'Darkmagenta', 'Darkseagreen', 'Darkslateblue', 'Darkslategray',
              'Deeppink', 'Gold', 'Goldenrod', 'Hotpink']
bgr_color_dict = {
    'Red': (0, 0, 255), 'Green': (0, 255, 0), 'Blue': (255, 0, 0), 'Cyan': (255, 255, 0),
    'Yellow': (0, 255, 255), 'Purple': (128, 0, 128), 'Orange': (0, 165, 255),
    'magenta': (255, 0, 255), 'Blueviolet': (226, 43, 138), 'Brown': (42, 42, 165),
    'Burlywood': (135, 184, 222), 'Cadetblue': (160, 158, 95), 'Chartruse': (0, 255, 127), 'Chocolate': (30, 105, 210),
    'Coral': (80, 127, 255), 'Cornflowerblue': (237, 149, 100), 'Crimson': (60, 20, 220), 'Darkblue': (139, 0, 0),
    'Darkcyan': (139, 139, 0), 'Darkgoldenrod': (11, 134, 184), 'Darkkhaki': (107, 183, 189),
    'Darkmagenta': (139, 0, 139), 'Darkseagreen': (143, 188, 143), 'Darkslateblue': (139, 61, 72),
    'Darkslategray': (79, 79, 47),
    'Deeppink': (255, 191, 0), 'Gold': (0, 215, 255), 'Goldenrod': (32, 165, 218), 'Hotpink': (180, 105, 255),

}
def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)
try:
	xrange          # Python 2
except NameError:
	xrange = range  # Python 3


lr           = cfg.TRAIN.LEARNING_RATE
momentum     = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def imshow_det_bboxes(img,
                      result,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      mask_color=None,
                      thickness=2,
                      font_size=13,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None,
                      plt_text=False
                      ):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    # print(img)
    img = mmcv.imread(img).astype(np.uint8)

    # bbox_color = color_val_matplotlib(bbox_color)
    # text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    EPS = 1e-2
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    polygons = []
    color = []

    for lable_index, bboxes in enumerate(result):
        labels = np.ones(len(bboxes),dtype='int') * lable_index
        assert bboxes.ndim == 2, \
            f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
        assert labels.ndim == 1, \
            f' labels ndim should be 1, but its ndim is {labels.ndim}.'
        assert bboxes.shape[0] == labels.shape[0], \
            'bboxes.shape[0] and labels.shape[0] should have the same length.'
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
            f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]

        # mask_colors = []
        #
        # if labels.shape[0] > 0:
        #     if mask_color is None:
        #         # random color
        #         np.random.seed(42)
        #         mask_colors = [
        #             np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        #             for _ in range(max(labels) + 1)
        #         ]
        #     else:
        #         # specify  color
        #         mask_colors = [
        #                           np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
        #                       ] * (
        #                               max(labels) + 1)

        for i, (bbox, label) in enumerate(zip(bboxes, labels)):

            bbox_int = bbox.astype(np.int32)
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons.append(Polygon(np_poly))

            bbox_color = color_val_matplotlib(bgr_color_dict[COLOR_LIST[int(label)]])
            text_color = bbox_color

            color.append(bbox_color)
            label_text = class_names[label] if class_names is not None else f'class {label}'
            if len(bbox) > 4:
                label_text += f'|{bbox[-1]:.02f}'
            if plt_text:
                ax.text(
                    bbox_int[0],
                    bbox_int[1],
                    f'{label_text}',
                    bbox={
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    },
                    color=text_color,
                    fontsize=font_size,
                    verticalalignment='top',
                    horizontalalignment='left')
            if segms is not None:
                color_mask = mask_colors[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # ipdb.set_trace()
    plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img

if __name__ == '__main__':
	# torch.set_num_threads(1)
	args = parse_args_test()

	print('Called with args:')
	print(args)

	if args.dataset == "cityscape":
		args.t_imdbtest_name = "cityscape_2007_test_t"
		args.set_cfgs = [
			"ANCHOR_SCALES",
			"[8,16,32]",
			"ANCHOR_RATIOS",
			"[0.5,1,2]",
			"MAX_NUM_GT_BOXES",
			"30",
		]
	elif args.dataset == "clipart":
		args.t_imdbtest_name = "clipart_trainval"
		args.set_cfgs = [
			"ANCHOR_SCALES",
			"[8,16,32]",
			"ANCHOR_RATIOS",
			"[0.5,1,2]",
			"MAX_NUM_GT_BOXES",
			"20",
		]
	if args.dataset == "kitti2cityscape":
		args.t_imdbtest_name = "cityscape_car_2007_test_s"
		args.set_cfgs = [
			"ANCHOR_SCALES",
			"[8,16,32]",
			"ANCHOR_RATIOS",
			"[0.5,1,2]",
			"MAX_NUM_GT_BOXES",
			"20",
		]
	else:
		print('Undefined Dataset')

	args.cfg_file = (
		"cfgs/{}_ls.yml".format(args.net)
		if args.large_scale
		else "cfgs/{}.yml".format(args.net)
	)

	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
		cfg_from_list(args.set_cfgs)

	print('Using config:')
	pprint.pprint(cfg)
	np.random.seed(cfg.RNG_SEED)

	if torch.cuda.is_available() and not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	cfg.TRAIN.USE_FLIPPED = False
	imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name, False)
	imdb.competition_mode(on=True)
	print(f'{len(roidb)} roidb entries')

	load_name = args.model_dir

	# initilize the network here.
	if args.net == 'vgg16':
		fasterRCNN = vgg16(
			imdb.classes, 
			pretrained=False,
			class_agnostic=args.class_agnostic,
			context=args.context,
			num_aux1=args.num_aux1, num_aux2=args.num_aux2)
	elif args.net == 'res101':
		fasterRCNN = resnet(imdb.classes, 101, pretrained=True,
							class_agnostic=args.class_agnostic,
							context=args.context,
							num_aux1=args.num_aux1, num_aux2=args.num_aux2)
	else:
		print('Undefined Network')

	fasterRCNN.create_architecture()

	print(f"load checkpoint {load_name}")
	checkpoint = torch.load(load_name)
	fasterRCNN.load_state_dict(       
		{k: v for k, v in checkpoint["model"].items() if k in fasterRCNN.state_dict()},
		strict=False
	)
	# fasterRCNN.load_state_dict(checkpoint['model'])
	if 'pooling_mode' in checkpoint.keys():
		cfg.POOLING_MODE = checkpoint['pooling_mode']

	print('load model successfully!')
	# initilize the tensor holder here.
	im_data = torch.FloatTensor(1)
	im_info = torch.FloatTensor(1)
	num_boxes = torch.LongTensor(1)
	gt_boxes = torch.FloatTensor(1)

	# ship to cuda
	if args.cuda:
		im_data = im_data.cuda()
		im_info = im_info.cuda()
		num_boxes = num_boxes.cuda()
		gt_boxes = gt_boxes.cuda()

	# make variable
	im_data = Variable(im_data)
	im_info = Variable(im_info)
	num_boxes = Variable(num_boxes)
	gt_boxes = Variable(gt_boxes)

	if args.cuda:
		cfg.CUDA = True

	if args.cuda:
		fasterRCNN.cuda()

	start = time.time()
	max_per_image = 100

	vis = args.vis

	if vis:
		thresh = 0.05
	else:
		thresh = 0.0

	save_name = args.model_dir.split("/")[-1]
	num_images = len(imdb.image_index)
	all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]

	output_dir = get_output_dir(imdb, save_name)
	dataset = roibatchLoader(
		roidb, 
		ratio_list, 
		ratio_index, 
		1,
		imdb.num_classes, 
		training=False, 
		normalize=False
	)

	dataloader = torch.utils.data.DataLoader(
		dataset, 
		batch_size=1,
		shuffle=False, 
		num_workers=0,
		pin_memory=True
	)

	data_iter = iter(dataloader)

	_t = {'im_detect': time.time(), 'misc': time.time()}
	det_file = os.path.join(output_dir, 'detections.pkl')

	fasterRCNN.eval()
	empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
	for i in range(num_images):

		data = next(data_iter)
		with torch.no_grad():
			im_data.resize_(data[0].size()).copy_(data[0])
			im_info.resize_(data[1].size()).copy_(data[1])
			gt_boxes.resize_(data[2].size()).copy_(data[2])
			num_boxes.resize_(data[3].size()).copy_(data[3])

		det_tic = time.time()
		(
			rois,
			cls_prob,
			bbox_pred,
			rpn_loss_cls,
			rpn_loss_box,
			RCNN_loss_cls,
			RCNN_loss_bbox,_, _, _, _, _, _, _, _
		) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

		scores = cls_prob.data
		boxes = rois.data[:, :, 1:5]

		if cfg.TEST.BBOX_REG:
			# Apply bounding-box regression deltas
			box_deltas = bbox_pred.data
			if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
				# Optionally normalize targets by a precomputed mean and stdev
				if args.class_agnostic:
					box_deltas = (
						box_deltas.view(-1, 4)
						* torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
						+ torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
					)
					box_deltas = box_deltas.view(1, -1, 4)
				else:
					box_deltas = (
						box_deltas.view(-1, 4)
						* torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
						+ torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
					)
					box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

			pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
			pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
		else:
			# Simply repeat the boxes, once for each class
			pred_boxes = np.tile(boxes, (1, scores.shape[1]))

		pred_boxes /= data[1][0][2].item()

		scores = scores.squeeze()
		pred_boxes = pred_boxes.squeeze()
		det_toc = time.time()
		detect_time = det_toc - det_tic
		misc_tic = time.time()
		if vis:
			im = cv2.imread(imdb.image_path_at(i))
			im2show = np.copy(im)
        
		det_results = []
		for j in xrange(1, imdb.num_classes):
			inds = torch.nonzero(scores[:, j] > thresh).view(-1)
			# if there is det
			if inds.numel() > 0:
				cls_scores = scores[:, j][inds]
				_, order = torch.sort(cls_scores, 0, True)
				if args.class_agnostic:
					cls_boxes = pred_boxes[inds, :]
				else:
					cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

				cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
				# cls_dets = torch.cat((cls_boxes, cls_scores), 1)
				cls_dets = cls_dets[order]
				keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
				cls_dets = cls_dets[keep.view(-1).long()]
				if vis:
					im2show = vis_detections(
						im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3
					)
				all_boxes[j][i] = cls_dets.cpu().numpy()
				det_results.append(cls_dets.cpu().numpy())
			else:
				all_boxes[j][i] = empty_array
				det_results.append(empty_array)                
		# draw bbox*
   
		if vis:
			save_name = 'msips_' + imdb.image_path_at(i).rsplit('/')[-1]
			out_file = os.path.join(output_dir + save_name)
			imshow_det_bboxes(im2show,
                          det_results,
                          segms=None,
                          class_names=imdb.classes[1:],
                          score_thr=0.5,
                          mask_color=None,
                          thickness=2,
                          font_size=13,
                          win_name='',
                          show=False,
                          wait_time=0,
                          out_file=out_file,
                          plt_text=False
                          )            
		# Limit to max_per_image detections *over all classes*
		if max_per_image > 0:
			image_scores = np.hstack(
				[all_boxes[j][i][:, -1] for j in xrange(1, imdb.num_classes)]
			)
			if len(image_scores) > max_per_image:
				image_thresh = np.sort(image_scores)[-max_per_image]
				for j in xrange(1, imdb.num_classes):
					keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
					all_boxes[j][i] = all_boxes[j][i][keep, :]

		misc_toc = time.time()
		nms_time = misc_toc - misc_tic

		sys.stdout.write(
			"im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r".format(
				i + 1, num_images, detect_time, nms_time
			)
		)
		sys.stdout.flush()
		if vis:
			cv2.imwrite('result.png', im2show)
			pdb.set_trace()
			cv2.imshow('test', im2show)
			cv2.waitKey(0)

	with open(det_file, 'wb') as f:
		pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

	print('Evaluating detections')
	imdb.evaluate_detections(all_boxes, output_dir)

	end = time.time()
	print("test time: %0.4fs" % (end - start))
