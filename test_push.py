from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict

from my_utils import *
from robot import Panda, UR5Robotiq85, UR5Robotiq140
import matplotlib.pyplot as plt
import cv2

import pybullet_data as pd
import pybullet as p



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                        shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                        display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True

    if args.seed is not None:
        random.seed(args.seed)


#iou_thresholds = [x / 100 for x in range(50, 100, 5)]
iou_thresholds = [0.8] # 0.8
coco_cats = {}  # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})


def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                        crop_masks=args.crop,
                        score_threshold=args.score_threshold)
        # print("class", t[0].shape)
        # print("scores", t[1].shape)
        # print("boxes", t[2].shape)
        # print("maskes", t[3].shape)
        # print("entropy", t[4].shape)
        # for i in range(t[3].shape[0]):
        #    print(np.unique(t[3][i, :, :].cpu()))
        # exit()
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    if args.display_fps:
        # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if num_dets_to_consider == 0:
        return img_numpy

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

    return img_numpy


def prep_benchmark(dets_out, h, w):
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

    with timer.env('Copy'):
        classes, scores, boxes, masks = [x[:args.top_k] for x in t]
        if isinstance(scores, list):
            box_scores = scores[0].cpu().numpy()
            mask_scores = scores[1].cpu().numpy()
        else:
            scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        boxes = boxes.cpu().numpy()
        masks = masks.cpu().numpy()

    with timer.env('Sync'):
        # Just in case
        torch.cuda.synchronize()


def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]


def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats_inv[coco_cat_id]


class Detections:

    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x) * 10) / 10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })

    def dump(self):
        dump_arguments = [
            (self.bbox_data, args.bbox_det_file),
            (self.mask_data, args.mask_det_file)
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)

    def dump_web(self):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                       'use_yolo_regressors', 'use_prediction_matching',
                       'train_masks']

        output = {
            'info': {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }

        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)


def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()


def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()


def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections: Detections = None):
    """ Returns a list of APs for this image, with each element being for a class  """
    if not args.output_coco_json:
        with timer.env('Prepare gt'):
            gt_boxes = torch.Tensor(gt[:, :4])
            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h
            gt_classes = list(gt[:, 4].astype(int))
            gt_masks = torch.Tensor(gt_masks).view(-1, h * w)

            if num_crowd > 0:
                split = lambda x: (x[-num_crowd:], x[:-num_crowd])
                crowd_boxes, gt_boxes = split(gt_boxes)
                crowd_masks, gt_masks = split(gt_masks)
                crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        #classes, scores, boxes, masks, _ = postprocess(dets, w, h, crop_masks=args.crop,
        #                                               score_threshold=args.score_threshold)
        classes = dets[0]
        scores = dets[1]
        boxes = dets[2]
        masks = dets[3]
        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int))
        if isinstance(scores, list):
            box_scores = list(scores[0].cpu().numpy().astype(float))
            mask_scores = list(scores[1].cpu().numpy().astype(float))
        else:
            scores = list(scores.cpu().numpy().astype(float))
            box_scores = scores
            mask_scores = scores
        masks = masks.view(-1, h * w).cuda()
        boxes = boxes.cuda()

    if args.output_coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes[i, :], box_scores[i])
                    detections.add_mask(image_id, classes[i], masks[i, :, :], mask_scores[i])
            return

    with timer.env('Eval Setup'):
        num_pred = len(classes)
        num_gt = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks)
        # bbox_iou_cache [n, m] 每个预测与GT的交并比

        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float())


        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box', lambda i, j: bbox_iou_cache[i, j].item(),
             lambda i, j: crowd_bbox_iou_cache[i, j].item(),
             lambda i: box_scores[i], box_indices),
            ('mask', lambda i, j: mask_iou_cache[i, j].item(),
             lambda i, j: crowd_mask_iou_cache[i, j].item(),
             lambda i: mask_scores[i], mask_indices)
        ]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])
        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)

                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue

                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue

                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j

                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue

                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)
    timer.stop('Main loop')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0
        self.data_points_print = []

    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))
        self.data_points_print.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0, self.data_points, self.data_points_print

        # Sort descending by score

        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        num_true = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives
            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range), self.data_points, self.data_points_print


def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.

    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = ((x >> 16) ^ x) & 0xFFFFFFFF
    return x


def evalimage(net: Yolact, path: str, save_path: str = None):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)

    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]

    if save_path is None:
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)


def evalimages(net: Yolact, input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print()
    for p in Path(input_folder).glob('*'):
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)

        evalimage(net, path, out_path)
        print(path + ' -> ' + out_path)
    print('Done.')


from multiprocessing.pool import ThreadPool
from queue import Queue


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


def evalvideo(net: Yolact, path: str, out_path: str = None):
    # If the path is a digit, parse it as a webcam index
    is_webcam = path.isdigit()

    # If the input image size is constant, this make things faster (hence why we can use it in a video setting).
    cudnn.benchmark = True

    if is_webcam:
        vid = cv2.VideoCapture(int(path))
    else:
        vid = cv2.VideoCapture(path)

    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)

    target_fps = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_webcam:
        num_frames = float('inf')
    else:
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(100)
    fps = 0
    frame_time_target = 1 / target_fps
    running = True
    fps_str = ''
    vid_done = False
    frames_displayed = 0

    if out_path is not None:
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        if out_path is not None:
            out.release()
        cv2.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        frames = []
        for idx in range(args.video_multiframe):
            frame = vid.read()[1]
            if frame is None:
                return frames
            frames.append(frame)
        return frames

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            num_extra = 0
            while imgs.size(0) < args.video_multiframe:
                imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                num_extra += 1
            out = net(imgs)
            if num_extra > 0:
                out = out[:-num_extra]
            return frames, out

    def prep_frame(inp, fps_str):
        with torch.no_grad():
            frame, preds = inp
            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True, fps_str=fps_str)

    frame_buffer = Queue()
    video_fps = 0

    # All this timing code to make sure that
    def play_video():
        try:
            nonlocal frame_buffer, running, video_fps, is_webcam, num_frames, frames_displayed, vid_done

            video_frame_times = MovingAverage(100)
            frame_time_stabilizer = frame_time_target
            last_time = None
            stabilizer_step = 0.0005
            progress_bar = ProgressBar(30, num_frames)

            while running:
                frame_time_start = time.time()

                if not frame_buffer.empty():
                    next_time = time.time()
                    if last_time is not None:
                        video_frame_times.add(next_time - last_time)
                        video_fps = 1 / video_frame_times.get_avg()
                    if out_path is None:
                        cv2.imshow(path, frame_buffer.get())
                    else:
                        out.write(frame_buffer.get())
                    frames_displayed += 1
                    last_time = next_time

                    if out_path is not None:
                        if video_frame_times.get_avg() == 0:
                            fps = 0
                        else:
                            fps = 1 / video_frame_times.get_avg()
                        progress = frames_displayed / num_frames * 100
                        progress_bar.set_val(frames_displayed)

                        print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                              % (repr(progress_bar), frames_displayed, num_frames, progress, fps), end='')

                # This is split because you don't want savevideo to require cv2 display functionality (see #197)
                if out_path is None and cv2.waitKey(1) == 27:
                    # Press Escape to close
                    running = False
                if not (frames_displayed < num_frames):
                    running = False

                if not vid_done:
                    buffer_size = frame_buffer.qsize()
                    if buffer_size < args.video_multiframe:
                        frame_time_stabilizer += stabilizer_step
                    elif buffer_size > args.video_multiframe:
                        frame_time_stabilizer -= stabilizer_step
                        if frame_time_stabilizer < 0:
                            frame_time_stabilizer = 0

                    new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
                else:
                    new_target = frame_time_target

                next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
                target_time = frame_time_start + next_frame_target - 0.001  # Let's just subtract a millisecond to be safe

                if out_path is None or args.emulate_playback:
                    # This gives more accurate timing than if sleeping the whole amount at once
                    while time.time() < target_time:
                        time.sleep(0.001)
                else:
                    # Let's not starve the main thread, now
                    time.sleep(0.001)
        except:
            # See issue #197 for why this is necessary
            import traceback
            traceback.print_exc()

    extract_frame = lambda x, i: (
    x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])

    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')
    first_batch = eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)
    pool.apply_async(play_video)
    active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]

    print()
    if out_path is None: print('Press Escape to close.')
    try:
        while vid.isOpened() and running:
            # Hard limit on frames in buffer so we don't run out of memory >.>
            while frame_buffer.qsize() > 100:
                time.sleep(0.001)

            start_time = time.time()

            # Start loading the next frames from the disk
            if not vid_done:
                next_frames = pool.apply_async(get_next_frame, args=(vid,))
            else:
                next_frames = None

            if not (vid_done and len(active_frames) == 0):
                # For each frame in our active processing queue, dispatch a job
                # for that frame using the current function in the sequence
                for frame in active_frames:
                    _args = [frame['value']]
                    if frame['idx'] == 0:
                        _args.append(fps_str)
                    frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)

                # For each frame whose job was the last in the sequence (i.e. for all final outputs)
                for frame in active_frames:
                    if frame['idx'] == 0:
                        frame_buffer.put(frame['value'].get())

                # Remove the finished frames from the processing queue
                active_frames = [x for x in active_frames if x['idx'] > 0]

                # Finish evaluating every frame in the processing queue and advanced their position in the sequence
                for frame in list(reversed(active_frames)):
                    frame['value'] = frame['value'].get()
                    frame['idx'] -= 1

                    if frame['idx'] == 0:
                        # Split this up into individual threads for prep_frame since it doesn't support batch size
                        active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in
                                          range(1, len(frame['value'][0]))]
                        frame['value'] = extract_frame(frame['value'], 0)

                # Finish loading in the next frames and add them to the processing queue
                if next_frames is not None:
                    frames = next_frames.get()
                    if len(frames) == 0:
                        vid_done = True
                    else:
                        active_frames.append({'value': frames, 'idx': len(sequence) - 1})

                # Compute FPS
                frame_times.add(time.time() - start_time)
                fps = args.video_multiframe / frame_times.get_avg()
            else:
                fps = 0

            fps_str = 'Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d' % (
            fps, video_fps, frame_buffer.qsize())
            if not args.display_fps:
                print('\r' + fps_str + '    ', end='')

    except KeyboardInterrupt:
        print('\nStopping...')

    cleanup_and_exit()


def evaluate(net: Yolact, dataset, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    # TODO Currently we do not support Fast Mask Re-scroing in evalimage, evalimages, and evalvideo
    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            evalimage(net, inp, out)
        else:
            evalimage(net, args.image)
        return
    elif args.images is not None:
        inp, out = args.images.split(':')
        evalimages(net, inp, out)
        return
    elif args.video is not None:
        if ':' in args.video:
            inp, out = args.video.split(':')
            evalvideo(net, inp, out)
        else:
            evalvideo(net, args.video)
        return

    frame_times = MovingAverage()
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)

    print()

    if not args.display and not args.benchmark:
        # For each class and iou, stores tuples (score, isPositive)
        # Index ap_data[type][iouIdx][classIdx]
        ap_data = {
            'box': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
            'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
        }
        detections = Detections()
    else:
        timer.disable('Load Data')

    dataset_indices = list(range(len(dataset)))

    if args.shuffle:
        random.shuffle(dataset_indices)
    elif not args.no_sort:
        # Do a deterministic shuffle based on the image ids
        #
        # I do this because on python 3.5 dictionary key order is *random*, while in 3.6 it's
        # the order of insertion. That means on python 3.6, the images come in the order they are in
        # in the annotations file. For some reason, the first images in the annotations file are
        # the hardest. To combat this, I use a hard-coded hash function based on the image ids
        # to shuffle the indices we use. That way, no matter what python version or how pycocotools
        # handles the data, we get the same result every time.
        hashed = [badhash(x) for x in dataset.ids]
        dataset_indices.sort(key=lambda x: hashed[x])

    dataset_indices = dataset_indices[:dataset_size]

    try:
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):
            timer.reset()

            with timer.env('Load Data'):
                img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)
                # print("load label==============================")
                # print(img.shape)
                # print(gt.shape)
                # print(gt[:4, :])
                # print(gt_masks.shape)
                # print(np.unique(gt_masks[0, :, :]))
                # print((h, w))
                # print(num_crowd)
                # exit()
                # Test flag, do not upvote
                if cfg.mask_proto_debug:
                    with open('scripts/info.txt', 'w') as f:
                        f.write(str(dataset.ids[image_idx]))
                    np.save('scripts/gt.npy', gt_masks)

                batch = Variable(img.unsqueeze(0))
                if args.cuda:
                    batch = batch.cuda()

            with timer.env('Network Extra'):
                preds = net(batch)
            # Perform the meat of the operation here depending on our mode.
            if args.display:
                img_numpy = prep_display(preds, img, h, w)
            elif args.benchmark:
                prep_benchmark(preds, h, w)
            else:
                prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections)

            # First couple of images take longer because we're constructing the graph.
            # Since that's technically initialization, don't include those in the FPS calculations.
            if it > 1:
                frame_times.add(timer.total_time())

            if args.display:
                if it > 1:
                    print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                plt.imshow(img_numpy)
                plt.title(str(dataset.ids[image_idx]))
                plt.show()
            elif not args.no_bar:
                if it > 1:
                    fps = 1 / frame_times.get_avg()
                else:
                    fps = 0
                progress = (it + 1) / dataset_size * 100
                progress_bar.set_val(it + 1)
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                      % (repr(progress_bar), it + 1, dataset_size, progress, fps), end='')

        if not args.display and not args.benchmark:
            print()
            if args.output_coco_json:
                print('Dumping detections...')
                if args.output_web_json:
                    detections.dump_web()
                else:
                    detections.dump()
            else:
                if not train_mode:
                    print('Saving data...')
                    with open(args.ap_data_file, 'wb') as f:
                        pickle.dump(ap_data, f)

                return calc_map(ap_data)
        elif args.benchmark:
            print()
            print()
            print('Stats for the last frame:')
            timer.print_stats()
            avg_seconds = frame_times.get_avg()
            print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000 * avg_seconds))

    except KeyboardInterrupt:
        print('Stopping...')

def get_tp(ap_data, class_list):
    tp_index = []
    for _class in range(len(cfg.dataset.class_names)):
        index = np.where(class_list == _class)[0]
        for iou_idx in range(len(iou_thresholds)):
            iou_type = 'box'
            ap_obj = ap_data[iou_type][iou_idx][_class]
            if not ap_obj.is_empty():
                ap_list = ap_obj.get_ap()[2]
                # ap_list 保存该class下的样本判定
                for i in range(len(ap_list)):
                    if ap_list[i][1]:
                        tp_index.append(index[i])
    return np.asarray(tp_index)

def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]
    tp_box = []
    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]
                if not ap_obj.is_empty():
                    #print(ap_obj.get_ap())
                    aps[iou_idx][iou_type].append(ap_obj.get_ap()[0])
                    #if iou_type == 'box':
                    #    print(_class)
                    #    print(ap_obj.get_ap()[2])

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

    #print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    #all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps


def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n: ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root_base_path = os.getcwd()

    trained_model = os.path.join(root_base_path, "weights", "yolact_resnet50.pth")
    config_name = "yolact_resnet50_config_my"

    set_cfg(config_name)
    with torch.no_grad():
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        dataset = None
        net = Yolact()
        net = net.to(device)
        net.load_weights(trained_model)
        net.eval()

    p.connect(p.DIRECT) #p.DIRECT

    viewMat_list = [
        [1.0, 0.0, -0.0, 0.0, -0.0, 1.0, -0.00017452123574912548, 0.0, 0.0, 0.00017452123574912548, 1.0, 0.0, 0.5,
         -7.417152664856985e-05, -0.4266100227832794, 1.0],
        [1.0, 0.0, -0.0, 0.0, -0.0, 0.8660240173339844, -0.5000023245811462, 0.0, 0.0, 0.5000023245811462,
         0.8660240173339844, 0.0, 0.5, 2.925097942352295e-05, -0.426557332277298, 1.0],
        [0.017451846972107887, 0.8658873438835144, -0.49993473291397095, 0.0, -0.9998477697372437, 0.015113634057343006,
         -0.00872611254453659, 0.0, 0.0, 0.5000109076499939, 0.8660191893577576, 0.0, 0.012434440664947033,
         0.4329407513141632, -0.6765085458755493, 1.0],
        [-0.999847412109375, 0.015133202075958252, -0.008737090043723583, 0.0, -0.017474284395575523,
         -0.8658948540687561, 0.49992072582244873, 0.0, 0.0, 0.4999971091747284, 0.8660269975662231, 0.0,
         -0.4962104260921478, 0.007570326328277588, -0.4309096336364746, 1.0],
        [-0.01745261251926422, -0.8658953309059143, 0.4999208450317383, 0.0, 0.9998477697372437, -0.015114436857402325,
         0.008726253174245358, 0.0, 0.0, 0.49999701976776123, 0.8660271763801575, 0.0, -0.005017626099288464,
         -0.43294382095336914, -0.1765807718038559, 1.0]]

    projMat = [0.74999994, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.00002003, -1.0, 0.0, 0.0, -0.0200002, 0.0]

    heightmap_resolution = 0.002
    list_1 = []
    list_2 = []
    list_mIoU_mean = []
    retrite_mean = []
    retrite_all = []
    AR_mean = []

    name_list_path = os.path.join(root_base_path, "objects", "name_list.txt")
    f = open(name_list_path)
    base_path = os.path.join(root_base_path, "objects", "urdf27")
    object_mesh_list = []
    object_name_list = []
    for line in f:
        object_mesh_list.append(line.strip())
    f.close()
    label_list_path = os.path.join(root_base_path, "objects", "label_list.txt")
    f = open(label_list_path)
    object_label_list = []
    for line in f:
        object_label_list.append(line.strip())
    f.close()

    case_path = os.path.join(root_base_path, "objects", "world_ints", "case_1.txt")

    f1 = open(case_path)
    pose_path = os.path.join(root_base_path, "objects", "world_ints", "pos_1.txt")

    obj_pos = np.loadtxt(pose_path)
    for line in f1:
        object_name_list.append(line.strip())
    f1.close()
    for test_time_all in range(1):

        planeId = p.loadURDF(os.path.join(pd.getDataPath(), "plane.urdf"))


        # robot_workspace_limits = np.asarray([[-0.884, -0.085], [-0.206, 0.217], [-0.0001, 0.45]])
        robot_workspace_limits_one_stage = np.asarray([[-1.5, -0.0], [-0.5, 0.5], [-0.0001, 0.45]])
        robot_workspace_limits_two_stage = np.asarray([[-2, -0.0], [-1, 1], [-0.0001, 0.45]])
        flags = p.URDF_USE_INERTIA_FROM_FILE
        number_list = [25, 26]

        number_one = len(object_name_list)
        obj_id_list = []
        object_mesh_id = []


        # ---------------------------------------
        for object_idx in range(number_one):
            object_name = object_name_list[object_idx]
            object_mesh_id.append(object_mesh_list.index(object_name) + 1)
            obj_id = p.loadURDF(os.path.join(base_path, object_name),
                                basePosition=obj_pos[object_idx, :3],
                                baseOrientation=obj_pos[object_idx, 3:], flags=flags, globalScaling=1.0)
            obj_id_list.append(obj_id)
        p.setGravity(0.000000, 0.000000, -9.800000)
        p.stepSimulation()

        object_mesh_id.insert(0, 0)
        robot = UR5Robotiq85((-0.5, 0.5, 0.25), (0, 0, 0))
        robot.load()

        robot.move_ee([-0.0, 0.0, 0.1, 0.0, np.pi / 2, np.pi / 2], 'end')
        robot.move_gripper(0)
        for _ in range(1000):
            p.stepSimulation()
        robot.move_ee([0.0, 0.0, 0.611, 0.0, np.pi / 2, np.pi / 2], 'end')
        for _ in range(1000):
            p.stepSimulation()
        robot.move_ee([-0.5, 0.0, 0.611, 0.0, np.pi / 2, np.pi / 2], 'end')
        for _ in range(2000):
            p.stepSimulation()

        # robot.reset()
        mIoU_list = []
        w_mIoU_list = []
        data_list = []
        start_list = []
        end_list = []
        retrite_one = []
        AR_one = []
        shift = 0.11
        retrite_start = 0.0
        mIoU_start = 0.0
        flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX

        push_time = 0
        for test_time in range(5):
            segment_heat_map_list = []
            position_list = []
            mask_list = []
            mask_tp_list = []
            class_tp_list = []
            view_result_list = []
            class_list = []
            proj_factor_list = []
            score_list = []
            result_list = []


            for view_id in range(5):
                viewMat = viewMat_list[view_id]

                width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=1440, height=1024, viewMatrix=viewMat,
                                                                           projectionMatrix=projMat, flags=flags)
                if view_id == 0:
                    segImg_top = segImg.copy()

                color_img = np.asarray(rgbImg)
                color_img = color_img[:, :, :3]
                color_img = color_img.astype(np.uint8)
                color_img = color_img[:, :, ::-1]  # RGB-->BGR
                pred_img = color_img.copy()
                #if view_id == 0:
                print_img = color_img.copy()

                frame = torch.from_numpy(pred_img).to(device).float()
                batch = FastBaseTransform()(frame.unsqueeze(0))
                with torch.no_grad():
                    preds = net(batch)

                    h, w, _ = frame.shape
                    t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=0.35)
                view_result_list.append(t)
                segment_map = t[4].permute(1, 2, 0).clone()
                segment_heat_map = entropy_map(segment_map)

                segment_heat_map_list.append(segment_heat_map)

                position, proj_factor = get_point_view(depthImg, viewMat, projMat)
                position_list.append(position)
                proj_factor_list.append(proj_factor)
                mask = t[3].clone()
                mask_list.append(mask.cpu().numpy())
                class_one = t[0].clone()
                class_list.append(class_one.cpu().numpy())
                score = t[1].clone()
                score_list.append(score.cpu().numpy())

                ap_data = {
                    'box': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
                    'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
                }
                detections = Detections()
                gt, gt_masks, h, w, num_crowd = get_gt_label(segImg, object_mesh_id, obj_id_list)
                prep_metrics(ap_data, t, color_img, gt, gt_masks, h, w, num_crowd, 0, detections)
                result = calc_map(ap_data)
                result_list.append(result['box']['all'])
                tp_index = get_tp(ap_data, class_one.cpu().numpy())
                tp_mask = mask.cpu().numpy()[tp_index]
                tp_class = class_one.cpu().numpy()[tp_index]
                mask_tp_list.append(tp_mask)
                class_tp_list.append(tp_class)

            obj_pts_list_all, obj_class_all = get_instance(mask_tp_list, position_list,  class_tp_list, device)

            obj_class_all = np.asarray(obj_class_all)
            obj_class_all = obj_class_all + 1
            result_list = np.asarray(result_list)
            AP = np.mean(result_list) / 100
            P, R, _ = get_retrieval(gt_list=object_mesh_id[1:], pred_list=obj_class_all)
            F1_score = (2 * AP * R) / (AP + R)
            print("3D F1_score is: ", F1_score)
            print("3D P is ", AP)
            print("3D R is", R)
            retrite_one.append(F1_score)
            AR_one.append(R)

            if test_time > 1:
                shift = get_shift(start_list, shift=shift)
            obj_pts_list = []

            mask_sum_list = []
            entropy_one_normal_list = []
            for mask_id in range(len(mask_list)):
                mask_one = mask_list[mask_id]
                mask_one = np.sum(mask_one, axis=0)
                mask_sum_list.append(mask_one)

                entropy_one = segment_heat_map_list[mask_id]
                entropy_normal_one = data_normal_2d(entropy_one)
                entropy_one_normal_list.append(entropy_normal_one.cpu())

            entropy_heightmap_list = []
            mask_heightmap_list = []
            for view_id in range(len(mask_sum_list)):
                position = position_list[view_id]
                entropy_map_all = entropy_one_normal_list[view_id]
                masks = mask_sum_list[view_id]
                if view_id == 0:
                    depth_heightmap, entropy_heightmap, mask_heightmap = get_heightmap(position[:, :3], robot_workspace_limits_one_stage,
                                                                          entropy_map_all, masks,
                                                                          heightmap_resolution=heightmap_resolution)
                else:
                    _, entropy_heightmap, mask_heightmap = get_heightmap(position[:, :3], robot_workspace_limits_one_stage,entropy_map_all, masks,
                                                                             heightmap_resolution=heightmap_resolution)
                entropy_heightmap_list.append(entropy_heightmap)
                mask_heightmap_list.append(mask_heightmap.reshape([1, mask_heightmap.shape[0], -1]))

            entropy_heightmap = get_entropy_mask(entropy_heightmap_list, mask_heightmap_list, time=test_time)

            entropy_heightmap = torch.from_numpy(entropy_heightmap).float().to(device)
            entropy_map_pred = entropy_heightmap.clone()
            point_list = creat_patch_conv(heat_map=entropy_heightmap, batch_size=1, device=device, num_patch=1,
                                          test_time=test_time)

            # start_point
            print(point_list)
            # push
            valid_depth_heightmap = depth_heightmap.copy()
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

            entropy_heightmap = entropy_heightmap.cpu()
            pushes, target_center, push_score = push_generator(point_list, valid_depth_heightmap, entropy_heightmap)

            if push_time > 0:
                if push_score < 0.89: # 0.89
                    break
            push_time = push_time + 1
            best_pix_x = pushes[0]
            best_pix_y = pushes[1]

            primitive_position = [best_pix_x * heightmap_resolution + robot_workspace_limits_one_stage[0][0],
                                  best_pix_y * heightmap_resolution + robot_workspace_limits_one_stage[1][0],
                                  valid_depth_heightmap[best_pix_y][best_pix_x] + robot_workspace_limits_one_stage[2][0]]

            end_pix_x = target_center[0]
            end_pix_y = target_center[1]

            pos_robot_end = [end_pix_x * heightmap_resolution + robot_workspace_limits_one_stage[0][0],
                             end_pix_y * heightmap_resolution + robot_workspace_limits_one_stage[1][0],
                             valid_depth_heightmap[end_pix_y][end_pix_x] + robot_workspace_limits_one_stage[2][0]]
            primitive_position_end = get_push_end(primitive_position, pos_robot_end, shift=shift)
            primitive_position = np.array(primitive_position).reshape([1, -1])

            primitive_position[:, 2] = primitive_position[:, 2] + 0.168 #0.158
            primitive_position_end[:, 2] = primitive_position_end[:, 2] + 0.168 #0.158

            np.clip(primitive_position[:, 0], -0.884, -0.085, out=primitive_position[:, 0])
            np.clip(primitive_position[:, 1], -0.206, 0.217, out=primitive_position[:, 1])
            np.clip(primitive_position[:, 2], 0.189, 0.6, out=primitive_position[:, 2])
            np.clip(primitive_position_end[:, 0], -0.884, -0.085, out=primitive_position_end[:, 0])
            np.clip(primitive_position_end[:, 1], -0.206, 0.217, out=primitive_position_end[:, 1])
            np.clip(primitive_position_end[:, 2], 0.189, 0.6, out=primitive_position_end[:, 2])
            print("start point", primitive_position)
            print("end point", primitive_position_end)
            print("shift", shift)
            start_list.append(primitive_position)
            end_list.append(primitive_position_end)


            robot.move_ee([primitive_position[0, 0], primitive_position[0, 1], 0.55, 0.0, np.pi / 2, np.pi / 2], 'end')
            robot.move_gripper(0)
            for _ in range(1000):
                p.stepSimulation()
            robot.move_ee([primitive_position[0, 0], primitive_position[0, 1], 0.35, 0.0, np.pi / 2, np.pi / 2], 'end')
            robot.move_gripper(0)
            for _ in range(1000):
                p.stepSimulation()
            down_z = np.linspace(0.35, primitive_position[0, 2], num=5)
            for down_z_id in range(1, down_z.shape[0]):
                robot.move_ee(
                    [primitive_position[0, 0], primitive_position[0, 1], down_z[down_z_id], 0.0, np.pi / 2, np.pi / 2],
                    'end')
                for _ in range(480):
                    p.stepSimulation()

            x_list, y_list = get_push_slow_list(primitive_position, primitive_position_end)
            # robot.move_ee([primitive_position_end[0, 0], primitive_position_end[0, 1], primitive_position[0, 2], 0.0, np.pi / 2, np.pi / 2], 'end')
            # robot.move_gripper(0)
            for push_id in range(x_list.shape[0]):
                robot.move_ee(
                    [x_list[push_id], y_list[push_id], primitive_position[0, 2], 0.0,
                     np.pi / 2, np.pi / 2], 'end')
                robot.move_gripper(0)
                for _ in range(480):
                    p.stepSimulation()

            up_z = np.linspace(primitive_position[0, 2], 0.35, num=5)
            for up_id in range(1, up_z.shape[0]):
                robot.move_ee(
                    [primitive_position_end[0, 0], primitive_position_end[0, 1], up_z[up_id], 0.0,
                     np.pi / 2, np.pi / 2], 'end')
                for _ in range(480):
                    p.stepSimulation()
            robot.move_ee(
                [primitive_position_end[0, 0], primitive_position_end[0, 1], 0.611, 0.0, np.pi / 2, np.pi / 2], 'end')
            robot.move_gripper(0)
            for _ in range(1000):
                p.stepSimulation()
            robot.move_ee([-0.5, 0.0, 0.611, 0.0, np.pi / 2, np.pi / 2], 'end')
            robot.move_gripper(0)
            for _ in range(1000):
                p.stepSimulation()


        retrite_mean.append(retrite_one[-1] - retrite_one[0])
        retrite_all.append(retrite_one[-1])
        AR_mean.append(AR_one[-1] - AR_one[0])
        print("now, F1_score mean increase: ", np.mean(retrite_mean))
        print("now, AR mean increase:", np.mean(AR_mean))
        print("now, retrite mean: ", np.mean(retrite_all))
        print("now, we have push:", push_time)
        p.resetSimulation()


if __name__ == '__main__':
    parse_args()
    main()


