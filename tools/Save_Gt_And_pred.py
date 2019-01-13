#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import xml.etree.ElementTree as ET
import numpy as np
import os, cv2
import argparse
import pickle
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'Insulator', 'Rotary_double_ear', 'Binaural_sleeve', 'Brace_sleeve',
           'Steady_arm_base', 'Bracing_wire_hook', 'Double_sleeve_connector', 'Messenger_wire_base',
           'Windproof_wire_ring', 'Insulator_base', 'Isoelectric_line', 'Brace_sleeve_screw')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_40000.ckpt',),'res101': ('res101_faster_rcnn_iter_40000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    im_file = os.path.join('data', 'VOCdevkit2007', 'VOC2007', 'JPEGImages', image_name[:-1]+'.jpg')
    # 真实图片路径
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)  # 这里边检测完毕了
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    prediction = {}
    prediction['Brace_sleeve'] = []
    prediction['Brace_sleeve_screw'] = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        if cls_ind == 3:
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            dets = dets[inds, :]
            prediction['Brace_sleeve'].append(dets[:, :-1])
        if cls_ind == 11:
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            dets = dets[inds, :]
            prediction['Brace_sleeve_screw'].append(dets[:, :-1])
    return prediction

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args


def GetGtRec(im_name):
    '''
    输入xml文件名 返回真实框的位置
    '''
    filename = os.path.join('data', 'VOCdevkit2007', 'VOC2007', 'Annotations', im_name[:-1] + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    GT_rec = {}
    GT_rec['Brace_sleeve_screw'] = []
    GT_rec['Brace_sleeve'] = []
    for ix, obj in enumerate(objs):
        if obj.find('name').text == 'Brace_sleeve_screw':
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            GT_rec['Brace_sleeve_screw'].append([x1, y1, x2, y2])
            # print(x1, y1, x2, y2)
        if obj.find('name').text == 'Brace_sleeve':
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            GT_rec['Brace_sleeve'].append([x1, y1, x2, y2])
            # print(x1, y1, x2, y2)
    return GT_rec


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 13,
                          tag='default', anchor_scales=[1, 2, 4, 8, 16])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    TrainPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', 'VOCdevkit2007', 'VOC2007',
                             'ImageSets', 'Main', 'train.txt')
    f = open(TrainPath, 'r')
    im_names = f.readlines()
    f.close()
    GT_and_Pre = {}
    GT_and_Pre['GT'] = []
    GT_and_Pre['Pre'] = []
    names = []

    for im_name in im_names:
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        GT_Rec = GetGtRec(im_name)
        if len(GT_Rec['Brace_sleeve_screw']) > 0 and len(GT_Rec['Brace_sleeve']) > 0:
            GT_and_Pre['GT'].append(GT_Rec)
            Pre_Rec = demo(sess, net, im_name)
            GT_and_Pre['Pre'].append(Pre_Rec)
            names.append(im_name) # name 与 gt 一一对应

    with open('./GT_and_Pre.pkl', 'wb') as f:
        pickle.dump(GT_and_Pre, f, pickle.HIGHEST_PROTOCOL)

    with open('./names.pkl', 'wb') as f:
        pickle.dump(names, f, pickle.HIGHEST_PROTOCOL)


