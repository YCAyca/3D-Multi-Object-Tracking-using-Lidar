#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:04:53 2022

@author: yagmur
"""

import argparse
import glob
from pathlib import Path
import os

import mayavi.mlab as mlab
from visual_utils import visualize_utils as V  

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='cfgs/pandaset_models/pv_rcnn.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='inference data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--show_gt', action='store_true',help='show ground truth boxes')
    parser.add_argument('--out_folder', type=str, default=None, help='save demo results to the output folder')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Pandaset Inference & Visualizer -------------------------')
    demo_dataset = PandasetDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger
    )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
 
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info('Visualized sample name %s: \t' , demo_dataset[idx]["frame_idx"])
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
        #    print("PRED BOXES",pred_dicts[0]['pred_boxes'])
            
        #    print("PRED BOXES",pred_dicts[0]['pred_labels'])
            
        #    print("GT BOXES",data_dict['gt_boxes'][0,:,:7])
            
            
                
            # Draw Inference boxes
            if args.show_gt:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0,:,:7],ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                    save_outputs = args.out_folder, filename=demo_dataset[idx]["frame_idx"]
                )
            else :
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], gt_boxes=None,ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                    save_outputs = args.out_folder, filename=demo_dataset[idx]["frame_idx"]
                )
                
              

            mlab.show(stop=False)

    logger.info('Inference done.')


if __name__ == '__main__':
    main()
