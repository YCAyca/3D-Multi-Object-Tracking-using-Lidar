import argparse
import glob
from pathlib import Path
import os

import mayavi.mlab as mlab
from mayavi.mlab import close
from visual_utils import visualize_utils as V 

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.dataset_cfg.DATASET == 'LyftDataset' or self.dataset_cfg.DATASET == 'NuScenesDataset':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5) #[:, :4]
        elif self.dataset_cfg.DATASET == 'KittiDataset' or self.dataset_cfg.DATASET == 'KITWARE':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
            'frame_name' : self.sample_file_list[index].split('/')[-1],
        }
        
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--out_folder', type=str, default=None, help='save demo results to the output folder')
    parser.add_argument('--create_annotation', type=bool, default=None, help='save demo results as kitti annotation files')
    parser.add_argument('--mode', type=str, default=None, help='choose output show mode "save" "show" or "default" ')
    parser.add_argument('--show_gt', action='store_true',help='show ground truth boxes')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    
    if cfg.DATA_CONFIG.DATASET ==  'PandasetDataset':
        demo_dataset = PandasetDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), logger=logger
        )
    else:
        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), ext=args.ext, logger=logger
        )
    
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    dummy = np.zeros((1,1,1),dtype=np.uint8)
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            if cfg.DATA_CONFIG.DATASET ==  'PandasetDataset':
                logger.info('Visualized sample name %s: \t' , demo_dataset[idx]["frame_idx"])
                if args.show_gt: # for pandaset. kitti doesnt have ground truths on test dataset
                    V.draw_scenes(
                        points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0,:,:7],ref_boxes=pred_dicts[0]['pred_boxes'],
                        ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                        save_outputs = args.out_folder, filename=str(demo_dataset[idx]["frame_idx"]), dummy = dummy, mode=args.mode, 
                    )
                else:
                    V.draw_scenes(
                        points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                        ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                        save_outputs = args.out_folder, filename=str(demo_dataset[idx]["frame_idx"]), dummy = dummy, mode=args.mode, 
                    )
            else:    
                logger.info('Visualized sample name %s: \t' , demo_dataset[idx]["frame_name"])
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                    save_outputs = args.out_folder, filename=demo_dataset[idx]["frame_name"], dummy = dummy, mode=args.mode, 
                )

                  

    logger.info('Demo done.')


if __name__ == '__main__':
     main()
