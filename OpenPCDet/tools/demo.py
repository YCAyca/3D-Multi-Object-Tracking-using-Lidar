import argparse
import glob
from pathlib import Path
import os

import mayavi.mlab as mlab
from mayavi.mlab import close
from visual_utils import visualize_utils as V 
import math

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset

import sys
sys.path.append("/home/yagmur/lidartracking")

from kitti_calib import Calibration


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
    parser.add_argument('--calib_path', type=str, default=None,
                        help='calibration file path')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--out_folder', type=str, default=None, help='save demo results to the output folder')
    parser.add_argument('--mode', type=str, default="default", help='choose output show mode "save" "show" or "default" ')
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
    
    
    results_folder_path_pedcyc = os.path.join(args.data_path, "ped_cyc_results")
    results_folder_path_car = os.path.join(args.data_path, "car_results")
    
    sequence = args.data_path.split("/")[-1]
   
    calibration_file_path = os.path.join(args.calib_path, sequence + ".txt")
  
    calib = Calibration(calibration_file_path)
    
  
    if not Path(results_folder_path_pedcyc).exists():
        os.makedirs(results_folder_path_pedcyc)
        
    if not Path(results_folder_path_car).exists():
        os.makedirs(results_folder_path_car)
        
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
                
                """ 
                save results in kitti detection format
                
                """
               
                file_path = demo_dataset[idx]["frame_name"].split(".")[0] + ".txt"
                results_path_car = os.path.join(results_folder_path_car, file_path)
                results_car = open(results_path_car, "w")
                
                results_path_pedcyc = os.path.join(results_folder_path_pedcyc, file_path)
                results_pedcyc = open(results_path_pedcyc, "w")
                
                corners3d = V.boxes_to_corners_3d(np.array(pred_dicts[0]['pred_boxes'].cpu())) # [x, y, z, dx, dy, dz, heading]
                   
                
                for i,label in enumerate(pred_dicts[0]['pred_labels']):
                    corner_in_cam = calib.project_velo_to_rect(corners3d[i]) 
                    
                    # 2d box coordinates will be added here 
                    
                    pts_2d = calib.project_rect_to_image(corner_in_cam)
                    
                   
                    
                    # 3D boxes in image plane
                    
                    x_cam = (corner_in_cam[1][0] + corner_in_cam[7][0]) / 2
                    y_cam = 0.9 + (corner_in_cam[1][1] + corner_in_cam[7][1]) / 2
                    z_cam = (corner_in_cam[1][2] + corner_in_cam[7][2]) / 2
                   
                    l_cam = math.sqrt((corner_in_cam[2][0] - corner_in_cam[1][0])**2 + (corner_in_cam[2][1] - corner_in_cam[1][1])**2 + (corner_in_cam[2][2] - corner_in_cam[1][2])**2)
                    w_cam = math.sqrt((corner_in_cam[2][0] - corner_in_cam[3][0])**2 + (corner_in_cam[2][1] - corner_in_cam[3][1])**2 + (corner_in_cam[2][2] - corner_in_cam[3][2])**2)
                    h_cam = math.sqrt((corner_in_cam[2][0] - corner_in_cam[6][0])**2 + (corner_in_cam[2][1] - corner_in_cam[6][1])**2 + (corner_in_cam[2][2] - corner_in_cam[6][2])**2)
       
                    angle = -math.atan2(corner_in_cam[1][2] - corner_in_cam[2][2], corner_in_cam[1][0] - corner_in_cam[2][0])
       
                    if label == 1:
                        results_car.write("Car ")
                        results_car.write("-1 -1 0 ")
                        results_car.write(str(pts_2d[7][0]) + " " + str(pts_2d[7][1]) + " " + str(pts_2d[1][0]) + " " + str(pts_2d[1][1])  + " ")
                        results_car.write(str(h_cam) + " ")
                        results_car.write(str(w_cam) + " ")
                        results_car.write(str(l_cam) + " ")
                        results_car.write(str(x_cam) + " ")
                        results_car.write(str(y_cam) + " ")
                        results_car.write(str(z_cam) + " ")
                        results_car.write(str(angle) + " ")
                        results_car.write(str(float(pred_dicts[0]['pred_scores'][i])) + "\n")
                    elif label == 2:
                        results_pedcyc.write("Pedestrian ")
                        results_pedcyc.write("-1 -1 0 ")
                        results_pedcyc.write(str(pts_2d[7][0]) + " " + str(pts_2d[7][1]) + " " + str(pts_2d[1][0]) + " " + str(pts_2d[1][1])  + " ")
                        results_pedcyc.write(str(h_cam) + " ")
                        results_pedcyc.write(str(w_cam) + " ")
                        results_pedcyc.write(str(l_cam) + " ")
                        results_pedcyc.write(str(x_cam) + " ")
                        results_pedcyc.write(str(y_cam) + " ")
                        results_pedcyc.write(str(z_cam) + " ")
                        results_pedcyc.write(str(angle) + " ")
                        results_pedcyc.write(str(float(pred_dicts[0]['pred_scores'][i])) + "\n")
                    elif label == 3:
                        results_pedcyc.write("Cyclist ")
                        results_pedcyc.write("-1 -1 0 ")
                        results_pedcyc.write(str(pts_2d[7][0]) + " " + str(pts_2d[7][1]) + " " + str(pts_2d[1][0]) + " " + str(pts_2d[1][1])  + " ")
                        results_pedcyc.write(str(h_cam) + " ")
                        results_pedcyc.write(str(w_cam) + " ")
                        results_pedcyc.write(str(l_cam) + " ")
                        results_pedcyc.write(str(x_cam) + " ")
                        results_pedcyc.write(str(y_cam) + " ")
                        results_pedcyc.write(str(z_cam) + " ")
                        results_pedcyc.write(str(angle) + " ")
                        results_pedcyc.write(str(float(pred_dicts[0]['pred_scores'][i])) + "\n")
                
                  
                    
                results_car.close()  
                results_pedcyc.close()
                    


    logger.info('Demo done.')


if __name__ == '__main__':
     main()
