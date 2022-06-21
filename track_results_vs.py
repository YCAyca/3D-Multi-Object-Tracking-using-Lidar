#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:11:37 2022

@author: yagmur
"""

import argparse
import os
import cv2
import numpy as np
from OpenPCDet.tools.visual_utils import visualize_utils as V
from kitti_calib import Calibration
import copy

def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--file_path', type=str, default=None,
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=None,
                        help='specify the point cloud data file or directory')
    parser.add_argument('--calib_path', type=str, default=None)
    parser.add_argument('--out_folder', type=str, default=None, help='save demo results to the output folder')
  
    args = parser.parse_args()
    
    imgDir = args.data_path
    outputDir = args.out_folder
    calib = Calibration(args.calib_path)

        
    for frame, imgName in enumerate(os.listdir(imgDir)):
      #  print(imgName)
        frame_name = imgName.split(".")[0]
        if frame_name == "000000":
            frame_name = "0"
        else:    
            frame_name = frame_name.lstrip("0")
        
     #   print("FRAME NAME",frame_name)
        
        img = cv2.imread(imgDir+imgName)
        
        im_2d = copy.deepcopy(img)  
        
        try:
            results = open(args.file_path, 'r')
        except  OSError:
            print('cannot open', args.file_path)
        
              
        for line in results:
           # print(line[0])
           # print(frame_name)
        
            words = line.split(" ")
            if words[0] == frame_name and words[2] != "DontCare" :
                xmin_2d = float(words[6])
                xmax_2d = float(words[7])
                ymin_2d = float(words[8])
                ymax_2d = float(words[9])
                h,w,l = float(words[10]),float(words[11]), float(words[12])
                x,y,z = float(words[13]),float(words[14]),float(words[15])
                teta =  float(words[16].split("\n")[0])
                
                cv2.rectangle(im_2d,(int(xmin_2d), int(xmax_2d)),(int(ymin_2d), int(ymax_2d)),(0,255,0),3)               
                                    
                corners3d_in_cam = compute_3d_box_cam2(h,w,l,x,y,z,teta)
                
                pts_2d = calib.project_rect_to_image(corners3d_in_cam.T)
                
                img = V.draw_projected_box3d(img, pts_2d, color=(255,0,255), thickness=1)
                # if frame_name == "109" or frame_name == "110":
                #    print("fn",frame_name)
                #    print(words[2])
                #    cv2.imshow('image',img)
                #    cv2.waitKey(0)
               

        cv2.destroyAllWindows()
        results.close()
        saveName = imgName.split('.png')[0] +'_result' + '.png'
        cv2.imwrite(outputDir+"/"+saveName, img)
        saveName = imgName.split('.png')[0] +'_result2d' + '.png'
        cv2.imwrite(outputDir+"/"+saveName, im_2d)
                
    
        
       
        
        
if __name__ == '__main__':
    main()
        