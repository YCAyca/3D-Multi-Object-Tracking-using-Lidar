#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:34:09 2022

@author: yagmur
"""

import argparse
import glob
from pathlib import Path
import os
import os.path, copy, numpy as np, time, sys

import mayavi.mlab as mlab
from OpenPCDet.tools.visual_utils import visualize_utils as V
from numba import jit
from pyquaternion import Quaternion
from filterpy.kalman import KalmanFilter
from scipy.spatial import ConvexHull
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from dataclasses import dataclass
import math
from kitti_oxts import get_ego_traj, egomotion_compensation_ID, load_oxts, roty,rotz,rotx
from kitti_calib import Calibration
import cv2
from scipy.optimize import linear_sum_assignment as linear_assignment



TRACKING_CLASSES = [
  'Car',
  'Pedestrian',
  'Cyclist',
]


POINT_CLOUD_X = 70
POINT_CLOUD_Y = 40

@jit    
def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

@jit        
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

@jit       
def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0 

def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    '''
    # corner points are in counter clockwise order
    
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def convert_3dbox_to_8corner(bbox3d_input):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
        
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    R = roty(bbox3d[3])    

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + bbox3d[0]
    corners_3d[1,:] = corners_3d[1,:] + bbox3d[1]
    corners_3d[2,:] = corners_3d[2,:] + bbox3d[2]
 
    return np.transpose(corners_3d)


kalman_logger = open("kalman_logger.txt", "w+")

class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  global kalman_logger
  def __init__(self, bbox3D, info, covariance_id=0, track_score=None, tracking_name='car', constant_vel = True):
    """
    Initialises a tracker using initial bounding box.
    """
    
    if constant_vel:
            
        """              
        observation: 
          [x, y, z, rot_y, l, w, h]
        state:
          [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot, rot_y_dot]
        """
    
        #define constant velocity model with angular velocity
        
        self.kf = KalmanFilter(dim_x=11, dim_z=7)    
                            #  x y z a l w h dxdydzda 
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],  # x    # state transition matrix
                              [0,1,0,0,0,0,0,0,1,0,0],  # y
                              [0,0,1,0,0,0,0,0,0,1,0],  # z
                              [0,0,0,1,0,0,0,0,0,0,1],  # a
                              [0,0,0,0,1,0,0,0,0,0,0],  # l
                              [0,0,0,0,0,1,0,0,0,0,0],  # w
                              [0,0,0,0,0,0,1,0,0,0,0],  # h
                              [0,0,0,0,0,0,0,1,0,0,0],  # dx
                              [0,0,0,0,0,0,0,0,1,0,0],  # dy
                              [0,0,0,0,0,0,0,0,0,1,0],  # dz
                              [0,0,0,0,0,0,0,0,0,0,1]]) # da     
       
        
                            #  x y z a l w h dxdydzda   
        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],  #  x  # measurement function,
                              [0,1,0,0,0,0,0,0,0,0,0],  #  y
                              [0,0,1,0,0,0,0,0,0,0,0],  #  z
                              [0,0,0,1,0,0,0,0,0,0,0],  #  a
                              [0,0,0,0,1,0,0,0,0,0,0],  #  l
                              [0,0,0,0,0,1,0,0,0,0,0],  #  w
                              [0,0,0,0,0,0,1,0,0,0,0]]) #  h

        # Initialize the covariance matrix    
        self.kf.R[0:,0:] *= 10.   # measurement uncertainty 
        self.kf.P[7:,7:] *= 1000. # state uncertainty, 
        self.kf.x[:7] = bbox3D.reshape((7, 1))

    

    
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 1          # number of total hits including the first detection
    self.hit_streak = 1     # number of continuing hit considering the first detection
    self.first_continuing_hit = 1
    self.still_first = True
    self.age = 0
    self.info = info        # other info
    self.track_score = track_score
    self.tracking_name = tracking_name
    self.mean_speed = 0
    self.compensated = False    
        
    kalman_logger.write("INITIAL KALMAN FILTER VALUES tracker id :" + str(self.id + 1) + "\nKF X \n" + str(self.kf.x) + "\nKF F \n" + str(self.kf.F) + "\nKF H \n" + str(self.kf.H) + "\nKF R \n" + str(self.kf.R) + "\nKF P \n" + str(self.kf.P) + "\nKF Q \n"+ str(self.kf.Q))


  def update(self, bbox3D, info, oxts,class_name,frame): 
    """ 
    Updates the state vector with observed bbox.
    """
    tmp_time = self.time_since_update
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1          # number of continuing hit
    if self.still_first:
      self.first_continuing_hit += 1      # number of continuing hit in the fist time
    
    ######################### orientation correction
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    new_theta = bbox3D[3]
    if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
    if new_theta < -np.pi: new_theta += np.pi * 2
    bbox3D[3] = new_theta

    predicted_theta = self.kf.x[3]
    if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
      self.kf.x[3] += np.pi       
      if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
      if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
      
    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
      if new_theta > 0: self.kf.x[3] += np.pi * 2
      else: self.kf.x[3] -= np.pi * 2
    
    ######################### 

    bbox3D = bbox3D.reshape(7,1)
    kalman_logger.write("\n PREDICTED 3D BBOX \n" + str(bbox3D))
    
    tmp_P = self.kf.P
    tmp_x = self.kf.x
    self.kf.update(bbox3D)
    
    """ match deny mechanism begin"""
    
    vf = oxts[0][frame-1]
    vl = oxts[1][frame-1]
    vz = oxts[2][frame-1]
      
    pose_x = (vf / 100)
    pose_y = (vl / 100)
    vz = (vz / 100)
    
    dist = math.sqrt((self.kf.x[0] + pose_x -  tmp_x[0])**2 + (self.kf.x[1] + pose_y - tmp_x[1])**2) #distance according to absolute speed 
 
   
    if class_name == "Car":
        if dist > 4.5: #162 km/h
            self.kf.x = tmp_x
            self.kf.P = tmp_P
            self.hits -= 1
            self.time_since_update = tmp_time
            
    """ match deny mechanism end"""

    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
    self.info = info
    
    
    kalman_logger.write("\n UPDATED KALMAN FILTER VALUES tracker id: " + str(self.id + 1) + "\nKF X \n" + str(self.kf.x) +  "\nKF P \n" + str(self.kf.P) +"\n")


  def predict(self):       
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
       
    self.kf.predict() 
    kalman_logger.write("\n PREDICTED KALMAN FILTER VALUES tracker id: " + str(self.id + 1) + "\nKF X \n" + str(self.kf.x) +  "\nKF P \n" + str(self.kf.P) +"\n" + "\nKF R \n" + str(self.kf.R) +"\n" + "\nKF K \n" + str(self.kf.K) +"\n")
    
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
      self.still_first = False
    self.time_since_update += 1
    self.history.append(self.kf.x)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf.x[:7].reshape((7, ))


class MOT(object):
  def __init__(self,covariance_id=0, calib=None,oxts=None,max_age=5,min_hits=3, tracking_name='car'):
   
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.covariance_id = covariance_id
    self.tracking_name = tracking_name
    self.calib = calib
    self.oxts = oxts
  
  def ego_motion_compensation(self,frame, trks):
      
      """ applying ego motion compensation using lidar car motion and KITTI oxts data for extracting the car motion in x and y direction """
      
      vf = self.oxts[0][frame-1]
      vl = self.oxts[1][frame-1]
      vz = self.oxts[2][frame-1]
      
      pose_x = -(vf / 100)
      pose_y = -(vl / 100)
      vz = -(vz / 100)
      
      for trk in trks:
          rot_mat = rotz(vz)
          rotated = np.dot(rot_mat, [[float(trk.kf.x[0])], [float(trk.kf.x[1])]])
          trk.kf.x[0] = rotated[0] + pose_x
          trk.kf.x[1] = rotated[1] + pose_y
          
             
      return trks
  
  def update(self,dets_all, match_distance, match_threshold, match_algorithm, tracking_log=None, points=None, filename=None):
    """
    Params:
      dets_all: dict
        dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
        info: a array of other info for each det
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
    
  
    trks = np.zeros((len(self.trackers),7))         # N x 7 , #get predicted locations from existing trackers.
    boxes = np.zeros((len(self.trackers),7))  
    boxes_compensated = np.zeros((len(self.trackers),7))  
    to_del = []
    ret = []
    ids = []
      
    pos_predictions = np.zeros((len(dets),7)) 
    for i,det in enumerate(dets):
        pos_predictions[i][:] = [det[0], det[1], det[2], det[4], det[5], det[6],det[3]]
      
    for t,trk in enumerate(trks):
        pos = self.trackers[t].predict().reshape((-1, 1))
        
        boxes[t][:] = [pos[0], pos[1], pos[2], pos[4], pos[5], pos[6],pos[3]] 
        if (self.frame_count > 0) and (self.oxts is not None):
            self.trackers = self.ego_motion_compensation(self.frame_count, self.trackers)
      
                 
        trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]  
        
        boxes_compensated[t][:] = [self.trackers[t].kf.x[0],  self.trackers[t].kf.x[1],  self.trackers[t].kf.x[2],  self.trackers[t].kf.x[4],  self.trackers[t].kf.x[5],  self.trackers[t].kf.x[6], self.trackers[t].kf.x[3]]
        ids.append(self.trackers[t].id+1)
                 
        if(np.any(np.isnan(pos))):
          to_del.append(t)
     
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))   
    for t in reversed(to_del):
        self.trackers.pop(t)
       
      
    # predicted tracker locations with detection predictions
    if self.tracking_name == "Car":
        V.draw_scenes(
                  points=points, ref_boxes=boxes,gt_boxes=pos_predictions,
                  track_ids = ids, filename=filename,mode="pred", 
            )
  
    dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
    if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
    else: dets_8corner = []

    trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
    trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in self.trackers]

    if len(trks_8corner) > 0: 
      trks_8corner = np.stack(trks_8corner, axis=0)
      trks_S = np.stack(trks_S, axis=0)
    if match_distance == 'iou':
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, iou_threshold=match_threshold, match_algorithm=match_algorithm,tracking_log=tracking_log)
    else:
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, use_mahalanobis=True, dets=dets, trks=trks, trks_S=trks_S, mahalanobis_threshold=match_threshold, match_algorithm=match_algorithm,tracking_log=tracking_log)
   
    #update matched trackers with assigned detections
    
    if self.tracking_name == 'Car':
        for t,trk in enumerate(self.trackers):
          
          if t not in unmatched_trks:      
            d = matched[np.where(matched[:,1]==t)[0],0]     # a list of index
            detection_score = info[d, :][0][-1]  
            trk.update(dets[d,:][0], info[d, :][0],self.oxts,self.tracking_name,self.frame_count)     
            trk.track_score = detection_score
            tracking_log.write("\n matched tracker:" + str(trk.id+1) + " dist mat index " + str(d))
    else:
        for t,trk in enumerate(self.trackers):
          if t not in unmatched_trks:
            d = matched[np.where(matched[:,1]==t)[0],0]     # a list of index
            detection_score = info[d, :][0][-1]
            
            tracking_log.write("\n matched tracker:" + str(trk.id+1) + " dist mat index " + str(d))       
            if detection_score > 0.65:
                trk.update(dets[d,:][0], info[d, :][0],self.oxts,self.tracking_name,self.frame_count)
                trk.track_score = detection_score
                tracking_log.write("\n matched tracker:" + str(trk.id+1) + " dist mat index " + str(d))
            else:    
                  tracking_log.write("\n matched but low precision score not uptated tracker:" + str(trk.id+1) + " dist mat index " + str(d))
          else:
             tracking_log.write("\n unmatched tracker:" + str(trk.id+1))
             

    #create and initialise new trackers for unmatched detections
    
    overlapping_new_track = False
    for i in unmatched_dets:        # a scalar of index
        for t,trk_corner in enumerate(trks_8corner): 
            newtrack_location = convert_3dbox_to_8corner(dets[i,:])   
            iou = iou3d(newtrack_location,trk_corner)
            if iou[0] > 0.3:
                overlapping_new_track = True
                break
        if overlapping_new_track:
            overlapping_new_track = False
            continue # dont create a tracker its overlapping
    
        detection_score = info[i][-1]
        track_score = detection_score
        if self.tracking_name == 'Car':
            trk = KalmanBoxTracker(dets[i,:], info[i, :], self.covariance_id, track_score, self.tracking_name) 
            tracking_log.write("\n new tracker for unmatched det:"+ str(trk.id+1) + "\n")
            print("new tracker :", str(trk.id+1))
            self.trackers.append(trk)
        elif detection_score >= 0.65:    
            trk = KalmanBoxTracker(dets[i,:], info[i, :], self.covariance_id, track_score, self.tracking_name) 
            tracking_log.write("\n new tracker for unmatched det:"+ str(trk.id+1) + "\n")
            self.trackers.append(trk)
        
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()      # bbox location
        

        if((trk.time_since_update < self.max_age) and (self.frame_count < self.min_hits + 2 or trk.hits >= self.min_hits)):        
          ret.append(np.concatenate((d, [trk.id+1], trk.info[:-1], [trk.track_score])).reshape(1,-1)) # +1 as MOT benchmark requires positive
      
        i -= 1
        
        # remove dead tracklet
        
        if(trk.time_since_update >= self.max_age + 5):
          tracking_log.write("\n removed tracker :"+ str(trk.id+1))
          self.trackers.pop(i)
        elif((d[0] >= POINT_CLOUD_X or d[0] < 0 or abs(d[1]) >= POINT_CLOUD_Y)):
          tracking_log.write("\n removed tracker :"+ str(trk.id+1))
          self.trackers.pop(i)  
          
    self.frame_count += 1
    
    if(len(ret)>0):
      return np.concatenate(ret)      # x, y, z, theta, l, w, h, ID, other info, confidence
    return np.empty((0,15 + 7))  

def angle_in_range(angle):
  '''
  Input angle: -2pi ~ 2pi
  Output angle: -pi ~ pi
  '''
  if angle > np.pi:
    angle -= 2 * np.pi
  if angle < -np.pi:
    angle += 2 * np.pi
  return angle

def diff_orientation_correction(det, trk):
  '''
  return the angle diff = det - trk
  if angle diff > 90 or < -90, rotate trk and update the angle diff
  '''
  diff = det - trk
  diff = angle_in_range(diff)
  if diff > np.pi / 2:
    diff -= np.pi
  if diff < -np.pi / 2:
    diff += np.pi
  diff = angle_in_range(diff)
  return diff



def greedy_match(distance_matrix):
  '''
  Find the one-to-one matching using greedy allgorithm choosing small distance
  distance_matrix: (num_detections, num_tracks)
  '''
    
  matched_indices = []

  num_detections, num_tracks = distance_matrix.shape
  distance_1d = distance_matrix.reshape(-1)
  index_1d = np.argsort(distance_1d)
  index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
  detection_id_matches_to_tracking_id = [-1] * num_detections
  tracking_id_matches_to_detection_id = [-1] * num_tracks
  for sort_i in range(index_2d.shape[0]):
    detection_id = int(index_2d[sort_i][0])
    tracking_id = int(index_2d[sort_i][1])
    if detection_id_matches_to_tracking_id[detection_id] == -1:
        if tracking_id_matches_to_detection_id[tracking_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id]) 
        else: # optimized greedy match
          distance = 0
          for i,pair in enumerate(matched_indices):
              if tracking_id == pair[1]:
                  index = i
                  distance = distance_matrix[pair[0]][tracking_id]
                  break
          if distance_matrix[detection_id][tracking_id] < distance:
              matched_indices.append([detection_id, tracking_id]) 
              matched_indices.pop(index)
              # print("match corrected")

  matched_indices = np.array(matched_indices)
  return matched_indices



def associate_detections_to_trackers(detections,trackers,iou_threshold=0.25, use_mahalanobis=False, dets=None, trks=None, trks_S=None, mahalanobis_threshold=0.15, match_algorithm='h',tracking_log=None):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  detections:  N x 8 x 3
  trackers:    M x 8 x 3
  dets: N x 7
  trks: M x 7
  trks_S: N x 7 x 7
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8,3),dtype=int)    
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
  distance_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  if use_mahalanobis:
    assert(dets is not None)
    assert(trks is not None)
    assert(trks_S is not None)

  if use_mahalanobis :
    S_inv = [np.linalg.inv(S_tmp) for S_tmp in trks_S]  # 7 x 7
    S_inv_diag = [S_inv_tmp.diagonal() for S_inv_tmp in S_inv]# 7
   
  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      if use_mahalanobis:
        S_inv = np.linalg.inv(trks_S[t]) # 7 x 7
        diff = np.expand_dims(dets[d] - trks[t], axis=1) # 7 x 1
        # manual reversed angle by 180 when diff > 90 or < -90 degree
        corrected_angle_diff = diff_orientation_correction(dets[d][3], trks[t][3])
        diff[3] = corrected_angle_diff
        distance_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])
      else:
        iou_matrix[d,t] = iou3d(det,trk)[0]             # det: 8 x 3, trk: 8 x 3
        distance_matrix = -iou_matrix

  if match_algorithm == 'greedy':
    matched_indices = greedy_match(distance_matrix)
  elif match_algorithm == 'pre_threshold':
    if use_mahalanobis:
      to_max_mask = distance_matrix > mahalanobis_threshold
      distance_matrix[to_max_mask] = mahalanobis_threshold + 1
    else:
      to_max_mask = iou_matrix < iou_threshold
      distance_matrix[to_max_mask] = 0
      iou_matrix[to_max_mask] = 0
    matched_indices = linear_assignment(distance_matrix)      # houngarian algorithm
  else:
    matched_indices = []
    matched_indices_tmp = linear_assignment(distance_matrix)      # houngarian algorithm
    length = len(matched_indices_tmp[0])
    for i in range(length): 
        matched_indices.append([matched_indices_tmp[0][i], matched_indices_tmp[1][i]])
    matched_indices = np.array(matched_indices)
    #print(matched_indices[:,0])

  tracking_log.write('\n distance_matrix.shape: ' + str(distance_matrix.shape) + "\n")
  tracking_log.write('\n distance_matrix: ' + str(distance_matrix) + "\n")
  tracking_log.write('\n matched_indices: ' + str(matched_indices)+ "\n")

  unmatched_detections = []

  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if len(matched_indices) == 0 or (t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    match = True
    if use_mahalanobis:
      if distance_matrix[m[0],m[1]] > mahalanobis_threshold:
        match = False
    else:
      if(iou_matrix[m[0],m[1]]<iou_threshold):
        match = False
    if not match:
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)


  tracking_log.write('matches: '+ str(matches) + "\n")
  tracking_log.write('unmatched_detections: ' + str(unmatched_detections) + "\n")
  tracking_log.write('unmatched_trackers: '+ str(unmatched_trackers) + "\n")
    
   
  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


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
        elif self.dataset_cfg.DATASET == 'KittiDataset':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.dataset_cfg.DATASET ==  'PandasetDataset':
             points = np.load(self.sample_file_list[index])
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
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',help='specify the config for detection')
    parser.add_argument('--base_path', type=str, default=None, help='specify directory includes lidar, image, calib, label folders. If only lidar is giving, only track results on lidar space will be calculated')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--out_folder', type=str, default=None, help='save results to the output folder')
    parser.add_argument('--mode', type=str, default=None, help='choose output show mode "save" or "show" ')
    parser.add_argument('--visualize', action='store_true',help='show ground truth boxes')
    parser.add_argument('--ext', type=str, default=".bin")
    

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('----------------- Tracking Starts-------------------------')

    basedir = args.base_path
    calib_dir = os.path.join(basedir, "calib")
    oxts_dir = os.path.join(basedir, "oxts")
    img_dir = os.path.join(basedir, "image_02")
    label_dir = os.path.join(basedir, "label_02")
    velodyne_dir = os.path.join(basedir, "velodyne")
    output_dir = os.path.join(basedir, "outputs")
    
    if not Path(output_dir).exists():
        os.makedirs(output_dir)
    
    calib = Calibration("/home/yagmur/Desktop/data/kitti_tracking/test/calib/0000.txt")
    
    start_time = time.time()
    total_frames = 0
   
    """ Apply tracking for each sequence one by one """
   
    for sequence in os.listdir(velodyne_dir):
        print("Tracking begins for the sequence: ",sequence)
        output_seq_dir = os.path.join(output_dir, sequence)
        if not Path(output_seq_dir).exists():
            os.makedirs(output_seq_dir)
        
        dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(velodyne_dir+"/"+sequence), ext=args.ext, logger=logger
        )
    
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
    
        logger.info(f'Total number of samples: \t{len(dataset)}')

    
        oxts_file = oxts_dir + "/"+ sequence + ".txt"
        
        if Path(oxts_file).exists(): # ego motion compensation will be applied if oxts data is given
            vf,vl,vz = load_oxts(oxts_file) 
            mot_trackers = {tracking_name: MOT(tracking_name=tracking_name, calib=calib,oxts=(vf,vl,vz)) for tracking_name in TRACKING_CLASSES}
        else:   
            mot_trackers = {tracking_name: MOT(tracking_name=tracking_name, calib=calib,oxts=None) for tracking_name in TRACKING_CLASSES}

  
        
        dummy = np.zeros((1,1,1),dtype=np.uint8) # a unimportant variable to fix a bug in visualization
        
        global kalman_logger
        
        
        track_results = output_seq_dir + "/" + sequence + ".txt"
        tracking_log = output_seq_dir + "/" + sequence + "_log.txt"
    
        results = open(track_results, "w")
        track_log = open(tracking_log, "w")
    
        with torch.no_grad():
            """ 
             for each sequence, process the frames 1  by 1 in object detection - kalman filter - 
             mahalanobis distance - greedy algorithm - match deny mechanism - visualize final result order
             
            """
            for idx, data_dict in enumerate(dataset):       
                logger.info('Visualized sample name %s: \t' , dataset[idx]["frame_name"])
                data_dict = dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)
            
                imgName = dataset[idx]["frame_name"].split(".")[0] + ".png"
                img_path = os.path.join(img_dir, sequence, imgName)
                img = cv2.imread(img_path)
            
                track_log.write("FRAME "+ str(dataset[idx]["frame_name"])+"\n")
                kalman_logger.write("FRAME "+ str(dataset[idx]["frame_name"])+"\n")
            
                dets = {tracking_name: [] for tracking_name in TRACKING_CLASSES}
                info = {tracking_name: [] for tracking_name in TRACKING_CLASSES}
            
                detection_count = len(pred_dicts[0]['pred_labels'])
            
           
                for i in range(detection_count):
                    box_detection_name = TRACKING_CLASSES[pred_dicts[0]['pred_labels'][i]-1]
                    box_detection_score = pred_dicts[0]['pred_scores'][i].cpu()
                  
                    prediction = pred_dicts[0]['pred_boxes'][i].cpu().numpy()
                    #[x,y,z,theta,l,w,h]
                    detection = np.array([
                        prediction[0], prediction[1], prediction[2], 
                        prediction[6], prediction[3], prediction[4],
                        prediction[5]])
              
                    information = np.array([box_detection_score])
                    dets[box_detection_name].append(detection)
                    info[box_detection_name].append(information)
               
                dets_all = {tracking_name: {'dets': np.array(dets[tracking_name]), 'info': np.array(info[tracking_name])} for tracking_name in TRACKING_CLASSES}
        
                total_frames += 1
                all_boxes = []
                all_ids = []
                all_labels = []
                all_scores = []
                
                for tracking_name in TRACKING_CLASSES:
                    """ 
                    update trackers using detections. From mot_tracker[].update function 
                    we pass to Kalman - Mahalanobis - Greedy algo steps
                    """
                    trackers = mot_trackers[tracking_name].update(dets_all[tracking_name], 'm', 0.4, match_algorithm ='greedy',tracking_log=track_log, points=data_dict['points'][:, 1:], filename=dataset[idx]["frame_name"]) #0.1
                  
                    """
                    Arrange all the output trackers for visualizing function
                    """
                    
                    boxes_x = trackers[:,0] #[x,y,z,theta,l,w,h]
                    boxes_y = trackers[:,1]
                    boxes_z = trackers[:,2]
                    boxes_theta = trackers[:,3]
                    boxes_l = trackers[:,4]
                    boxes_w = trackers[:,5]
                    boxes_h = trackers[:,6]
                          
                    boxes = []
           
            
                    for i in range(len(boxes_x)):
                      boxes.append([boxes_x[i], boxes_y[i], boxes_z[i], boxes_l[i], boxes_w[i], boxes_h[i],boxes_theta[i]])
                  
          
                    ids = [int(x) for x in trackers[:,7]] 
                    scores = list(trackers[:,8])
                             
                    labels = [tracking_name] * len(ids)
            
                
                    all_boxes += copy.deepcopy(boxes)
                    all_ids += copy.deepcopy(ids)
                    all_labels += copy.deepcopy(labels)
                    all_scores += copy.deepcopy(scores)
         
                      
                    calib_file = Path(calib_dir+"/"+sequence+".txt")
                     
                    """
                    
                    pass from lidar to image frame to be able to apply evaluation after saving them in KITTI format
                    
                    """
                    if boxes != [] and calib_file.exists(): 
                    
                        calib = Calibration(calib_file)         
                        corners3d = V.boxes_to_corners_3d(np.array(boxes)) # [x, y, z, dx, dy, dz, heading]
                   
                        im_2d = copy.deepcopy(img)  
                   
                        
                        for i,corner in enumerate(corners3d):
                              corner_in_cam = calib.project_velo_to_rect(corner) 
                              pts_2d = calib.project_rect_to_image(corner_in_cam)
                              img_height, img_width,channels = img.shape
                             
                              cv2.rectangle(im_2d,(int(pts_2d[7][0]), int(pts_2d[7][1])),(int(pts_2d[1][0]), int(pts_2d[1][1])),(0,255,0),3)               
                         
                              x_cam = (corner_in_cam[1][0] + corner_in_cam[7][0]) / 2
                              y_cam = 0.9 + (corner_in_cam[1][1] + corner_in_cam[7][1]) / 2
                              z_cam = (corner_in_cam[1][2] + corner_in_cam[7][2]) / 2
                             
                              l_cam = math.sqrt((corner_in_cam[2][0] - corner_in_cam[1][0])**2 + (corner_in_cam[2][1] - corner_in_cam[1][1])**2 + (corner_in_cam[2][2] - corner_in_cam[1][2])**2)
                              w_cam = math.sqrt((corner_in_cam[2][0] - corner_in_cam[3][0])**2 + (corner_in_cam[2][1] - corner_in_cam[3][1])**2 + (corner_in_cam[2][2] - corner_in_cam[3][2])**2)
                              h_cam = math.sqrt((corner_in_cam[2][0] - corner_in_cam[6][0])**2 + (corner_in_cam[2][1] - corner_in_cam[6][1])**2 + (corner_in_cam[2][2] - corner_in_cam[6][2])**2)
                 
                              angle = -math.atan2(corner_in_cam[1][2] - corner_in_cam[2][2], corner_in_cam[1][0] - corner_in_cam[2][0])
                 
                              results.write(str(idx) + " " + str(ids[i]) + " " + labels[i] + " -1 " + "3 " + "0 " + "0 "  + "0 "  + "0 " + "0 "  + str(h_cam) + " " + str(w_cam) + " " + str(l_cam) + " " + str(x_cam) + " " + str(y_cam) + " " + str(z_cam) + " " + str(angle) + " " + str(scores[i]) + "\n")
          
                """ 
                
                save the results in image space with 2D and 3D bounding boxes.
                This may help  to see if the results are good since the evaluation code uses 2D bounding boxes in somehow even 
                for 3D tracking results
                
                """
          
                saveName = imgName.split('.png')[0] +'_result' + '.png'
                save_dir = os.path.join(output_seq_dir, "direct_output")
                if not Path(save_dir).exists():
                    os.makedirs(save_dir)
                cv2.imwrite(save_dir+"/"+saveName, img)
                saveName = imgName.split('.png')[0] +'_result2d' + '.png'
                cv2.imwrite(save_dir+"/"+saveName, im_2d)
                
                """ visualize the results """       
            
                if args.visualize:
                    V.draw_scenes(
                        points=data_dict['points'][:, 1:], ref_boxes=all_boxes,
                        ref_scores=all_scores, ref_labels=all_labels,
                        save_outputs = output_seq_dir, track_ids = all_ids, filename=dataset[idx]["frame_name"],dummy = dummy, mode=args.mode, 
                        )
        
                    
        track_log.close()    
        results.close()
        kalman_logger.close()

    print("total time:", time.time() - start_time)
    print("total frame", total_frames)
    print("track time per frame", (time.time() - start_time) / total_frames)
        
if __name__ == '__main__':
    main()
