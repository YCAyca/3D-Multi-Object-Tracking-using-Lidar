import mayavi.mlab as mlab
import numpy as np
import torch
import cv2
import os
from pathlib import Path

box_colormap = [
    [1, 1, 1],
    [0, 1, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 0, 0],
    [0.5, 0, 0],
    [0.5, 0.25, 0],
    [0.5, 0.25, 0.15],
    [0.5, 0.25, 0.80],
    [0.5, 0.85, 0.90],
    [0.1, 0.65, 0.15],
    [0.25, 1, 0.50],
    [0.7, 0.85, 0.70],
]

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''

    qs = qs.astype(np.int32)
    
    height, width, channels = image.shape
    
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
     #  if qs[i,0] < width and qs[i,1] < height and qs[j,0] < width and qs[j,1] < height: 
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

       i,j=k+4,(k+1)%4 + 4
       #if qs[i,0] < width and qs[i,1] < height and qs[j,0] < width and qs[j,1] < height: 
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

       i,j=k,k+4
       #if qs[i,0] < width and qs[i,1] < height and qs[j,0] < width and qs[j,1] < height: 
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
    return image

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    
  
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    

    return corners3d.numpy() if is_numpy else corners3d


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(1024, 1024), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure="Scene", bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1024, 1024))

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()), mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_scenes(points, gt_boxes=None, updated_boxes=None,ref_boxes=None, ref_scores=None, ref_labels=None,save_outputs = None, filename=None, track_ids = None,dummy=None, mode=None):
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        try : 
            ref_boxes = ref_boxes.cpu().numpy()
        except:
            ref_boxes = np.array(ref_boxes)    
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        try : 
            ref_scores = ref_scores.cpu().numpy()
        except:
            ref_scores = np.array(ref_scores)
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        try : 
            ref_labels = ref_labels.cpu().numpy()
        except:
            ref_labels = np.array(ref_labels)
    if track_ids is not None and not isinstance(track_ids, np.ndarray):
        try : 
            track_ids = track_ids.cpu().numpy()
        except:
            track_ids = np.array(track_ids)        

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)                                
        if track_ids is not None:
            for k in track_ids:
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                
                mask = (track_ids == k)
               # print("MASK", ref_corners3d[mask])
                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=track_ids[mask], max_num=100,add_id =True)
                if ref_labels is not None:
                    fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_labels[mask], max_num=100,add_label =True)   
                if ref_scores is not None:
                    fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100) 
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = (ref_labels == k)
                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)
   
    
    mlab.view(azimuth=-179, elevation=60.0, distance=150.0, roll=90.0)
    
    
    
    if mode == "pred":
        output_path = "/home/yagmur/Desktop/pred_outputs/"
        frame_name = output_path + filename.split(".")[0] + ".png"
        mlab.savefig(frame_name)     
    elif mode == "save":
        output_path = save_outputs + "/lidar_final"
        if not Path(output_path).exists():
            os.makedirs(output_path)
        frame_name = output_path + "/" + filename.split(".")[0] + ".png"
        mlab.savefig(frame_name)
    elif mode == "show": #show non stop    
        mlab.show(1)
        cv2.imshow('dummy',dummy)
        cv2.waitKey(1)
    else:
        mlab.show(stop=False)
    mlab.clf(figure=fig)
  

def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None, add_id=False, add_label=False):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if add_id:
                if isinstance(cls, np.ndarray):
                    mlab.text3d(b[6, 0], b[6, 1], b[6, 2]+1, 'ID: %d' % cls[n], scale=(0.75, 0.75, 0.75), color=color, figure=fig)
                else:
                    mlab.text3d(b[6, 0], b[6, 1], b[6, 2]+1, 'ID :%d' % cls[n], scale=(0.75, 0.75, 0.75), color=color, figure=fig)
            elif add_label:
                 if isinstance(cls, np.ndarray):
                     mlab.text3d(b[6, 0], b[6, 1], b[6, 2]+2, '%s' % cls[n], scale=(0.75, 0.75, 0.75), color=color, figure=fig)
                 else:
                     mlab.text3d(b[6, 0], b[6, 1], b[6, 2]+2, ':%s' % cls[n], scale=(0.75, 0.75, 0.75), color=color, figure=fig)        
            else:
                if isinstance(cls, np.ndarray):
                    mlab.text3d(b[6, 0], b[6, 1], b[6, 2], 'score %.2f' % cls[n], scale=(0.75, 0.75, 0.75), color=color, figure=fig)
                else:
                    mlab.text3d(b[6, 0], b[6, 1], b[6, 2], 'score %s' % cls[n], scale=(0.75, 0.75, 0.75), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig
