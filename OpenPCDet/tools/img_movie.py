#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:54:11 2022

@author: yagmur

ffmpeg -f image2 -r 10 -i /home/yagmur/Desktop/tracking/%06d.png anim.mp4 
"""

# img_movie.py

import mayavi
from pyface.timer.api import Timer

def animate(src, N=10):
    for j in range(N):
        for i in range(len(src.file_list)):
            src.timestep = i
            yield

if __name__ == '__main__':
    src = mayavi.engine.scenes[0].children[0]
    animator = animate(src)
    t = Timer(250, animator.next)
    
    
