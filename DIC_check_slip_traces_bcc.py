# -*- coding: utf-8 -*-
"""
Produced at Los Alamos National Laboratory.
Written by Duncan Greeley (dgreeley@lanl.gov) and others.
"""

# ---- Import packages ----
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

# =============================================================================
# %% Functions
# =============================================================================

def calcSlipTracesBCC110(bunge_euler,ang_format):
    
    # Rotation corrections to align SEM view and EBSD orientations
    # Default for EDAX Setting 2
    
    # # Rot 1: 90 deg about Z
    rmat_90_z = np.array([[0,-1,0], \
                          [1,0,0], \
                          [0,0,1]])    
        
    # # Rot 2: -180 deg about X
    rmat_neg180_x = np.array([[-1,0,0], \
                              [0,-1,0], \
                              [0,0,1]])

    n_bcc_110_slip = np.array([[0, 0.707106781186548, 0.707106781186548],\
                              [0.707106781186548, 0, 0.707106781186548], \
                              [0.707106781186548, 0.707106781186548, 0], \
                              [0, -0.707106781186548, 0.707106781186548], \
                              [0.707106781186548, 0, -0.707106781186548], \
                              [-0.707106781186548, 0.707106781186548, 0]])
    
    if ang_format == 'degrees':
        bunge_euler = bunge_euler*(np.pi/180)
        
    # Identify the parent orientation in quaternion Crystal -> Lab convention    
    # Bunge Euler default orientation: Lab -> Crystal   
    par_rot_scipy = R.from_euler('ZXZ',bunge_euler,degrees=False) # Crystal -> Lab 
    par_rot_rmat = par_rot_scipy.as_matrix() # Crystal -> Lab
    
    # =========================================================================    
    # ---- BCC {110} Slip Planes ----
    # =========================================================================        
    # Loop through all three prismatic slip systems
    bcc_110_trace_vec = np.zeros((6,3))
    for i in range(6):
        
        slip_norm_crystal = n_bcc_110_slip[i,:]
        slip_norm_lab = np.matmul(par_rot_rmat,slip_norm_crystal)
    
        # Rotate slip norm by 90 about z and 180 about  - alignes EDAX & Lab (image) 
        # reference frames
        slip_norm_lab = np.matmul(rmat_90_z,slip_norm_lab)
        slip_norm_lab = np.matmul(rmat_neg180_x,slip_norm_lab)     
    
        # Project the slip_norm direction into the XY plane
        proj_xy_slip_norm = np.array([slip_norm_lab[0],slip_norm_lab[1],0]) \
            /np.linalg.norm(np.array([slip_norm_lab[0],slip_norm_lab[1],0]))
    
        # Find the orthogonal vector (trace of slip plane)
        bcc_110_trace_vec[i,:] = np.cross(proj_xy_slip_norm,\
                                        np.array([0,0,1])).reshape(1,3)
    # =========================================================================    
    return bcc_110_trace_vec


def calcSlipTracesBCC112(bunge_euler,ang_format):
    
    # Rotation corrections to align SEM view and EBSD orientations
    # Default for EDAX Setting 2
    
    # # Rot 1: 90 deg about Z
    rmat_90_z = np.array([[0,-1,0], \
                          [1,0,0], \
                          [0,0,1]])    
        
    # # Rot 2: -180 deg about X
    rmat_neg180_x = np.array([[-1,0,0], \
                              [0,-1,0], \
                              [0,0,1]])

    n_bcc_112_slip = np.array([[0.8165,   0.4082,    0.4082],\
                              [0.4082,    0.8165,    0.4082], \
                              [0.4082,    0.4082,    0.8165], \
                              [0.4082,    0.8165,   -0.4082], \
                              [-0.4082,    0.4082,    0.8165], \
                              [0.8165,   -0.4082,    0.4082], \
                              [-0.4082,    0.8165,    0.4082], \
                              [0.4082,   -0.4082,    0.8165], \
                              [0.8165,    0.4082,   -0.4082], \
                              [0.8165,   -0.4082,   -0.4082], \
                              [-0.4082,    0.8165,   -0.4082], \
                              [-0.4082,   -0.4082,    0.8165]])
    
    if ang_format == 'degrees':
        bunge_euler = bunge_euler*(np.pi/180)
        
    # Identify the parent orientation in quaternion Crystal -> Lab convention    
    # Bunge Euler default orientation: Lab -> Crystal   
    par_rot_scipy = R.from_euler('ZXZ',bunge_euler,degrees=False) # Crystal -> Lab 
    par_rot_rmat = par_rot_scipy.as_matrix() # Crystal -> Lab
    
    # =========================================================================    
    # ---- BCC {112} Slip Planes ----
    # =========================================================================        
    # Loop through all three prismatic slip systems
    bcc_112_trace_vec = np.zeros((12,3))
    for i in range(12):
        
        slip_norm_crystal = n_bcc_112_slip[i,:]
        slip_norm_lab = np.matmul(par_rot_rmat,slip_norm_crystal)
    
        # Rotate slip norm by 90 about z and 180 about  - alignes EDAX & Lab (image) 
        # reference frames
        slip_norm_lab = np.matmul(rmat_90_z,slip_norm_lab)
        slip_norm_lab = np.matmul(rmat_neg180_x,slip_norm_lab)     
    
        # Project the slip_norm direction into the XY plane
        proj_xy_slip_norm = np.array([slip_norm_lab[0],slip_norm_lab[1],0]) \
            /np.linalg.norm(np.array([slip_norm_lab[0],slip_norm_lab[1],0]))
    
        # Find the orthogonal vector (trace of slip plane)
        bcc_112_trace_vec[i,:] = np.cross(proj_xy_slip_norm,\
                                        np.array([0,0,1])).reshape(1,3)
    # =========================================================================    
    return bcc_112_trace_vec

