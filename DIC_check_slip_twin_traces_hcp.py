# -*- coding: utf-8 -*-
"""
Produced at Los Alamos National Laboratory.
Written by Duncan Greeley (dgreeley@lanl.gov) and others.
"""

# ---- Import packages ----
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path

# =============================================================================
# %% Functions
# =============================================================================


HCP_MILLER_INDEX_DIR = Path(__file__).with_name("hcp_slip_twin_miller_indices")


def load_hcp_miller_indices(filename):
    path = HCP_MILLER_INDEX_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required HCP Miller-index table is missing: {path}"
        )
    return np.loadtxt(path, delimiter=",")


def calcSlipTracesHCPBasal(bunge_euler,ang_format,ebsd_convention):
    
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

        
    # Prismatic slip plane normals, assuming <a> || X
    n_pslip = np.array([[0.866025403784, 0.50, 0.00],\
                        [0.00, 1.00, 0.00],\
                        [-0.866025403784, 0.50, 0.00]])         
        
    # Basal slip plane normals
    n_bslip = np.array([0.00, 0.00, 1.00])           
    
    if ang_format == 'degrees':
        bunge_euler = bunge_euler*(np.pi/180)
        
    # Identify the parent orientation in quaternion Crystal -> Lab convention    
    # Bunge Euler default orientation: Lab -> Crystal   
    par_rot_scipy = R.from_euler('ZXZ',bunge_euler,degrees=False) # Crystal -> Lab 
    par_rot_rmat = par_rot_scipy.as_matrix() # Crystal -> Lab
  

    # =========================================================================    
    # ---- Basal Slip Mode ----
    # =========================================================================          
    # Check the basal slip plane   
    slip_norm_crystal = n_bslip
    slip_norm_lab = np.matmul(par_rot_rmat,slip_norm_crystal)

    # Rotate slip norm by 90 about z and 180 about  - alignes EDAX & Lab (image) 
    # reference frames
    if ebsd_convention == 'EDAX':
        slip_norm_lab = np.matmul(rmat_90_z,slip_norm_lab)
        slip_norm_lab = np.matmul(rmat_neg180_x,slip_norm_lab)    
    else:
        raise ValueError("Invalid EBSD convention provided.")

    # Project the slip_norm direction into the XY plane
    proj_xy_slip_norm = np.array([slip_norm_lab[0],slip_norm_lab[1],0]) \
        /np.linalg.norm(np.array([slip_norm_lab[0],slip_norm_lab[1],0]))

    # Find the orthogonal vector (trace of slip plane)
    basal_trace_vec = np.cross(proj_xy_slip_norm,\
                                    np.array([0,0,1])).reshape(1,3)
    # =========================================================================          
        
    return basal_trace_vec        


def calcSlipTracesHCPPrism(bunge_euler,ang_format,ebsd_convention):
    
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

        
    # Prismatic slip plane normals, assuming <a> || X
    n_pslip = np.array([[0.866025403784, 0.50, 0.00],\
                        [0.00, 1.00, 0.00],\
                        [-0.866025403784, 0.50, 0.00]])         
        
    # Basal slip plane normals
    n_bslip = np.array([0.00, 0.00, 1.00])           
    
    if ang_format == 'degrees':
        bunge_euler = bunge_euler*(np.pi/180)
        
    # Identify the parent orientation in quaternion Crystal -> Lab convention    
    # Bunge Euler default orientation: Lab -> Crystal   
    par_rot_scipy = R.from_euler('ZXZ',bunge_euler,degrees=False) # Crystal -> Lab 
    par_rot_rmat = par_rot_scipy.as_matrix() # Crystal -> Lab
  

    # =========================================================================    
    # ---- Prismatic Slip Mode ----
    # =========================================================================        
    # Loop through all three prismatic slip systems
    prism_trace_vec = np.zeros((3,3))
    for i in range(3):
        
        slip_norm_crystal = n_pslip[i,:]
        slip_norm_lab = np.matmul(par_rot_rmat,slip_norm_crystal)
    
        # Rotate slip norm by 90 about z and 180 about  - alignes EDAX & Lab (image) 
        # reference frames
        if ebsd_convention == 'EDAX':
            slip_norm_lab = np.matmul(rmat_90_z,slip_norm_lab)
            slip_norm_lab = np.matmul(rmat_neg180_x,slip_norm_lab)    
        else:
            raise ValueError("Invalid EBSD convention provided.") 
    
        # Project the slip_norm direction into the XY plane
        proj_xy_slip_norm = np.array([slip_norm_lab[0],slip_norm_lab[1],0]) \
            /np.linalg.norm(np.array([slip_norm_lab[0],slip_norm_lab[1],0]))
    
        # Find the orthogonal vector (trace of slip plane)
        prism_trace_vec[i,:] = np.cross(proj_xy_slip_norm,\
                                        np.array([0,0,1])).reshape(1,3)
    # =========================================================================          
        
    return prism_trace_vec        


def calcSlipTracesHCPPyra_I_A(bunge_euler,ang_format,latparm_a,latparm_c,ebsd_convention):
    
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

    sliptwinsys_miller = load_hcp_miller_indices('hcp_pyra_i_a_miller.csv')
    sliptwinsys_hkil_n = sliptwinsys_miller[:,0:4]    
    
    # Calculate the slip plane normals in a cartesian reference frame
    # Method copied from MTEX 5.10.0
    rot_rad = -30*(np.pi/180);
    rotz = np.array([[np.cos(rot_rad), -1*np.sin(rot_rad), 0], \
                     [np.sin(rot_rad), np.cos(rot_rad), 0], \
                     [0, 0, 1]]);
    a = latparm_a # Angstroms (any equivalent a & c units should be fine)
    c = latparm_c #    
    hcp_hkil_axis1 = np.matmul(rotz,np.array([a,0,0]).reshape(-1,1)).reshape(-1)
    hcp_hkil_axis2 = np.array([0, a, 0])
    hcp_hkil_axis3 = np.array([0, 0, c])
    bc = np.cross(hcp_hkil_axis2,hcp_hkil_axis3)
    V  = np.dot(hcp_hkil_axis1,bc)
    vec = np.zeros((3,3))
    vec[0,:] = bc / V;
    vec[1,:] = np.cross(hcp_hkil_axis3,hcp_hkil_axis1) / V
    vec[2,:] = np.cross(hcp_hkil_axis1,hcp_hkil_axis2) / V
    
    n_slip = np.zeros((6,3))
    for i in range(n_slip.shape[0]):
        hkl_tmp = np.array([sliptwinsys_hkil_n[i,0],sliptwinsys_hkil_n[i,1],sliptwinsys_hkil_n[i,3]])
        y_tmp = np.matmul(hkl_tmp,vec[:,0]) # Flip X and Y w.r.t. MTEX ref frame for HCP
        x_tmp = np.matmul(hkl_tmp,vec[:,1])
        z_tmp = np.matmul(hkl_tmp,vec[:,2])
        n_slip[i,:] = np.array([x_tmp, y_tmp, z_tmp])/np.linalg.norm(np.array([x_tmp, y_tmp, z_tmp]))  
        
    if ang_format == 'degrees':
        bunge_euler = bunge_euler*(np.pi/180)
        
    # Identify the parent orientation in quaternion Crystal -> Lab convention    
    # Bunge Euler default orientation: Lab -> Crystal   
    par_rot_scipy = R.from_euler('ZXZ',bunge_euler,degrees=False) # Crystal -> Lab 
    par_rot_rmat = par_rot_scipy.as_matrix() # Crystal -> Lab
  

    # =========================================================================    
    # ---- Pyramidal I <a> Slip Mode ----
    # =========================================================================        
    # Loop through all 6 pyramidal I <a> slip systems
    sliptwinsys_trace_vec  = np.zeros((6,3))
    for i in range(6):
        
        slip_norm_crystal = n_slip[i,:]
        slip_norm_lab = np.matmul(par_rot_rmat,slip_norm_crystal)
    
        # Rotate slip norm by 90 about z and 180 about  - alignes EDAX & Lab (image) 
        # reference frames
        if ebsd_convention == 'EDAX':
            slip_norm_lab = np.matmul(rmat_90_z,slip_norm_lab)
            slip_norm_lab = np.matmul(rmat_neg180_x,slip_norm_lab)    
        else:
            raise ValueError("Invalid EBSD convention provided.")  
    
        # Project the slip_norm direction into the XY plane
        proj_xy_slip_norm = np.array([slip_norm_lab[0],slip_norm_lab[1],0]) \
            /np.linalg.norm(np.array([slip_norm_lab[0],slip_norm_lab[1],0]))
    
        # Find the orthogonal vector (trace of slip plane)
        sliptwinsys_trace_vec [i,:] = np.cross(proj_xy_slip_norm,\
                                        np.array([0,0,1])).reshape(1,3)
    # =========================================================================          
        
    return sliptwinsys_trace_vec  


def calcSlipTracesHCPPyra_I_CA(bunge_euler,ang_format,latparm_a,latparm_c,ebsd_convention):
    
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

    sliptwinsys_miller = load_hcp_miller_indices('hcp_pyra_i_ca_miller.csv')
    sliptwinsys_hkil_n = sliptwinsys_miller[:,0:4]    
    
    # Calculate the slip plane normals in a cartesian reference frame
    # Method copied from MTEX 5.10.0
    rot_rad = -30*(np.pi/180);
    rotz = np.array([[np.cos(rot_rad), -1*np.sin(rot_rad), 0], \
                     [np.sin(rot_rad), np.cos(rot_rad), 0], \
                     [0, 0, 1]]);
    a = latparm_a # Angstroms (any equivalent a & c units should be fine)
    c = latparm_c #    
    hcp_hkil_axis1 = np.matmul(rotz,np.array([a,0,0]).reshape(-1,1)).reshape(-1)
    hcp_hkil_axis2 = np.array([0, a, 0])
    hcp_hkil_axis3 = np.array([0, 0, c])
    bc = np.cross(hcp_hkil_axis2,hcp_hkil_axis3)
    V  = np.dot(hcp_hkil_axis1,bc)
    vec = np.zeros((3,3))
    vec[0,:] = bc / V;
    vec[1,:] = np.cross(hcp_hkil_axis3,hcp_hkil_axis1) / V
    vec[2,:] = np.cross(hcp_hkil_axis1,hcp_hkil_axis2) / V
    
    n_slip = np.zeros((6,3))
    for i in range(n_slip.shape[0]):
        hkl_tmp = np.array([sliptwinsys_hkil_n[i,0],sliptwinsys_hkil_n[i,1],sliptwinsys_hkil_n[i,3]])
        y_tmp = np.matmul(hkl_tmp,vec[:,0]) # Flip X and Y w.r.t. MTEX ref frame for HCP
        x_tmp = np.matmul(hkl_tmp,vec[:,1])
        z_tmp = np.matmul(hkl_tmp,vec[:,2])
        n_slip[i,:] = np.array([x_tmp, y_tmp, z_tmp])/np.linalg.norm(np.array([x_tmp, y_tmp, z_tmp]))  
            
    if ang_format == 'degrees':
        bunge_euler = bunge_euler*(np.pi/180)
        
    # Identify the parent orientation in quaternion Crystal -> Lab convention    
    # Bunge Euler default orientation: Lab -> Crystal   
    par_rot_scipy = R.from_euler('ZXZ',bunge_euler,degrees=False) # Crystal -> Lab 
    par_rot_rmat = par_rot_scipy.as_matrix() # Crystal -> Lab
  

    # =========================================================================    
    # ---- Pyramidal I <a> Slip Mode ----
    # =========================================================================        
    # Loop through all 6 pyramidal I <a> slip systems
    sliptwinsys_trace_vec  = np.zeros((6,3))
    for i in range(6):
        
        slip_norm_crystal = n_slip[i,:]
        slip_norm_lab = np.matmul(par_rot_rmat,slip_norm_crystal)
    
        # Rotate slip norm by 90 about z and 180 about  - alignes EDAX & Lab (image) 
        # reference frames
        if ebsd_convention == 'EDAX':
            slip_norm_lab = np.matmul(rmat_90_z,slip_norm_lab)
            slip_norm_lab = np.matmul(rmat_neg180_x,slip_norm_lab)    
        else:
            raise ValueError("Invalid EBSD convention provided.")  
    
        # Project the slip_norm direction into the XY plane
        proj_xy_slip_norm = np.array([slip_norm_lab[0],slip_norm_lab[1],0]) \
            /np.linalg.norm(np.array([slip_norm_lab[0],slip_norm_lab[1],0]))
    
        # Find the orthogonal vector (trace of slip plane)
        sliptwinsys_trace_vec [i,:] = np.cross(proj_xy_slip_norm,\
                                        np.array([0,0,1])).reshape(1,3)
    # =========================================================================          
        
    return sliptwinsys_trace_vec  


def calcSlipTracesHCPPyra_II_CA(bunge_euler,ang_format,latparm_a,latparm_c,ebsd_convention):
    
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

    sliptwinsys_miller = load_hcp_miller_indices('hcp_pyra_ii_ca_miller.csv')
    sliptwinsys_hkil_n = sliptwinsys_miller[:,0:4]    
    
    # Calculate the slip plane normals in a cartesian reference frame
    # Method copied from MTEX 5.10.0
    rot_rad = -30*(np.pi/180);
    rotz = np.array([[np.cos(rot_rad), -1*np.sin(rot_rad), 0], \
                     [np.sin(rot_rad), np.cos(rot_rad), 0], \
                     [0, 0, 1]]);
    a = latparm_a # Angstroms (any equivalent a & c units should be fine)
    c = latparm_c #    
    hcp_hkil_axis1 = np.matmul(rotz,np.array([a,0,0]).reshape(-1,1)).reshape(-1)
    hcp_hkil_axis2 = np.array([0, a, 0])
    hcp_hkil_axis3 = np.array([0, 0, c])
    bc = np.cross(hcp_hkil_axis2,hcp_hkil_axis3)
    V  = np.dot(hcp_hkil_axis1,bc)
    vec = np.zeros((3,3))
    vec[0,:] = bc / V;
    vec[1,:] = np.cross(hcp_hkil_axis3,hcp_hkil_axis1) / V
    vec[2,:] = np.cross(hcp_hkil_axis1,hcp_hkil_axis2) / V
    
    n_slip = np.zeros((6,3))
    for i in range(n_slip.shape[0]):
        hkl_tmp = np.array([sliptwinsys_hkil_n[i,0],sliptwinsys_hkil_n[i,1],sliptwinsys_hkil_n[i,3]])
        y_tmp = np.matmul(hkl_tmp,vec[:,0]) # Flip X and Y w.r.t. MTEX ref frame for HCP
        x_tmp = np.matmul(hkl_tmp,vec[:,1])
        z_tmp = np.matmul(hkl_tmp,vec[:,2])
        n_slip[i,:] = np.array([x_tmp, y_tmp, z_tmp])/np.linalg.norm(np.array([x_tmp, y_tmp, z_tmp]))  
            
    if ang_format == 'degrees':
        bunge_euler = bunge_euler*(np.pi/180)
        
    # Identify the parent orientation in quaternion Crystal -> Lab convention    
    # Bunge Euler default orientation: Lab -> Crystal   
    par_rot_scipy = R.from_euler('ZXZ',bunge_euler,degrees=False) # Crystal -> Lab 
    par_rot_rmat = par_rot_scipy.as_matrix() # Crystal -> Lab
  

    # =========================================================================    
    # ---- Pyramidal I <a> Slip Mode ----
    # =========================================================================        
    # Loop through all 6 pyramidal I <a> slip systems
    sliptwinsys_trace_vec  = np.zeros((6,3))
    for i in range(6):
        
        slip_norm_crystal = n_slip[i,:]
        slip_norm_lab = np.matmul(par_rot_rmat,slip_norm_crystal)
    
        # Rotate slip norm by 90 about z and 180 about  - alignes EDAX & Lab (image) 
        # reference frames
        if ebsd_convention == 'EDAX':
            slip_norm_lab = np.matmul(rmat_90_z,slip_norm_lab)
            slip_norm_lab = np.matmul(rmat_neg180_x,slip_norm_lab)    
        else:
            raise ValueError("Invalid EBSD convention provided.")  
    
        # Project the slip_norm direction into the XY plane
        proj_xy_slip_norm = np.array([slip_norm_lab[0],slip_norm_lab[1],0]) \
            /np.linalg.norm(np.array([slip_norm_lab[0],slip_norm_lab[1],0]))
    
        # Find the orthogonal vector (trace of slip plane)
        sliptwinsys_trace_vec [i,:] = np.cross(proj_xy_slip_norm,\
                                        np.array([0,0,1])).reshape(1,3)
    # =========================================================================          
        
    return sliptwinsys_trace_vec  


def calcTwinTracesHCP(bunge_euler,twin_mode,ang_format,latparm_a,latparm_c,ebsd_convention):
    
    
    # Twin modes:
    # t1: {10-12}
    # t2: {11-21}
    # c1: {10-12}
    # c2: {11-22}
    # c3: {10-11}

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

    sliptwinsys_miller = load_hcp_miller_indices('hcp_twin_%s_miller.csv' % twin_mode)
    sliptwinsys_hkil_n = sliptwinsys_miller[:,0:4]    
    
    # Calculate the slip plane normals in a cartesian reference frame
    # Method copied from MTEX 5.10.0
    rot_rad = -30*(np.pi/180);
    rotz = np.array([[np.cos(rot_rad), -1*np.sin(rot_rad), 0], \
                     [np.sin(rot_rad), np.cos(rot_rad), 0], \
                     [0, 0, 1]]);
    a = latparm_a # Angstroms (any equivalent a & c units should be fine)
    c = latparm_c #    
    hcp_hkil_axis1 = np.matmul(rotz,np.array([a,0,0]).reshape(-1,1)).reshape(-1)
    hcp_hkil_axis2 = np.array([0, a, 0])
    hcp_hkil_axis3 = np.array([0, 0, c])
    bc = np.cross(hcp_hkil_axis2,hcp_hkil_axis3)
    V  = np.dot(hcp_hkil_axis1,bc)
    vec = np.zeros((3,3))
    vec[0,:] = bc / V;
    vec[1,:] = np.cross(hcp_hkil_axis3,hcp_hkil_axis1) / V
    vec[2,:] = np.cross(hcp_hkil_axis1,hcp_hkil_axis2) / V
    
    n_slip = np.zeros((6,3))
    for i in range(n_slip.shape[0]):
        hkl_tmp = np.array([sliptwinsys_hkil_n[i,0],sliptwinsys_hkil_n[i,1],sliptwinsys_hkil_n[i,3]])
        y_tmp = np.matmul(hkl_tmp,vec[:,0]) # Flip X and Y w.r.t. MTEX ref frame for HCP
        x_tmp = np.matmul(hkl_tmp,vec[:,1])        
        z_tmp = np.matmul(hkl_tmp,vec[:,2])
        n_slip[i,:] = np.array([x_tmp, y_tmp, z_tmp])/np.linalg.norm(np.array([x_tmp, y_tmp, z_tmp]))  
            
    if ang_format == 'degrees':
        bunge_euler = bunge_euler*(np.pi/180)
        
    # Identify the parent orientation in quaternion Crystal -> Lab convention    
    # Bunge Euler default orientation: Lab -> Crystal   
    par_rot_scipy = R.from_euler('ZXZ',bunge_euler,degrees=False) # Crystal -> Lab 
    par_rot_rmat = par_rot_scipy.as_matrix() # Crystal -> Lab
  

    # =========================================================================    
    # ---- Twin T1 {10-12} Mode ----
    # =========================================================================        
    # Loop through all 6 pyramidal I <a> slip systems
    sliptwinsys_trace_vec = np.zeros((6,3))
    for i in range(6):
        
        slip_norm_crystal = n_slip[i,:]
        slip_norm_lab = np.matmul(par_rot_rmat,slip_norm_crystal)
    
        # Rotate slip norm by 90 about z and 180 about  - alignes EDAX & Lab (image) 
        # reference frames
        if ebsd_convention == 'EDAX':
            slip_norm_lab = np.matmul(rmat_90_z,slip_norm_lab)
            slip_norm_lab = np.matmul(rmat_neg180_x,slip_norm_lab)    
        else:
            raise ValueError("Invalid EBSD convention provided.")  
    
        # Project the slip_norm direction into the XY plane
        proj_xy_slip_norm = np.array([slip_norm_lab[0],slip_norm_lab[1],0]) \
            /np.linalg.norm(np.array([slip_norm_lab[0],slip_norm_lab[1],0]))
    
        # Find the orthogonal vector (trace of slip plane)
        sliptwinsys_trace_vec[i,:] = np.cross(proj_xy_slip_norm,\
                                        np.array([0,0,1])).reshape(1,3)
    # =========================================================================          
        
    return sliptwinsys_trace_vec 




def plotAnySlipTwinTrace(img,row,col,trace_vec):
    
    #%matplotlib qt5
    #%matplotlib inline
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(img, interpolation='nearest')
    
    quiver_scale = 10
    quiver_width = 0.005
    
    # Plot arrows in the (+/-) directions of the slip plane traces
    for i in range(trace_vec.shape[0]):
        ax.quiver(col,row, trace_vec[i,0], trace_vec[i,1], \
                  color='aqua', scale=quiver_scale, width=quiver_width)
        ax.quiver(col,row, trace_vec[i,0]*-1, trace_vec[i,1]*-1, \
                  color='aqua', scale=quiver_scale, width=quiver_width)                

        
if __name__ == "__main__":
    # =============================================================================
    # %% ---- User Inputs ----
    # =============================================================================

    raw_data_folder = "/mnt/hgfs/METIS/data_processing_scripts/bes_dic_ti/z_check_pk_data/"

    #in_filename = 'duncan_crop_event_trace_comparison_vector_overlay_corrected.png'
    in_filename = 'duncan_crop_event_trace_comparison_all_trace_vectors_overlay_corrected.png'
    img = imread(raw_data_folder+in_filename)

    # Bunge Euler angle (ZXZ)
    input_ori = np.array([80.7, 35.8, 338.7])*(np.pi/180) # Lab -> Crystal
    #input_ori = np.array([0, 0, 0])*(np.pi/180) # Lab -> Crystal

    basal_trace_vec = calcSlipTracesHCPBasal(input_ori,'radians','EDAX')
    prism_trace_vec = calcSlipTracesHCPPrism(input_ori,'radians','EDAX')
    pyra_i_a_trace_vec = calcSlipTracesHCPPyra_I_A(input_ori,'radians',2.95,4.686,'EDAX')
    pyra_ii_ca_trace_vec = calcSlipTracesHCPPyra_II_CA(input_ori,'radians',2.95,4.686,'EDAX')
    twin_t1_trace_vec = calcTwinTracesHCP(input_ori,'t1','radians',2.95,4.686,'EDAX')
    twin_t2_trace_vec = calcTwinTracesHCP(input_ori,'t2','radians',2.95,4.686,'EDAX')
    twin_c1_trace_vec = calcTwinTracesHCP(input_ori,'c1','radians',2.95,4.686,'EDAX')
    twin_c2_trace_vec = calcTwinTracesHCP(input_ori,'c2','radians',2.95,4.686,'EDAX')
    twin_c3_trace_vec = calcTwinTracesHCP(input_ori,'c3','radians',2.95,4.686,'EDAX')

    # Spot to plot traces for grain 1
    row_s = 1130 # THIS IS Y
    col_s = 820 # THIS IS X
    #plotAnySlipTwinTrace(img,row_s,col_s,basal_trace_vec)
    #plotAnySlipTwinTrace(img,row_s,col_s,prism_trace_vec)
    #plotAnySlipTwinTrace(img,row_s,col_s,pyra_i_a_trace_vec)
    #plotAnySlipTwinTrace(img,row_s,col_s,pyra_ii_ca_trace_vec)
    plotAnySlipTwinTrace(img,row_s,col_s,twin_c3_trace_vec)
