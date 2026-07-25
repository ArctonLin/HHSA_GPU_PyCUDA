import cupy as cp
from cupyx import scatter_add

def HSA(IF, AM, ndq_success, length, mode, time_cell, freq_cell, dt):
    # IF, AM are lists or arrays of shape (mode, length)
    max_frequency = 1.0 / dt
    
    # Calculate dT and dF
    maxT = float(length)
    dT = maxT / float(time_cell - 1)
    dF = max_frequency / float(freq_cell - 1)
    
    hsa_map = cp.zeros((time_cell, freq_cell), dtype=cp.float32)
    
    for j in range(mode):
        if ndq_success[j]:
            i = cp.arange(length, dtype=cp.float32)
            x = (i / dT).astype(cp.int32)
            y = (IF[j] / dF).astype(cp.int32)
            
            valid = (x >= 0) & (x < time_cell) & (y >= 0) & (y < freq_cell)
            
            scatter_add(hsa_map, (x[valid], y[valid]), AM[j][valid])
            
    return hsa_map, max_frequency, maxT, dT, dF

def HHS2(IF, Holo_IF, Holo_AM, length, mode, ndq_success, holo_ndq_success, 
         time_cell, freq_cell, holo_freq_cell, max_FM_frequency, max_AM_frequency):
         
    # Returns map (Freq_cell, Holo_Freq_cell) and map2 (Time_cell, Freq_cell, Holo_Freq_cell)
    hhs_map = cp.zeros((freq_cell, holo_freq_cell), dtype=cp.float32)
    hhsa_map = cp.zeros((time_cell, freq_cell, holo_freq_cell), dtype=cp.float32)
    
    maxT = float(length)
    dT = maxT / float(time_cell - 1)
    
    dFMF = max_FM_frequency / 2.0 / float(freq_cell - 1)
    dAMF = max_AM_frequency / 2.0 / float(holo_freq_cell - 1)
    
    for j in range(mode):
        if ndq_success[j]:
            i = cp.arange(length, dtype=cp.float32)
            t = (i / dT).astype(cp.int32)
            x = (IF[j] / dFMF).astype(cp.int32)
            
            valid_t_x = (t >= 0) & (t < time_cell) & (x >= 0) & (x < freq_cell)
            
            for k in range(mode):
                if holo_ndq_success[j][k]:
                    y = (Holo_IF[j][k] / dAMF).astype(cp.int32)
                    
                    valid = valid_t_x & (y >= 0) & (y < holo_freq_cell)
                    
                    # scatter add to hhs_map (2D)
                    scatter_add(hhs_map, (x[valid], y[valid]), Holo_AM[j][k][valid])
                    
                    # scatter add to hhsa_map (3D)
                    scatter_add(hhsa_map, (t[valid], x[valid], y[valid]), Holo_AM[j][k][valid])
                    
    return hhs_map, hhsa_map, maxT, dT, dFMF, dAMF
