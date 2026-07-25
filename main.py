import cupy as cp
import numpy as np
import os
import sys
import time

from config import Config
from spline_gpu import SplineGPU
from emd import ceemd, iceemdan
from ndq import ndq_single
from spectra import HSA, HHS2
from plotter import plot_hht, plot_hsa, plot_hhs, plot_hhsa_3d, create_hhsa_gif

def load_data(filename):
    with open(filename, 'r') as f:
        data = [float(x.strip()) for x in f.readlines() if x.strip()]
    return np.array(data, dtype=np.float32)

def save_csv(filename, data_dict, length):
    with open(filename, 'w') as f:
        headers = list(data_dict.keys())
        f.write(",".join(headers) + ",\n")
        
        for i in range(length):
            row = [f"{data_dict[h][i]:.6f}" if h in data_dict else "" for h in headers]
            f.write(",".join(row) + ",\n")

def save_map(filename_base, map_data, max_x, max_y, dt_x, dt_y):
    with open(f"{filename_base}.txt", 'w') as f:
        f.write(f"max_time={max_x}\nmax_frequecny={max_y}\n")
    with open(f"{filename_base}.csv", 'w') as f:
        for i in range(map_data.shape[0]):
            f.write(",".join(f"{v:.6f}" for v in map_data[i]) + ",\n")
    with open(f"{filename_base}.dat", 'w') as f:
        for i in range(map_data.shape[0]):
            for j in range(map_data.shape[1]):
                if abs(map_data[i, j]) < 1e12 and map_data[i, j] > 1e-7:
                    f.write(f"{i * dt_x:.6f} {j * dt_y:.6f} {map_data[i, j]:.6f}\n")

def main():
    print("HHSA GPU PyCUDA version")
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
    else:
        print("Please provide an input file (e.g., python main.py HeartRate.txt)")
        print("Defaulting to HeartRate.txt...")
        input_filename = "HeartRate.txt"
    
    cfg = Config("HHSA_profile.txt")
    print(f"EMD type: {cfg.EMD_type}")
    print(f"Mode: {cfg.mode}")
    print(f"Ensemble: {cfg.ensemble}")
    
    cpu_data = load_data(input_filename)
    data_len = len(cpu_data)
    print(f"DataLength={data_len}")
    
    # Move to GPU
    data = cp.array(cpu_data)
    
    # Init Spline GPU with batched support up to chunk_size=50 or max ensemble
    spline_gpu = SplineGPU(max_length=data_len, num_ensembles=max(cfg.ensemble, 1))
    
    # 1. HHT (EMD)
    print("Apply HHT on input data")
    start_time = time.time()
    if cfg.EMD_type == "ICEEMDAN":
        result, residual, emd_success = iceemdan(data, cfg.mode, cfg.ensemble, spline_gpu)
    elif cfg.EMD_type == "CEEMD":
        result, residual, emd_success = ceemd(data, cfg.mode, cfg.TargetSTDEV, cfg.ensemble, cfg.dt, spline_gpu, ceemd_flag=True)
    elif cfg.EMD_type == "EEMD":
        result, residual, emd_success = ceemd(data, cfg.mode, cfg.TargetSTDEV, cfg.ensemble, cfg.dt, spline_gpu, ceemd_flag=False)
    
    print(f"HHT Time: {time.time() - start_time:.2f}s")
    
    # 2. NDQ on EMD result
    print("NDQ report:")
    FM = cp.zeros((cfg.mode, data_len), dtype=cp.float32)
    AM = cp.zeros((cfg.mode, data_len), dtype=cp.float32)
    IP = cp.zeros((cfg.mode, data_len), dtype=cp.float32)
    IF = cp.zeros((cfg.mode, data_len), dtype=cp.float32)
    ndq_success = cp.zeros(cfg.mode, dtype=cp.bool_)
    
    for j in range(cfg.mode):
        if emd_success[j]:
            fm, am, ip, _if, success = ndq_single(result[j], spline_gpu, cfg.dt)
            FM[j] = fm
            AM[j] = am
            IP[j] = ip
            IF[j] = _if
            ndq_success[j] = success
            
    # Output HHT to CPU
    result_cpu = result.get()
    residual_cpu = residual.get()
    
    print("Saving HHT CSV...")
    hht_dict = {"data": cpu_data}
    up_cpu = np.zeros_like(result_cpu)
    low_cpu = np.zeros_like(result_cpu)
    mean_cpu = np.zeros_like(result_cpu)
    
    spline_single = SplineGPU(max_length=data_len, num_ensembles=1)
    
    for j in range(cfg.mode):
        if emd_success[j]:
            imf_2d = result[j].reshape(1, -1)
            up_ex, low_ex, valid = spline_single.extract_extrema(imf_2d)
            if valid[0]:
                up_env = spline_single.compute_spline(imf_2d, up_ex, type_flag=1)[0]
                low_env = spline_single.compute_spline(imf_2d, low_ex, type_flag=1)[0]
                up_cpu[j] = up_env.get()
                low_cpu[j] = low_env.get()
                mean_cpu[j] = (up_cpu[j] + low_cpu[j]) * 0.5
            
            hht_dict[f"IMF{j+1}"] = result_cpu[j]
            hht_dict[f"Residual{j+1}"] = residual_cpu[j]
            hht_dict[f"Up{j+1}"] = up_cpu[j]
            hht_dict[f"Low{j+1}"] = low_cpu[j]
            hht_dict[f"Mean{j+1}"] = mean_cpu[j]
            
        if ndq_success[j]:
            hht_dict[f"FM{j+1}"] = FM[j].get()
            hht_dict[f"AM{j+1}"] = AM[j].get()
            hht_dict[f"IP{j+1}"] = IP[j].get()
            hht_dict[f"IF{j+1}"] = IF[j].get()
            
    save_csv(f"{input_filename}_HHT.csv", hht_dict, data_len)
    
    plot_hht(cpu_data, result_cpu, residual_cpu, cfg.mode, f"{input_filename}_HHT.png", cfg.ImageSizeX1, cfg.ImageSizeY1)
    
    # 3. HSA
    print("Generate HSA result")
    hsa_map, max_frequency, max_time, HSA_dt, HSA_df = HSA(IF, AM, ndq_success, data_len, cfg.mode, cfg.Time_cell, cfg.Freq_cell, cfg.dt)
    hsa_map_cpu = hsa_map.get()
    
    save_map(f"{input_filename}_HSA", hsa_map_cpu, max_time, max_frequency, HSA_dt, HSA_df)
    plot_hsa(hsa_map_cpu, max_time, max_frequency, f"{input_filename}_HSA.png", cfg.ImageSizeX1, cfg.ImageSizeY1)
    
    # 4. HHSA (EMD on Envelope)
    print("Apply HHSA")
    Holo_result = [None] * cfg.mode
    Holo_residual = [None] * cfg.mode
    Holo_emd_success = cp.zeros((cfg.mode, cfg.mode), dtype=cp.bool_)
    
    Holo_FM = [None] * cfg.mode
    Holo_AM = [None] * cfg.mode
    Holo_IP = [None] * cfg.mode
    Holo_IF = [None] * cfg.mode
    Holo_ndq_success = cp.zeros((cfg.mode, cfg.mode), dtype=cp.bool_)
    
    # Reshape for single spline evaluation
    spline_single = SplineGPU(max_length=data_len, num_ensembles=1)
    
    from tqdm import tqdm
    for j in tqdm(range(cfg.mode), desc="Apply HHSA"):
        if emd_success[j]:
            imf_2d = result[j].reshape(1, -1)
            up_ex, low_ex, valid = spline_single.extract_extrema(imf_2d)
            if up_ex.sum() >= 3:
                up_env = spline_single.compute_spline(imf_2d, up_ex, type_flag=0)[0]
                
                if cfg.EMD_type == "ICEEMDAN":
                    h_res, h_resid, h_succ = iceemdan(up_env, cfg.mode, cfg.ensemble, spline_gpu, desc=f"Mode {j+1} ICE", leave_progress=False)
                elif cfg.EMD_type == "CEEMD":
                    h_res, h_resid, h_succ = ceemd(up_env, cfg.mode, cfg.TargetSTDEV, cfg.ensemble, cfg.dt, spline_gpu, True, desc=f"Mode {j+1} CEEMD", leave_progress=False)
                elif cfg.EMD_type == "EEMD":
                    h_res, h_resid, h_succ = ceemd(up_env, cfg.mode, cfg.TargetSTDEV, cfg.ensemble, cfg.dt, spline_gpu, False, desc=f"Mode {j+1} EEMD", leave_progress=False)
                
                Holo_result[j] = h_res
                Holo_residual[j] = h_resid
                Holo_emd_success[j] = h_succ
                
                # NDQ on Holo
                h_fm = cp.zeros((cfg.mode, data_len), dtype=cp.float32)
                h_am = cp.zeros((cfg.mode, data_len), dtype=cp.float32)
                h_ip = cp.zeros((cfg.mode, data_len), dtype=cp.float32)
                h_if = cp.zeros((cfg.mode, data_len), dtype=cp.float32)
                
                for k in range(cfg.mode):
                    if h_succ[k]:
                        fm, am, ip, _if, success = ndq_single(h_res[k], spline_single, cfg.dt)
                        h_fm[k] = fm
                        h_am[k] = am
                        h_ip[k] = ip
                        h_if[k] = _if
                        Holo_ndq_success[j, k] = success
                        
                Holo_FM[j] = h_fm
                Holo_AM[j] = h_am
                Holo_IF[j] = h_if
                
                holo_dict = {"data": up_env.get()}
                h_res_cpu = h_res.get()
                h_resid_cpu = h_resid.get()
                up_holo_cpu = np.zeros_like(h_res_cpu)
                low_holo_cpu = np.zeros_like(h_res_cpu)
                mean_holo_cpu = np.zeros_like(h_res_cpu)
                for k in range(cfg.mode):
                    if h_succ[k]:
                        himf_2d = h_res[k].reshape(1, -1)
                        h_up_ex, h_low_ex, h_valid = spline_single.extract_extrema(himf_2d)
                        if h_valid[0]:
                            h_up_env = spline_single.compute_spline(himf_2d, h_up_ex, type_flag=1)[0]
                            h_low_env = spline_single.compute_spline(himf_2d, h_low_ex, type_flag=1)[0]
                            up_holo_cpu[k] = h_up_env.get()
                            low_holo_cpu[k] = h_low_env.get()
                            mean_holo_cpu[k] = (up_holo_cpu[k] + low_holo_cpu[k]) * 0.5
                            
                        holo_dict[f"IMF{k+1}"] = h_res_cpu[k]
                        holo_dict[f"Residual{k+1}"] = h_resid_cpu[k]
                        holo_dict[f"Up{k+1}"] = up_holo_cpu[k]
                        holo_dict[f"Low{k+1}"] = low_holo_cpu[k]
                        holo_dict[f"Mean{k+1}"] = mean_holo_cpu[k]
                        
                    if Holo_ndq_success[j, k]:
                        holo_dict[f"FM{k+1}"] = h_fm[k].get()
                        holo_dict[f"AM{k+1}"] = h_am[k].get()
                        holo_dict[f"IP{k+1}"] = h_ip[k].get()
                        holo_dict[f"IF{k+1}"] = h_if[k].get()
                
                save_csv(f"{input_filename}_HHSA_mode{j+1}.csv", holo_dict, data_len)
                
    # 5. HHS
    print("Generate HHS result")
    max_FM_frequency = 1.0 / cfg.dt
    max_AM_frequency = 1.0 / cfg.dt
    
    hhs_map, hhsa_map, maxT, dT, dFMF, dAMF = HHS2(
        IF, Holo_IF, Holo_AM, data_len, cfg.mode, ndq_success, Holo_ndq_success,
        cfg.Time_cell, cfg.Freq_cell, cfg.Holo_Freq_cell, max_FM_frequency, max_AM_frequency
    )
    
    hhs_map_cpu = hhs_map.get()
    hhsa_map_cpu = hhsa_map.get()
    
    save_map(f"{input_filename}_HHS", hhs_map_cpu, max_FM_frequency, max_AM_frequency, dFMF, dAMF)
    
    plot_hhs(hhs_map_cpu, max_FM_frequency, max_AM_frequency, f"{input_filename}_HHS.png", cfg.ImageSizeX2, cfg.ImageSizeY2)
    
    plot_hhsa_3d(hhsa_map_cpu, max_time, max_FM_frequency, max_AM_frequency, f"{input_filename}_HHSA_3D.png",
                 cfg.dt, dFMF, dAMF, cfg.ImageSizeX2, cfg.ImageSizeY2)
                 
    create_hhsa_gif(hhsa_map_cpu, cfg.Time_cell, max_FM_frequency, max_AM_frequency, 
                    f"{input_filename}_HHSA.gif", cfg.dt, cfg.Time_cell_group, cfg.ImageSizeX2, cfg.ImageSizeY2)
                    
    print("All tasks completed.")
    # GPU memory will be freed on exit

if __name__ == "__main__":
    main()
