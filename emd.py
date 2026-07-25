import cupy as cp
import numpy as np
from tqdm import tqdm

def STDEV_batched(data):
    return cp.std(data, axis=1)

def STDEV_single(data):
    return cp.std(data)

def full_emd_batched(data_batch, mode, spline_gpu, max_iterations=250, target_stdev=0.1, pbar=None):
    num_ensembles, length = data_batch.shape
    result = cp.zeros((mode, num_ensembles, length), dtype=cp.float32)
    residual_out = cp.zeros((mode, num_ensembles, length), dtype=cp.float32)
    
    current_residual = data_batch.copy()
    
    for m in range(mode):
        current = current_residual.copy()
        finished = cp.zeros(num_ensembles, dtype=cp.bool_)
        
        for iteration in range(max_iterations):
            if finished.all():
                break
                
            up_extrema, low_extrema, valid_mask = spline_gpu.extract_extrema(current)
            
            finished = finished | (~valid_mask)
            if finished.all():
                break
                
            up_env = spline_gpu.compute_spline(current, up_extrema, type_flag=1)
            low_env = spline_gpu.compute_spline(current, low_extrema, type_flag=1)
            
            mean_env = (up_env + low_env) * 0.5
            next_current = current - mean_env
            
            stdev_mean = STDEV_batched(mean_env)
            stdev_curr = STDEV_batched(current)
            
            converged = stdev_mean < (stdev_curr * target_stdev)
            finished = finished | converged
            
            current = cp.where(finished[:, None], current, next_current)
            
        result[m] = current
        current_residual -= current
        residual_out[m] = current_residual
        
        if pbar is not None:
            pbar.update(1)
            
    return result, residual_out

def ceemd(data, mode, target_stdev, ensemble, dt, spline_gpu, ceemd_flag=True, desc="Apply HHT", leave_progress=True):
    length = data.shape[0]
    
    avg = cp.mean(data)
    data = data - avg
    
    data_stdev = cp.std(data)
    if data_stdev < 0.01:
        data_stdev = cp.array(0.01, dtype=cp.float32)
        
    source = data / data_stdev
    # Target ~3GB VRAM usage.
    # Peak memory is roughly 130 bytes per element (chunk_size * length).
    # 3GB = 3 * 1024^3 bytes. target_elements = 3GB / 130 bytes ~= 24,000,000.
    target_elements = 20000000
    chunk_size = max(1, target_elements // length)
    chunk_size = min(chunk_size, ensemble)
    num_chunks = (ensemble + chunk_size - 1) // chunk_size
    
    mixture_result = cp.zeros((mode, length), dtype=cp.float32)
    mixture_residual = cp.zeros((mode, length), dtype=cp.float32)
    
    total_steps = mode * num_chunks * (2 if ceemd_flag else 1)
    with tqdm(total=total_steps, desc=desc, leave=leave_progress) as pbar:
        for c in range(num_chunks):
            current_chunk_size = min(chunk_size, ensemble - c * chunk_size)
            noise = cp.random.normal(0.0, 0.1, (current_chunk_size, length)).astype(cp.float32)
            
            # Forward CEEMD
            input_batch = source + noise
            imf_batch, res_batch = full_emd_batched(input_batch, mode, spline_gpu, target_stdev=target_stdev, pbar=pbar)
            mixture_result += cp.sum(imf_batch, axis=1)
            mixture_residual += cp.sum(res_batch, axis=1)
            
            # Backward CEEMD (Negative Noise)
            if ceemd_flag:
                input_batch = source - noise
                imf_batch, res_batch = full_emd_batched(input_batch, mode, spline_gpu, target_stdev=target_stdev, pbar=pbar)
                mixture_result += cp.sum(imf_batch, axis=1)
                mixture_residual += cp.sum(res_batch, axis=1)
            
        # Free memory
        noise = None
        input_batch = None
        imf_batch = None
        res_batch = None
        cp.get_default_memory_pool().free_all_blocks()
        
    ensemble_num = ensemble * 2 if ceemd_flag else ensemble
    result = (data_stdev * mixture_result) / ensemble_num
    residual = (data_stdev * mixture_residual) / ensemble_num
    
    emd_success = cp.zeros(mode, dtype=cp.bool_)
    for m in range(mode):
        emd_success[m] = STDEV_single(result[m]) > 0.01 * data_stdev
        
    return result, residual, emd_success
    
def iceemdan(data, mode, ensemble, spline_gpu, desc="Apply HHT (ICEEMDAN)", leave_progress=True):
    length = data.shape[0]
    result = cp.zeros((mode, length), dtype=cp.float32)
    residual = cp.zeros((mode, length), dtype=cp.float32)
    emd_success = cp.zeros(mode, dtype=cp.bool_)
    
    data_stdev = cp.std(data)
    if data_stdev < 0.01:
        data_stdev = cp.array(0.01, dtype=cp.float32)
        
    avg = cp.mean(data)
    res = (data - avg) / data_stdev
    # Target ~3GB VRAM usage.
    target_elements = 20000000
    chunk_size = max(1, target_elements // length)
    chunk_size = min(chunk_size, ensemble)
    total_steps = mode * ((ensemble + chunk_size - 1) // chunk_size)
    
    with tqdm(total=total_steps, desc=desc, leave=leave_progress) as pbar:
        for m in range(mode):
            res_stdev = cp.std(res)
            mean_envelope = cp.zeros(length, dtype=cp.float32)
            
            num_chunks = (ensemble + chunk_size - 1) // chunk_size
            
            for c in range(num_chunks):
                current_chunk_size = min(chunk_size, ensemble - c * chunk_size)
                noise = cp.random.normal(0.0, 0.2, (current_chunk_size, length)).astype(cp.float32)
                
                # Positive
                input_batch = res + noise * res_stdev
                up_extrema, low_extrema, valid = spline_gpu.extract_extrema(input_batch)
                up_env = spline_gpu.compute_spline(input_batch, up_extrema, 1)
                low_env = spline_gpu.compute_spline(input_batch, low_extrema, 1)
                mean_envelope += cp.sum(up_env + low_env, axis=0) * 0.5 / (ensemble * 2.0)
                
                # Negative
                input_batch = res - noise * res_stdev
                up_extrema, low_extrema, valid = spline_gpu.extract_extrema(input_batch)
                up_env = spline_gpu.compute_spline(input_batch, up_extrema, 1)
                low_env = spline_gpu.compute_spline(input_batch, low_extrema, 1)
                mean_envelope += cp.sum(up_env + low_env, axis=0) * 0.5 / (ensemble * 2.0)
                
                noise = None
                input_batch = None
                cp.get_default_memory_pool().free_all_blocks()
                
                pbar.update(1)
            
        result[m] = (res - mean_envelope) * data_stdev
        residual[m] = res * data_stdev
        
        emd_success[m] = STDEV_single(result[m]) > 0.01 * data_stdev
        res = mean_envelope.copy()
        
    return result, residual, emd_success
