import cupy as cp

def median_filter(data):
    # Apply 3-point median filter. data is 1D array.
    padded = cp.pad(data, (1, 1), mode='edge')
    stacked = cp.vstack((padded[:-2], padded[1:-1], padded[2:]))
    return cp.median(stacked, axis=0)

def ndq_single(data, spline_gpu, dt):
    # data is a 1D cupy array (for one mode)
    length = data.shape[0]
    
    # Needs to be 2D for batched spline, so reshape to (1, length)
    data_2d = data.reshape(1, -1)
    
    FM_2d = data_2d.copy()
    Absolute = cp.abs(FM_2d)
    
    # iterative FM normalization
    for k in range(100):
        Absolute = cp.abs(FM_2d)
        up_extrema, low_extrema, valid_mask = spline_gpu.extract_extrema(Absolute)
        
        up_count = up_extrema.sum(axis=1)[0]
        if up_count < 3:
            break
            
        up_env = spline_gpu.compute_spline(Absolute, up_extrema, type_flag=0)
        up_env = cp.maximum(up_env, 0.01)
        
        FM_2d = FM_2d / up_env
        
        # Check termination
        if cp.all((FM_2d >= -1.0) & (FM_2d <= 1.0)):
            break
            
    FM = FM_2d[0]
    
    # Check success
    ndq_success = bool(cp.all((FM >= -1.0) & (FM <= 1.0)))
    if not ndq_success:
        return cp.zeros_like(data), cp.zeros_like(data), cp.zeros_like(data), cp.zeros_like(data), False
        
    AM = cp.zeros_like(data)
    AM[1:-1] = data[1:-1] / FM[1:-1]
    
    # Handle potentially bad values in sqrt
    FM_clamped = cp.clip(FM, -0.999999, 0.999999)
    phase_angle = cp.arctan(FM_clamped / cp.sqrt(1.0 - FM_clamped * FM_clamped))
    
    IP = cp.zeros_like(data)
    IP[0] = phase_angle[0]
    # IP[i] = IP[i-1] + abs(phase_angle[i-1] - phase_angle[i])
    # This is a cumulative sum of differences
    diffs = cp.abs(phase_angle[:-1] - phase_angle[1:])
    IP[1:] = IP[0] + cp.cumsum(diffs)
    
    real_freq = cp.zeros_like(data)
    real_freq[:-1] = (IP[1:] - IP[:-1]) / dt
    real_freq[-1] = real_freq[-2]
    
    IF = median_filter(real_freq)
    IF[0] = real_freq[0]
    IF[-1] = real_freq[-1]
    
    FM = cp.nan_to_num(FM, nan=0.0, posinf=0.0, neginf=0.0)
    AM = cp.nan_to_num(AM, nan=0.0, posinf=0.0, neginf=0.0)
    IP = cp.nan_to_num(IP, nan=0.0, posinf=0.0, neginf=0.0)
    IF = cp.nan_to_num(IF, nan=0.0, posinf=0.0, neginf=0.0)
    
    return FM, AM, IP, IF, True
