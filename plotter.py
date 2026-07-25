import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio.v2 as imageio
import os
import io
from tqdm import tqdm

def get_color_scale(img, exclude_bottom_rows=5):
    # Exclude the bottom rows (low frequencies/DC) from the scaling calculation
    # to completely prevent them from affecting the contrast.
    if img.shape[0] > exclude_bottom_rows * 2:
        eval_img = img[exclude_bottom_rows:]
    else:
        eval_img = img
        
    valid_pixels = eval_img[eval_img > 1e-7]
    if len(valid_pixels) > 0:
        # Use 99.9th percentile of the NON-DC area to find the true signal peaks
        vmax = float(np.percentile(valid_pixels, 99.9))
        # Use 10th percentile of valid pixels to suppress background noise into dark blue
        vmin = float(np.percentile(valid_pixels, 10.0))
    else:
        vmax = float(img.max())
        vmin = float(img.min())
        
    vmax = max(vmax, vmin * 10.0)
    return vmin, vmax

def plot_hht(data, result, residual, mode, output_name, width, height):
    length = len(data)
    # C++ Uses ImageSizeY1*2 and ImageSizeX1*2 for HHT height and width respectively.
    fig, axes = plt.subplots(mode + 2, 1, figsize=(width/100.0, height/100.0))
    
    axes[0].plot(data, 'b-')
    axes[0].set_title('Signal')
    
    for m in range(mode):
        axes[m+1].plot(result[m], 'b-')
        axes[m+1].set_title(f'IMF {m+1}')
        
    axes[mode+1].plot(residual[mode-1], 'b-')
    axes[mode+1].set_title('Residual')
    
    plt.tight_layout()
    plt.savefig(output_name, dpi=100)
    plt.close()

def plot_hsa(hsa_map_cpu, max_time, max_freq, output_name, width, height):
    plt.figure(figsize=(width/100.0, height/100.0))
    # hsa_map is (Time_cell, Freq_cell)
    # mathgl Dens plots (x, y), so we transpose
    # We add 1e-8 to avoid log(0)
    img = np.maximum(hsa_map_cpu.T, 1e-8)
    img = np.nan_to_num(img, nan=1e-8, posinf=1e-8, neginf=1e-8)
    
    vmin, vmax = get_color_scale(img)
    
    plt.imshow(img, extent=[0, max_time, 0, max_freq], origin='lower',
               aspect='auto', cmap='jet', norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    
    plt.title('Hilbert Spectra Analysis')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='log-scale')
    
    plt.savefig(output_name, dpi=100)
    plt.close()

def plot_hhs(hhs_map_cpu, max_fm_freq, max_am_freq, output_name, width, height):
    plt.figure(figsize=(width/100.0, height/100.0))
    img = np.maximum(hhs_map_cpu.T, 1e-8)
    img = np.nan_to_num(img, nan=1e-8, posinf=1e-8, neginf=1e-8)
    
    vmin, vmax = get_color_scale(img)
    
    plt.imshow(img, extent=[0, max_fm_freq, 0, max_am_freq], origin='lower',
               aspect='auto', cmap='jet', norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
               
    plt.title('Holo-Hilbert Spectra')
    plt.xlabel('FM Frequency')
    plt.ylabel('AM Frequency')
    plt.colorbar(label='log-scale')
    
    plt.savefig(output_name, dpi=100)
    plt.close()

def plot_hhsa_3d(hhsa_map_cpu, max_time, max_fm_freq, max_am_freq, output_name, 
                 dt, dFMF, dAMF, width, height, threshold=1.0):
    fig = plt.figure(figsize=(width/100.0, height/100.0))
    ax = fig.add_subplot(111, projection='3d')
    
    t_idx, fm_idx, am_idx = np.where(hhsa_map_cpu > threshold)
    vals = hhsa_map_cpu[t_idx, fm_idx, am_idx]
    
    t_vals = t_idx * dt
    fm_vals = fm_idx * dFMF
    am_vals = am_idx * dAMF
    
    if len(vals) > 0:
        vmax = np.percentile(vals, 99.5)
        vmin = max(np.min(vals), vmax / 10000.0)
        vals_clipped = np.clip(vals, vmin, vmax)
    else:
        vals_clipped = vals
        
    sc = ax.scatter(t_vals, fm_vals, am_vals, c=np.log10(vals_clipped), cmap='jet', s=1)
    
    ax.set_title('Holo-Hilbert Spectra Analysis 3D')
    ax.set_xlabel('Time')
    ax.set_ylabel('FM Frequency')
    ax.set_zlabel('AM Frequency')
    
    # Adjust view angle to match C++
    ax.view_init(elev=45, azim=45)
    plt.colorbar(sc, label='log-scale (10^n)')
    
    plt.savefig(output_name, dpi=100)
    plt.close()

def create_hhsa_gif(hhsa_map_cpu, time_cell, max_fm_freq, max_am_freq, output_name, dt, time_cell_group, width, height):
    # hhsa_map_cpu: (Time_cell, Freq_cell, Holo_Freq_cell)
    frames = []
    
    for k in tqdm(range(0, time_cell, time_cell_group), desc="Generate HHSA GIF"):
        img = np.full((hhsa_map_cpu.shape[1], hhsa_map_cpu.shape[2]), 1e-8)
        
        for ki in range(time_cell_group):
            if k + ki < time_cell:
                img += hhsa_map_cpu[k+ki]
                
        img = np.nan_to_num(img, nan=1e-8, posinf=1e-8, neginf=1e-8)
        img = img.T
        
        plt.figure(figsize=(width/100.0, height/100.0))
        vmin, vmax = get_color_scale(img)
        
        plt.imshow(img, extent=[0, max_fm_freq, 0, max_am_freq], origin='lower',
                   aspect='auto', cmap='jet', norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
        plt.title(f'Holo-Hilbert Spectra Analysis T={k*dt:.2f}')
        plt.xlabel('FM Frequency')
        plt.ylabel('AM Frequency')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        
        image = imageio.imread(buf)
        frames.append(image)
        
    with imageio.get_writer(output_name, mode='I', duration=0.1) as writer:
        for image in frames:
            writer.append_data(image)
