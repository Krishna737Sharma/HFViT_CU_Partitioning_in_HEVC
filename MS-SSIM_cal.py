import os
import sys
import numpy as np
import torch
import math
from piq import multi_scale_ssim

# ================= Configuration =================
WIDTH = 3840
HEIGHT = 2160
ORIGINAL_FILE = '/root/myproject/HEVC-CNN/test_video/Scarf.yuv'
RECON_FILE = '/root/myproject/HEVC_Intra_Models-ViT/HM-16.5-official/bin/rec_baseline.yuv'  # This is the output file from your summary
FRAMES_TO_TEST = 253         # Your summary indicates only 2 frames were encoded
# =================================================

def get_yuv_frame_size(width, height):
    """Calculate total frame size for YUV 4:2:0 format"""
    y_size = width * height
    u_size = (width // 2) * (height // 2)
    v_size = (width // 2) * (height // 2)
    return y_size + u_size + v_size

def read_y_frame(f, width, height):
    """
    Reads just the Y (Luminance) component from a YUV 4:2:0 file.
    Returns None if end of file or read error.
    """
    y_size = width * height
    uv_size = (width // 2) * (height // 2) * 2
    
    # Read Y data
    y_buf = f.read(y_size)
    if not y_buf or len(y_buf) < y_size:
        return None
    
    # Skip U and V data
    uv_buf = f.read(uv_size)
    if len(uv_buf) < uv_size:
        print(f"Warning: Incomplete UV data read (expected {uv_size}, got {len(uv_buf)})")
    
    # Convert to numpy array
    try:
        # FIX: Add .copy() to make the array writable and avoid warning
        data = np.frombuffer(y_buf, dtype=np.uint8).copy().reshape(height, width)
        return data
    except ValueError as e:
        print(f"Error reshaping Y buffer: {e}")
        return None

def verify_file_size(filepath, width, height, expected_frames):
    """Verify if file has enough data for expected frames"""
    if not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"
    
    file_size = os.path.getsize(filepath)
    frame_size = get_yuv_frame_size(width, height)
    actual_frames = file_size // frame_size
    
    if file_size == 0:
        return False, f"File is empty: {filepath}"
    
    if actual_frames < expected_frames:
        return False, f"File only contains {actual_frames} frames, but {expected_frames} expected"
    
    return True, f"File contains {actual_frames} frames"

def calculate_metrics():
    print("=" * 60)
    print("YUV MS-SSIM Calculator")
    print("=" * 60)
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Reference: {ORIGINAL_FILE}")
    print(f"Distorted: {RECON_FILE}")
    print(f"Frames to test: {FRAMES_TO_TEST}")
    print("=" * 60)
    
    # Verify files exist and have correct size
    print("\nVerifying files...")
    valid, msg = verify_file_size(ORIGINAL_FILE, WIDTH, HEIGHT, FRAMES_TO_TEST)
    print(f"Original: {msg}")
    if not valid:
        print("ERROR: Original file validation failed!")
        return
    
    valid, msg = verify_file_size(RECON_FILE, WIDTH, HEIGHT, FRAMES_TO_TEST)
    print(f"Reconstructed: {msg}")
    if not valid:
        print("ERROR: Reconstructed file validation failed!")
        return
    
    # Open files
    try:
        f_orig = open(ORIGINAL_FILE, 'rb')
        f_recon = open(RECON_FILE, 'rb')
    except FileNotFoundError as e:
        print(f"Error opening files: {e}")
        return
    except PermissionError as e:
        print(f"Permission denied: {e}")
        return
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\nProcessing frames...")
    print("-" * 60)
    
    ms_ssim_scores = []
    frames_processed = 0
    
    for i in range(FRAMES_TO_TEST):
        # 1. Read Frames
        y_orig = read_y_frame(f_orig, WIDTH, HEIGHT)
        y_recon = read_y_frame(f_recon, WIDTH, HEIGHT)
        
        if y_orig is None:
            print(f"Warning: Could not read frame {i} from original file")
            break
        
        if y_recon is None:
            print(f"Warning: Could not read frame {i} from reconstructed file")
            break
        
        # Verify data integrity
        if y_orig.shape != (HEIGHT, WIDTH):
            print(f"Error: Original frame {i} has wrong shape: {y_orig.shape}")
            break
        
        if y_recon.shape != (HEIGHT, WIDTH):
            print(f"Error: Reconstructed frame {i} has wrong shape: {y_recon.shape}")
            break
        
        # 2. Preprocess for PIQ (Convert to Tensor, Normalize 0-1, Add Batch Dim)
        try:
            t_orig = torch.from_numpy(y_orig).float() / 255.0
            t_recon = torch.from_numpy(y_recon).float() / 255.0
            
            # Add channel and batch dimensions: (1, 1, Height, Width)
            t_orig = t_orig.unsqueeze(0).unsqueeze(0).to(device)
            t_recon = t_recon.unsqueeze(0).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error converting frame {i} to tensor: {e}")
            break
        
        # 3. Calculate MS-SSIM
        try:
            # data_range=1.0 because we normalized to [0, 1]
            score = multi_scale_ssim(t_recon, t_orig, data_range=1.0)
            score_value = score.item()
            ms_ssim_scores.append(score_value)
            
            print(f"Frame {i:3d}: MS-SSIM = {score_value:.6f}")
            frames_processed += 1
            
        except Exception as e:
            print(f"Error calculating MS-SSIM for frame {i}: {e}")
            print(f"  Original shape: {t_orig.shape}")
            print(f"  Reconstructed shape: {t_recon.shape}")
            break
    
    # Close files
    f_orig.close()
    f_recon.close()
    
    # Final Results
    print("-" * 60)
    print("\nResults:")
    print("=" * 60)
    
    if ms_ssim_scores:
        avg_ms_ssim = sum(ms_ssim_scores) / len(ms_ssim_scores)
        min_ms_ssim = min(ms_ssim_scores)
        max_ms_ssim = max(ms_ssim_scores)
        
        print(f"Frames processed: {frames_processed}/{FRAMES_TO_TEST}")
        print(f"Average MS-SSIM:  {avg_ms_ssim:.6f}")
        print(f"Min MS-SSIM:      {min_ms_ssim:.6f}")
        print(f"Max MS-SSIM:      {max_ms_ssim:.6f}")
        
        if avg_ms_ssim > 0.99:
            print("\n✓ Excellent quality (MS-SSIM > 0.99)")
        elif avg_ms_ssim > 0.95:
            print("\n✓ Very good quality (MS-SSIM > 0.95)")
        elif avg_ms_ssim > 0.90:
            print("\n✓ Good quality (MS-SSIM > 0.90)")
        else:
            print("\n⚠ Quality may need improvement (MS-SSIM < 0.90)")
    else:
        print("ERROR: No frames were successfully processed!")
        print("\nPossible issues:")
        print("  1. Files may be corrupted or in wrong format")
        print("  2. Resolution settings may be incorrect")
        print("  3. Files may not be YUV 4:2:0 format")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        calculate_metrics()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)