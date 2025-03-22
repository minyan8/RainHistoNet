import os
import cv2
import numpy as np
import pandas as pd
from skimage.util import img_as_ubyte
import utils  # Import your custom utils.py file

# Paths
GT_FOLDER = "/home/min/Documents/ntire25/data/my_val/GT"  # Change to your ground truth folder path
RESULT_FOLDER = "/home/min/Documents/ntire25/sysu_v2/Enhancement/my_val_res/NtireLL"  # Change to your result folder path

# Get sorted list of image filenames
gt_images = sorted(os.listdir(GT_FOLDER))
result_images = sorted(os.listdir(RESULT_FOLDER))

# Ensure both folders contain the same files
if gt_images != result_images:
    print("⚠️ Warning: Filenames do not match between ground truth and result folders!")

# Store results
psnr_values = []
ssim_values = []

# Loop through all images
for filename in gt_images:
    gt_path = os.path.join(GT_FOLDER, filename)
    result_path = os.path.join(RESULT_FOLDER, filename)

    # Read images
    gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    result_img = cv2.imread(result_path, cv2.IMREAD_COLOR)

    # if gt_img is None or result_img is None:
    #     print(f"❌ Skipping {filename} (could not read)")
    #     continue

    # # Convert to grayscale for SSIM (if necessary)
    # gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    # result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)

    # Compute PSNR & SSIM using your specified methods
    psnr_value = utils.PSNR(gt_img, result_img)
    ssim_value = utils.calculate_ssim(img_as_ubyte(gt_img), img_as_ubyte(result_img))

    # Store results
    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)

# Compute mean PSNR and SSIM
mean_psnr = np.mean(np.array(psnr_values))
mean_ssim = np.mean(np.array(ssim_values))


# Print results
print(f"\n📊 Mean PSNR: {mean_psnr:.4f}")
print(f"📊 Mean SSIM: {mean_ssim:.4f}")
