import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import math
from numpy import pi
import cv2
from PIL import Image
import os

# === STEP 1: Enhanced Image Loading (TIFF, PNG, JPG, ASCII) ===
def load_image():
    tiff_files = ["Lenna_512_grey.tiff"]
    ascii_files = ["lincon.txt", "ascii_image.txt", "image.txt"]

    for tiff_file in tiff_files:
        if os.path.exists(tiff_file):
            img = Image.open(tiff_file)
            if img.mode != 'L':
                img = img.convert('L')
            print(f"Loaded image: {tiff_file}, shape: {img.size}")
            return np.array(img, dtype=np.uint8), f"TIFF: {tiff_file}"

    for ascii_file in ascii_files:
        if os.path.exists(ascii_file):
            with open(ascii_file, 'r') as f:
                ascii_lines = [line.strip() for line in f if line.strip()]
            original_img = np.array([[ord(c) for c in line] for line in ascii_lines], dtype=np.uint8)
            print(f"Loaded ASCII image: {ascii_file}, shape: {original_img.shape}")
            return original_img, f"ASCII: {ascii_file}"

    print("Using default generated pattern (64x64)")
    return np.tile(np.arange(64, dtype=np.uint8), (64, 1)), "Default: Generated pattern"

# === STEP 2: Downscaling ===
def downscale_to_64x64(img):
    if img.shape == (64, 64):
        print("Image already 64x64")
        return img.copy()
    zoom_factors = (64 / img.shape[0], 64 / img.shape[1])
    downscaled = zoom(img, zoom=zoom_factors, order=1).astype(np.uint8)
    print(f"Downscaled to: {downscaled.shape}")
    return downscaled

# === STEP 3: R Kernels ===
def R_kernel(u, order):
    u = abs(u)
    if order == 3:
        if u <= 1:
            return 1.5 * u**3 - 2.5 * u**2 + 1
        elif 1 < u < 2:
            return -0.5 * u**3 + 2.5 * u**2 - 4 * u + 2
        return 0
    elif order == 5:
        if u <= 1:
            return 1 - 2.5 * u**2 + 1.5 * u**3
        elif 1 < u <= 2:
            return -0.5 * u**3 + 2.5 * u**2 - 4 * u + 2
        elif 2 < u <= 3:
            return 0.166667 * u**3 - 0.833333 * u**2 + 1.333333 * u - 0.666667
        return 0
    elif order == 7:
        if u <= 1:
            return 1 - 3.5 * u**2 + 2.5 * u**3
        elif 1 < u <= 2:
            return -0.5 * u**3 + 2.5 * u**2 - 4 * u + 2
        elif 2 < u <= 3:
            return 0.166667 * u**3 - 0.833333 * u**2 + 1.333333 * u - 0.666667
        elif 3 < u <= 4:
            return -0.02 * u**3 + 0.1 * u**2 - 0.15 * u + 0.07
        return 0
    elif order == 9:
        if u <= 1:
            return 1 - 4.5 * u**2 + 3.5 * u**3
        elif 1 < u <= 2:
            return -0.5 * u**3 + 2.5 * u**2 - 4 * u + 2
        elif 2 < u <= 3:
            return 0.166667 * u**3 - 0.833333 * u**2 + 1.333333 * u - 0.666667
        elif 3 < u <= 4:
            return -0.02 * u**3 + 0.1 * u**2 - 0.15 * u + 0.07
        elif 4 < u <= 5:
            return 0.001 * u**3 - 0.005 * u**2 + 0.008 * u - 0.004
        return 0
    return 0

# === Interpolation Function ===
def interpolate_with_kernel(img, scale_factor, kernel_order, kernel_size):
    old_height, old_width = img.shape
    new_height = int(old_height * scale_factor)
    new_width = int(old_width * scale_factor)
    result = np.zeros((new_height, new_width), dtype=np.float64)
    half_k = kernel_size // 2
    padded_img = np.pad(img, pad_width=half_k, mode='reflect')

    for y_new in range(new_height):
        for x_new in range(new_width):
            x_old = x_new / scale_factor
            y_old = y_new / scale_factor
            x_int = int(x_old)
            y_int = int(y_old)
            interpolated_value = 0.0
            weight_sum = 0.0

            for j in range(-half_k, half_k + 1):
                for i in range(-half_k, half_k + 1):
                    src_x = x_int + i
                    src_y = y_int + j
                    pad_x = src_x + half_k
                    pad_y = src_y + half_k
                    dx = x_old - src_x
                    dy = y_old - src_y
                    weight_x = R_kernel(dx, kernel_order)
                    weight_y = R_kernel(dy, kernel_order)
                    weight = weight_x * weight_y
                    interpolated_value += weight * padded_img[pad_y, pad_x]
                    weight_sum += weight

            result[y_new, x_new] = interpolated_value / weight_sum if weight_sum > 0 else 0
    return np.clip(result, 0, 255).astype(np.uint8)

# === Bilinear Interpolation ===
def bilinear_interpolation(img, scale_factor):
    old_height, old_width = img.shape
    new_height = int(old_height * scale_factor)
    new_width = int(old_width * scale_factor)
    result = np.zeros((new_height, new_width), dtype=np.float64)
    for y_new in range(new_height):
        for x_new in range(new_width):
            x_old = x_new / scale_factor
            y_old = y_new / scale_factor
            x_int = int(x_old)
            y_int = int(y_old)
            dx = x_old - x_int
            dy = y_old - y_int
            for j in range(2):
                for i in range(2):
                    src_x = x_int + i
                    src_y = y_int + j
                    if 0 <= src_x < old_width and 0 <= src_y < old_height:
                        R_x = (1 - dx) if i == 0 else dx
                        R_y = (1 - dy) if j == 0 else dy
                        result[y_new, x_new] += img[src_y, src_x] * R_x * R_y
    return np.clip(result, 0, 255).astype(np.uint8)

# === Apply All Methods ===
def upscale_with_six_methods(img_64x64):
    methods = {}
    scale_factor = 2.0
    methods['Nearest Neighbor'] = zoom(img_64x64, zoom=scale_factor, order=0).astype(np.uint8)
    methods['Bilinear'] = bilinear_interpolation(img_64x64, scale_factor)
    methods['Bicubic'] = interpolate_with_kernel(img_64x64, scale_factor, 3, 4)
    methods['Biquintic'] = interpolate_with_kernel(img_64x64, scale_factor, 5, 5)
    methods['Biseptic'] = interpolate_with_kernel(img_64x64, scale_factor, 7, 7)
    methods['Binonic'] = interpolate_with_kernel(img_64x64, scale_factor, 9, 9)
    print("Interpolation completed using six methods.")
    return methods

# === MAIN RUN ===
if __name__ == "__main__":
    original, src_info = load_image()
    img64 = downscale_to_64x64(original)
    methods = upscale_with_six_methods(img64)
    for name, img in methods.items():
        print(f"{name} upscaled image shape: {img.shape}")
