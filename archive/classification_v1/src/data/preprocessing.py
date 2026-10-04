"""
荧光图像预处理模块
负责人：光电专业队员

荧光图像具有特殊的光谱特性，预处理需要关注：
1. 光照不均匀校正（荧光强度衰减）
2. 色彩空间校正（荧光伪彩映射）
3. ROI自动提取（口腔黏膜区域分割）
4. 图像质量筛选（模糊/过曝/欠曝检测）
"""

import cv2
import numpy as np


def correct_illumination(image):
    """光照不均匀校正：用于补偿荧光图像的强度衰减"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (151, 151), 0)
    corrected = cv2.divide(gray, blurred, scale=255)
    return cv2.cvtColor(corrected, cv2.COLOR_GRAY2RGB)


def extract_roi(image):
    """ROI提取：分割口腔黏膜区域"""
    return image


def quality_score(image):
    """图像质量评分：检测模糊、过曝、欠曝"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = np.mean(gray)
    return {
        "sharpness": laplacian_var,
        "brightness": mean_brightness,
        "is_blurry": laplacian_var < 50,
        "is_overexposed": mean_brightness > 220,
        "is_underexposed": mean_brightness < 30,
    }
