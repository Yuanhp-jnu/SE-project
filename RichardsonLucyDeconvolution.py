import math

import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import convolve
from skimage import restoration
from instrument import *


def create_Pgrid(r, PSF):
    # 初始Pgrid
    Pgrid = np.zeros_like(PSF)
    # 计算PSF中心位置
    x0 = np.ceil(PSF.shape[0] / 2) - 1
    y0 = np.ceil(PSF.shape[1] / 2) - 1
    # 计算Pgrid，在距离中心的(-r, +r)的范围就计算，其他全部设为0
    for i in range(int(x0 - r), int(x0 + r + 1)):
        for j in range(int(y0 - r), int(y0 + r + 1)):
            Pgrid[i, j] = (r - np.abs(i - x0)) * (r - np.abs(j - y0))

    Pgrid[Pgrid < 0] = 0

    # （调库的RLD里会实现该翻转）计算Pgrid在水平、竖直方向的翻转的平均（Pgrid在两个轴的反射）
    # Pgrid_ref = np.flip(Pgrid)

    return Pgrid


def RLD(PSF, r, iterations):
    # 初始P_rld为堆叠后的PSF，即Ps
    P_rld = np.ones_like(PSF)

    Pgrid, Pgrid_ref = create_Pgrid(r, PSF)

    for _ in range(iterations):
        # 计算更新因子
        blurred_Psf = convolve(P_rld, Pgrid, 'same')
        # 计算相关
        factor = PSF / blurred_Psf
        P_rld *= convolve(factor, Pgrid_ref, 'same')
    return P_rld


# 调库RLD
def rld_lib(PSF, r, iterations):
    # 生成模糊因子
    Pgrid = create_Pgrid(r, PSF)
    # 使用模糊因子Pgrid对PSF进去RLD去卷积
    P_rld = restoration.richardson_lucy(PSF, Pgrid, num_iter=iterations, clip=False)
    return P_rld


def advanced_RLD(Ps, r, iteration):
    # 获取Pgrid
    Pgrid = create_Pgrid(r, Ps)

    P_rld = Ps.copy()

    # 圆形阻尼
    mask = circular_damping(P_rld.shape, 15, 0.1)

    for i in range(iteration):
        # RLD 反卷积
        deconv_image = restoration.richardson_lucy(P_rld, Pgrid, num_iter=1, clip=False)

        # 更新结果
        P_rld = mask * deconv_image

    return P_rld
