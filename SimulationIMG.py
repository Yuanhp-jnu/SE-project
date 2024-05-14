# import numpy as np
# import cv2
# from astropy.io import fits
# from astropy.modeling.functional_models import Gaussian2D
# from astropy.table import Table
# from astropy.wcs import WCS
# from scipy import ndimage
#
# #
# # # 定义高斯函数
# # def gaussian(x, y, center_x, center_y, sigma):
# #     return np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
#
#
# # 定义图像大小
# height = 2048
# width = 2048
#
# # 定义高斯点源数量和标准差
# num_points = 100
# sigma = 2
#
# wcs = WCS(naxis=2)
#
# # 随机生成坐标
# np.random.seed(0)  # 设置随机种子，保证每次运行结果一致
# center_x = np.random.uniform(low=0, high=width, size=num_points)
# center_y = np.random.uniform(low=0, high=height, size=num_points)
#
# # 保存像素坐标，考虑删掉
# # ra_coords, dec_coords = wcs.all_pix2world(center_x, center_y, 0)
# source_table = Table([center_x, center_y], names=['x', 'y'])
#
# # 生成网格坐标
# x, y = np.meshgrid(np.arange(width), np.arange(height))
#
# # 初始化高斯点源图像(整数)
# gaussian_map = np.zeros((height, width))
#
# # 逐个计算并叠加高斯函数值，亮度55-255，且为整数，有溢出>255，一般不要，记录
# for i in range(num_points):
#     # gaussian_map += np.round(255 * gaussian(x, y, center_x[i], center_y[i], sigma))
#     gauss = Gaussian2D(255, center_x[i], center_y[i], sigma, sigma)
#     gaussian_map += gauss(*np.indices(gaussian_map.shape))
#
# # 对图像进行归一化，使其像素值在0到255之间
# gaussian_map = (gaussian_map - np.min(gaussian_map)) / (np.max(gaussian_map) - np.min(gaussian_map)) * 255
#
# # 将浮点像素值转换为整数
# gaussian_map = gaussian_map.astype(np.uint8)
#
# # 归一化处理（不应该）
# # gaussian_map = gaussian_map / np.max(gaussian_map)
#
# wcs.wcs.ctype = ["x---TAN", "y--TAN"]
#
# hdu_image = fits.PrimaryHDU(gaussian_map, header=wcs.to_header())
# hdu_table = fits.BinTableHDU(source_table)
# hdul = fits.HDUList([hdu_image, hdu_table])
# hdul.writeto('gaussian2.fits', overwrite=True)
#
# # 将高斯点源图像保存为PNG格式的图片
# cv2.imwrite('picture/gaussian2.png', gaussian_map)
# cv2.imshow('Gaussian', gaussian_map)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import time

import numpy as np
import cv2
from PIL import Image
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian2D
from astropy.table import Table
from astropy.wcs import WCS
from numpy import random
from scipy.signal import convolve2d
from astropy.modeling import models, fitting
from astropy.convolution import convolve, CustomKernel

from instrument import *

from IterativeBackProjection import map_to_large_image


def calculate_flux(mAB, SNR):
    flux_mAB = 10 ** ((mAB - 22.5) / -2.5)
    Fn = flux_mAB / np.sqrt(SNR)
    return Fn


def simulation_image(mask_size, r):
    # 定义图像大小
    height = 5120
    width = 5120

    # 定义高斯点源数量和标准差
    num_points = 1000
    sigma = 5

    # 模拟图像参数
    psf_size = mask_size
    SNR = 20
    mAB = 17.87
    downsample_size = r

    # Fn = calculate_flux(mAB, SNR)

    wcs = WCS(naxis=2)

    # 随机生成超分辨率图像中的星源坐标
    np.random.seed(0)  # 设置随机种子，保证每次运行结果一致
    # 生成1000个随机坐标，并且坐标保留两位小数
    center_x = np.round(np.random.uniform(low=0, high=width, size=num_points), 2)
    center_y = np.round(np.random.uniform(low=0, high=height, size=num_points), 2)
    # 星源通量随机
    # flux = np.round(np.random.uniform(100, 255, num_points), 3)
    # 所有星源都是统一通量
    Fn = round(random.uniform(100, 3500), 3)
    print(Fn)
    flux = np.full(num_points, Fn)

    # 计算下采样后的星源中心坐标
    x = center_x / downsample_size
    y = center_y / downsample_size

    # 用于保存星源坐标
    source_table = Table([x, y], names=['x', 'y'])

    # 初始化高斯点源图像
    simulated_image = np.full((height, width), 0.0)

    # 生成真实的PSF（高斯图像）
    # Gaussian2D(峰值，x均值，y均值)
    gauss = Gaussian2D(1, np.ceil(psf_size / 2) - 1, np.ceil(psf_size / 2) - 1, sigma, sigma)
    indices = np.indices((psf_size, psf_size))
    psf = gauss(*indices)
    psf = psf / np.sum(psf)

    # # 定义Moffat PSF参数
    # moffat_psf = models.Moffat2D(amplitude=1, x_0=0, y_0=0, gamma=2, alpha=2.5)
    #
    # # 生成Moffat PSF卷积核的网格
    # x, y = np.mgrid[-25:26, -25:26]  # 生成一个以点源为中心的网格，大小要足以包含大部分PSF能量
    # psf = moffat_psf(x, y)  # 使用Moffat PSF模型生成PSF图像（二维数组）
    #
    # # 将PSF数据转换为Convolution Kernel
    # psf_kernel = CustomKernel(psf_data)

    start_time = time.time()

    # 在每个星源坐标中心放置一个psf
    # for i in range(num_points):
    #     # 将PSF的值放大到0~Fn范围
    #     # psf1 = (psf - np.min(psf)) * (1 / (np.max(psf) - np.min(psf)))
    #     psf1 = flux[i] * (psf / np.sum(psf))
    #     Fn = np.sum(psf1)
    #     center = np.array((center_x[i], center_y[i]))
    #     simulated_image = map_to_large_image(simulated_image, psf1, center)

    for i in range(num_points):
        point = np.zeros((1, 1))
        point[0][0] = flux[i]
        center = np.array((center_x[i], center_y[i]))
        simulated_image = map_to_large_image(simulated_image, point, center)

    simulated_image = convolve2d(simulated_image, psf, mode='same')

    end_time = time.time()
    print(f"{end_time-start_time}秒")

    # 下采样图像，获取欠采样图像
    down_sampled_image = average_downsample(simulated_image, downsample_size)   # .astype(np.uint8)

    # 添加噪声
    down_sampled_image_gauss = gauss_noisy(down_sampled_image, 0, 1)
    down_sampled_image_poisson = add_poisson_noise(down_sampled_image, 3)

    # 保存模拟图像
    cv2.imwrite('picture/simulated_image.png', simulated_image)

    wcs.wcs.ctype = ["x---TAN", "y--TAN"]
    hdu_image = fits.PrimaryHDU(down_sampled_image, header=wcs.to_header())
    hdu_table = fits.BinTableHDU(source_table)
    hdul = fits.HDUList([hdu_image, hdu_table])
    hdul.writeto('gaussian.fits', overwrite=True)

    wcs.wcs.ctype = ["x---TAN", "y--TAN"]
    hdu_image = fits.PrimaryHDU(down_sampled_image_gauss, header=wcs.to_header())
    hdu_table = fits.BinTableHDU(source_table)
    hdul = fits.HDUList([hdu_image, hdu_table])
    hdul.writeto('gaussian_noise.fits', overwrite=True)
    wcs.wcs.ctype = ["x---TAN", "y--TAN"]

    wcs.wcs.ctype = ["x---TAN", "y--TAN"]
    hdu_image = fits.PrimaryHDU(down_sampled_image_poisson, header=wcs.to_header())
    hdu_table = fits.BinTableHDU(source_table)
    hdul = fits.HDUList([hdu_image, hdu_table])
    hdul.writeto('poisson_noise.fits', overwrite=True)

    # 将高斯点源图像保存为PNG格式的图片
    cv2.imwrite('picture/gaussian.png', down_sampled_image)
    cv2.imwrite('picture/gaussian_gauss_noise.png', down_sampled_image_gauss)
    cv2.imwrite('picture/gaussian_poisson_noise.png', down_sampled_image_poisson)

    # 返回模拟的真实的点扩散函数
    return psf, flux
