import math
import os
import pickle

import numpy as np

from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from IterativeBackProjection import *
from SimulationIMG import *


def arithmetic(image, star_list, radius, factor, method):
    # 保存合法点源
    valid_star = []
    print("原始源个数： ", len(star_list), '\n', '判断拥挤源...')

    # 判断拥挤源
    for i in range(0, len(star_list)):
        is_valid = True
        for j in range(0, len(star_list)):
            if i != j:
                # 判断x轴、y轴相差是否大于指定距离，如果有一个大于指定距离，就不用计算两点距离，减少运行成本
                if np.abs(star_list[i][0] - star_list[j][0]) < d and np.abs(star_list[i][1] - star_list[j][1] < d):
                    if distance(star_list[i], star_list[j]) < d:
                        is_valid = False
                        break
        if is_valid:
            valid_star.append(star_list[i])

    print("剩下源个数: ", len(valid_star))
    print("开始堆叠...")
    PS, new_star_list, resize_img, T_list = stack_star(image, r, valid_star, mask_size=mask_size)
    print("Ps成功创建")
    # plt.title('PS')
    # plt.imshow(PS)
    # plt.show()
    mask = circular_damping(PS.shape, radius, factor)
    PS = PS * mask
    P_ibp = np.zeros_like(PS)

    ''' 去卷积算法三选一 '''
    if method == 0:
        print("执行RLD算法...")
        P_ibp = rld_lib(PS, r, iterative_rld)
    elif method == 1:
        print("执行IBP_update算法...")
        P_ibp = IBP_update_RLD(PS, r, resize_img, new_star_list, iterative=iterative, iterative_rld=iterative_rld)
    elif method == 2:
        print("执行IBP_end算法...")
        P_ibp = IBP_end_RLD(PS, r, resize_img, new_star_list, iterative=iterative,
                            iterative_rld=iterative_rld)
    # P_ibp = IBP(PS, r, resize_img, new_star_list, iterative=iterative, iterative_rld=iterative_rld, Ptrue=P_true)

    F = 0

    # 求通量
    for i in range(len(new_star_list)):
        flux = calculate_optimal_photometry(P_ibp, image, new_star_list[i], r)
        F += (np.abs(flux - new_star_list[i][2])) / new_star_list[i][2]
        print(tuple(elem / r for elem in new_star_list[i]), '真实通量: ', new_star_list[i][2], ',  测试通量：', flux,
              ',  误差：', "{:.2f}%".format(100 * (np.abs(new_star_list[i][2] - flux))/new_star_list[i][2]))

    F /= len(new_star_list)
    print("F: {:.3f}%".format(F*100))
    return P_ibp, PS


def analog_image():
    """
    使用模拟图像，有真实的PSF作为评价标准
    :return:
    """
    # 生成模拟图像
    print("开始生成模拟图像...")
    P_true, true_flux = simulation_image(mask_size=mask_size, r=r)
    print("模拟图像生成成功！")

    image_name = 'gaussian_noise.fits'
    # 打开FITS文件并获取点源坐标表格
    image_star = fits.open(image_name)
    source_table = Table.read(image_star[1])
    # 提取点源坐标
    ra_cords = source_table['x']
    dec_cords = source_table['y']
    star_list = list(zip(ra_cords, dec_cords, true_flux))

    # 单张图片测试，需要返回Ps
    image = fits.getdata(image_name)
    true = P_true / np.sum(P_true)
    P1, P2 = arithmetic(image, star_list, 40, 0, 2)
    P1 = P1 / np.sum(P1)
    P2 = P2 / np.sum(P2)
    err = P1 - true

    print("相似度：", calculate_ssim(P1, true))

    true = {'P_ture': true}
    rld = {'P_ibp': P1}
    ibp_update = {'Ps': P2}
    err = {'ERROR': err}

    chart(true, rld, ibp_update, err)

    P_ibp = P1
    # 归一化
    P_ibp = P_ibp / np.sum(P_ibp)
    p_true = P_true / np.sum(P_true)
    # 计算预估值和真实值的差值
    subtraction = P_ibp - p_true

    subtraction_abs = np.abs(subtraction)
    max_value = np.max(subtraction_abs)
    max_coordinates = find_max_coordinates(subtraction_abs)
    max_error_proportion = 0
    for i in max_coordinates:
        true_value = p_true[i[0], i[1]]
        error_proportion = max_value / true_value
        if error_proportion > max_error_proportion:
            max_error_proportion = error_proportion
    print(f"最大差值： {max_value}")
    print(f"最大误差： {max_error_proportion:.3%}")


def real_image():
    """
    使用真实的cassin图像，没有真实的PSF作为评价标准
    :return:
    """
    Fits_dir = 'FITS'
    Qmpf_dir = 'QMPF'

    # 遍历源目录下的所有文件
    for fits_file_name in os.listdir(Fits_dir):
        # 检查文件扩展名是否为.fits
        if fits_file_name.endswith('.fits'):
            # 构建源文件的完整路径
            source_file_path = os.path.join(Fits_dir, fits_file_name)
            print(source_file_path)

            # 找到最后一个下划线的索引
            last_underscore_index = fits_file_name.rfind('_')
            # 如果最后一个下划线存在，去掉它以及其后面的所有字符
            if last_underscore_index != -1:
                fits_file_name = fits_file_name[:last_underscore_index]

            # 构建目标文件的完整路径，假设目标文件扩展名为.first
            target_file_path = os.path.join(Qmpf_dir, f"{fits_file_name}.QMPF")
            # 检查目标文件是否存在
            if os.path.isfile(target_file_path):
                print(target_file_path)
                # 读取fits图像文件
                image = fits.getdata(source_file_path)
                # 获取星源中心坐标
                star_list = get_star_list(target_file_path)
                # 读取的值太小，将其扩大
                image = (image - np.min(image)) * (4000 / np.max(image) - np.min(image))

                for star in star_list:
                    x = int(star[0])
                    y = int(star[1])
                    img = np.full((9, 9), 0).astype(np.float64)
                    img = map_to_small_image(image, img, (x + 0.5, y + 0.5))
                    x0, y0 = moment_method(img)
                    # x0 = x - 4 + x0
                    # y0 = y - 4 + y0
                    print(f"原坐标：({star[0]}, {star[1]}); 矩方法坐标：（{x0}, {y0}）")

                P_ibp, Ps = arithmetic(image, star_list, 15, 0, 2)
                P_ibp = P_ibp / np.sum(P_ibp)
                Ps = Ps / np.sum(Ps)
                rld = {'P_ibp': P_ibp}
                Ps = {'Ps': Ps}
                chart(rld, Ps)
                plt.title('Pibp')
                plt.imshow(P_ibp)
                plt.show()

                # 保存新建的PSF
                with open('psf_model.pkl', 'wb') as f:
                    pickle.dump(P_ibp, f)
                break


if __name__ == '__main__':
    r = 10  # 细化倍数
    d = 5  # 判断拥挤源的像素距离
    mask_size = 50  # 截取区域像素尺寸，6*sigma，正负3sigma，+1使得中心只有一个像素
    iterative = 15  # IBP算法迭代次数
    iterative_rld = 15  # RLD算法迭代次数

    # pattern为0使用模拟图像，为1使用Cassini图像
    pattern = 0

    if pattern == 0:
        analog_image()
    else:
        real_image()
