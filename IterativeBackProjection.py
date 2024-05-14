import time

import numpy as np

from matplotlib import pyplot as plt
from RichardsonLucyDeconvolution import *
from Stack import *
from SimulationIMG import *
from instrument import *


def repeate_stack(Prld, fine_grid, star_list, r):
    # 将Prld根据源位置插入到细网格中，再按照比例因子进行下采样和上采样，以模拟不同的像素位移
    # Prld是Ps经过去卷积算法RLD去卷积后的结果
    # fine_grid_shape是细网格形状
    # star_list是源位置列表
    # r是比例因子列表

    # 创建一个空的细网格，使用fine_grid_shape指定的形状
    Stack = np.zeros_like(Prld)     # .astype(np.float64)
    # 将细网格清零
    fine_grid = np.zeros_like(fine_grid)
    fine_grid = fine_grid.astype(np.float)

    # 将Prld根据点源坐标放置到细网格中
    for x, y, f in star_list:
        center = np.array((x, y))
        x, y = int(x), int(y)
        half_height = int(Prld.shape[0] / 2)
        half_wide = int(Prld.shape[1] / 2)
        if x - half_height > 0 and x + half_height + 1 < fine_grid.shape[0] and y - half_wide > 0 and \
                y + half_wide + 1 < fine_grid.shape[1]:
            # fine_grid[x - half_height:x + half_height + 1, y - half_wide:y + half_wide + 1] += Prld
            fine_grid = map_to_large_image(fine_grid, Prld, center)
    # 下采样上采样
    # down_sample_shape = (int(fine_grid.shape[0] / r), int(fine_grid.shape[1] / r))
    # down_sample_image = resample_image(fine_grid, down_sample_shape)
    down_sample_image = average_downsample(fine_grid, r)
    up_sample_image = superpixelize_image(down_sample_image, fine_grid.shape, r)   # .astype(np.uint8)
    up_sample_image[up_sample_image < 0] = 0

    # 重复堆叠步骤
    for star in star_list:
        # 返回以星源为中心的区域和星源外背景的平均值
        mask = cut_thumbnail(up_sample_image, star, Prld.shape[0])
        mask = suppress_offset(mask)
        add_to_stack(Stack, mask)
    PN_S = compute_stacked_PSF(Stack, len(star_list))   # .astype(np.uint8)
    return PN_S


def IBP(PS, r, fine_grid, star_list, iterative, iterative_rld, Ptrue):
    mask = circular_damping(PS.shape, 13, 0.1)
    PS = PS * mask

    PS_A = PS.copy()

    # 保存上一轮迭代结果
    P_ibp = PS.copy()
    last_var = 10

    for i in range(iterative):
        start_time = time.time()
        # 计算经过RLD去卷积算法，得到P_RLD
        print("IBP算法的第", i + 1, "迭代")
        print("执行RLD算法，迭代次数：", iterative_rld)

        # 调库RLD
        P_RLD = rld_lib(PS_A, r, iterative_rld)

        print("RLD算法结束，重复堆叠操作")

        # plt.title('P_RLD' + str(i))
        # plt.imshow(P_RLD)
        # plt.show()

        # 将P_RLD放入到细网格，进行整个堆叠操作
        PS_N = repeate_stack(P_RLD, fine_grid, star_list, r)

        # plt.title('PS_N' + str(i))
        # plt.imshow(PS_N)
        # plt.show()

        # 计算误差项
        error_term = PS_N - PS
        # 判断误差大小
        var = calculate_mse(PS_N, PS)
        smi = calculate_ssim(PS_N, PS)
        print("第", i + 1, "次迭代的均方差：", var)
        print("第", i + 1, "次迭代结构相似性指数：", smi)
        if var < 0.005 or var > last_var:
            print(f'最好迭代次数为{i+1}，均方差为：{last_var}')
            break
        last_var = var
        P_ibp = P_RLD
        # 更新PS_A
        PS_A = PS_A - error_term
        PS_A[PS_A < 0] = 0

        end_time = time.time()
        print(f'{end_time - start_time}秒')

    # 经过IBP算法得到更精细的PSF，返回
    return PS_A


def IBP_update_RLD(PS, r, fine_grid, star_list, iterative, iterative_rld):
    """
    IBP算法：使用RLD算法结果进行迭代
    :param PS:众多星源堆叠形成的堆叠PSF
    :param r:图像细化倍数
    :param fine_grid:欠采样图像细化后的图像
    :param star_list:星源列表
    :param iterative:IBP去卷积算法迭代次数
    :param iterative_rld:RLD去卷积算法迭代次数
    :return:
    """
    # 调库RLD
    P_RLD = rld_lib(PS, r, iterative_rld)

    # 保存上一轮迭代结果
    last_PS_N = PS.copy()

    for i in range(iterative):
        start_time = time.time()
        # 计算经过RLD去卷积算法，得到P_RLD
        print("IBP算法的第", i + 1, "迭代")
        print("P_RLD更新结束，重复堆叠操作")

        # 将P_RLD放入到细网格，进行整个堆叠操作
        PS_N = repeate_stack(P_RLD, fine_grid, star_list, r)

        # 计算误差项
        error_term = PS_N - PS
        # 判断误差大小
        var = calculate_mse(PS_N, last_PS_N)
        smi = calculate_ssim(PS_N, PS)
        print("第", i + 1, "次迭代结果与上一次结果的均方差：", var)
        print("第", i + 1, "次迭代结构相似性指数：", smi)
        if var < np.max(PS_N) * 0.0000001:
            return P_RLD

        last_PS_N = PS_N
        # 更新PSA
        P_RLD = P_RLD - error_term
        # P_RLD[P_RLD < 0] = 0
        minimum = np.min(P_RLD)
        if minimum < 0:
            P_RLD += np.abs(minimum)

        end_time = time.time()
        print(f'{end_time - start_time}秒')

    # 经过IBP算法得到更精细的PSF，返回
    return P_RLD


def IBP_end_RLD(PS, r, fine_grid, star_list, iterative, iterative_rld):
    """
    IBP算法：该算法得到的结果是一个还需经过RLD算法才可以接近Ptrue。所以返回的是PRLD，并非PSA。
    :param PS:众多星源堆叠形成的堆叠PSF
    :param r:图像细化倍数
    :param fine_grid:欠采样图像细化后的图像
    :param star_list:星源列表
    :param iterative:IBP去卷积算法迭代次数
    :param iterative_rld:RLD去卷积算法迭代次数
    :return:
    """
    # PS = PS / np.sum(PS)
    PS_A = PS.copy()

    # 保存上一轮迭代结果
    last_error = 100
    last_PIBP = np.zeros_like(PS)
    # 保存最后一次迭代的PRLD，用于返回
    P_ibp = np.zeros_like(PS)
    last_PS_N = PS.copy()

    for i in range(iterative):
        star_time = time.time()
        # 计算经过RLD去卷积算法，得到P_RLD
        print("IBP算法的第", i + 1, "迭代")
        print("执行RLD算法，迭代次数：", iterative_rld)

        # 调库RLD
        P_RLD = rld_lib(PS_A, r, iterative_rld)

        print("RLD算法结束，重复堆叠操作")

        # 将P_RLD放入到细网格，进行整个堆叠操作
        PS_N = repeate_stack(P_RLD, fine_grid, star_list, r)

        # 计算误差项
        error_term = PS_N - PS
        # 判断误差大小
        mse = calculate_mse(PS_N, last_PS_N)
        mse1 = calculate_mse(PS_N, PS)
        smi = calculate_ssim(PS_N, PS)
        print("第", i + 1, "次迭代结果与上一次结果的均方差：", mse)
        print("第", i + 1, "次迭代结果与Ps的均方差：", mse1)
        print("第", i + 1, "次迭代结构相似性指数：", smi)
        # if mse1 > last_error:
        #     return last_PIBP
        # if mse1 < np.max(PS_N) * 0.0000001:
        #     return P_RLD
        P_ibp = P_RLD
        last_PIBP = P_ibp
        last_PS_N = PS_N

        # 更新PSA
        PS_A = PS_A - error_term
        last_error = mse1
        minimum = np.min(PS_A)
        if minimum < 0:
            PS_A += np.abs(minimum)
        # PS_A[PS_A < 0] = 0
        # PS_A = (PS_A - np.min(PS_A)) / (np.max(PS_A) - np.min(PS_A))
        # PS_A = PS_A / np.sum(PS_A)

        end_time = time.time()
        print(f'{end_time - star_time}秒')

    # 经过IBP算法得到一个只需要经过RLD去卷积即可恢复Ptrue的标准，返回这个标准的RLD结果
    return P_ibp


def IBP_PSA(PS, r, fine_grid, star_list, iterative, iterative_rld):
    """
    IBP算法：误差项使用的是减去PSA
    :param PS:众多星源堆叠形成的堆叠PSF
    :param r:图像细化倍数
    :param fine_grid:欠采样图像细化后的图像
    :param star_list:星源列表
    :param iterative:IBP去卷积算法迭代次数
    :param iterative_rld:RLD去卷积算法迭代次数
    :return:
    """
    PS_A = PS.copy()

    for i in range(iterative):
        # 计算经过RLD去卷积算法，得到P_RLD
        print("IBP算法的第", i + 1, "迭代")
        print("执行RLD算法，迭代次数：", iterative_rld)

        # 调库RLD
        P_RLD = rld_lib(PS_A, r, iterative_rld)

        print("RLD算法结束，重复堆叠操作")

        # 将P_RLD放入到细网格，进行整个堆叠操作
        PS_N = repeate_stack(P_RLD, fine_grid, star_list, r)

        # 计算误差项
        error_term = PS_N - PS_A
        # 更新PSA
        PS_A = PS_A - error_term
        PS_A[PS_A < 0] = 0

    # 经过IBP算法得到一个只需要经过RLD去卷积即可恢复Ptrue的标准，返回这个标准的RLD结果
    return PS_A
