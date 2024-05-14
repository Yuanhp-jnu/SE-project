import math
import os

import cv2
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np


def distance(point1, point2):
    """
    计算星源点1中心和星源点2中心之间的距离
    :param point1:
    :param point2:
    :return:
    """
    x1, y1, f1 = point1
    x2, y2, f2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_mse(image1, image2):
    """
    计算image1和image2的均方误差
    :param image1:
    :param image2:
    :return:
    """
    err = np.sum((image1 - image2) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err


def calculate_ssim(image1, image2):
    """
    计算image1和image2的平均结构相似指数
    :param image1:
    :param image2:
    :return:
    """
    return ssim(image1, image2)


def circular_damping(size, r, value=0.0):
    """
    返回一个大小为（size，size），中心一个半径为r的区域没有阻尼，其余区域乘于阻尼值value
    :param size:整个图像大小
    :param r: 距离中心r个像素的区域没有阻尼
    :param value: 除中间圆形没有阻尼区域外，其他区域阻尼大小value
    :return:
    """
    # 圆形阻尼
    mask = np.full(size, value)
    for i in range(size[0]):
        for j in range(size[1]):
            if math.sqrt((i - int(size[0] / 2)) ** 2 + (j - int(size[1] / 2)) ** 2) < r:
                mask[i][j] = 1
    return mask


def center_circular_damping(size, r):
    """
    返回一个大小为（size，size），中心一个半径为r的圆形阻尼，其余区域没有阻尼
    :param size:整个图像大小
    :param r: 距离中心r个像素的区域没有阻尼
    :return:
    """
    # 圆形阻尼
    mask = np.full(size, 1)
    for i in range(size[0]):
        for j in range(size[1]):
            if math.sqrt((i - int(size[0] / 2)) ** 2 + (j - int(size[1] / 2)) ** 2) < r:
                mask[i][j] = 0
    return mask


def map_to_small_image(large_image, small_image, alignment_point):
    """
    将大尺寸图像的指定对齐坐标alignment_point和小尺寸图像的中心对齐
    将大尺寸图像的部分映射到小尺寸图像上
    :param large_image:
    :param small_image:
    :param alignment_point:
    :return: large_image
    """
    # 小尺寸图像的中心
    small_center = np.array(small_image.shape) // 2

    if small_image.shape[0] % 2 == 0:
        small_center[0] -= 1
    if small_image.shape[1] % 2 == 0:
        small_center[1] -= 1

    # 偏差
    alignment_point = tuple(elem - 0.5 for elem in alignment_point)
    bias_x = alignment_point[0] - np.floor(alignment_point[0])
    bias_y = alignment_point[1] - np.floor(alignment_point[1])

    # 计算每个像素的贡献权重
    w1 = bias_x * bias_y
    w2 = bias_x * (1 - bias_y)
    w3 = (1 - bias_x) * bias_y
    w4 = (1 - bias_x) * (1 - bias_y)

    # 计算在小尺寸图像在大尺寸图像中放置的起始坐标
    start_position = (np.floor(alignment_point) - small_center).astype(int)
    # 计算终止坐标是否在大尺寸图像中
    end_position = (alignment_point + small_center) - np.array(large_image.shape) + np.array((1, 1))

    # 在大尺寸的外围加1圈0，预防边界出错
    rows, cols = large_image.shape
    image = np.zeros((rows + 2, cols + 2), dtype=np.float64)
    image[1:-1, 1:-1] = large_image
    large_image = image

    # 判断是否已经超出边界
    if start_position[0] < 0 or start_position[1] < 0 or end_position[0] > 0 or end_position[1] > 0:
        return
    else:
        for i in range(0, small_image.shape[0]):
            for j in range(0, small_image.shape[1]):
                x = i + start_position[0] + 1
                y = j + start_position[1] + 1
                small_image[i, j] = small_image[i, j] + (w4 * large_image[x, y] + w2 * large_image[x + 1, y]
                                                         + w3 * large_image[x, y + 1] + w1 * large_image[x + 1, y + 1])
        return small_image


def map_to_large_image(large_image, small_image, alignment_point):
    """
    将小尺寸图像的中心和大尺寸图像中指定的对齐坐标(可以是浮点数)对齐，将小尺寸图像全部映射到大尺寸图像上
    有待改进：如果指定中心坐标超出A的范围，不添加。由于本项目不会出现这种情况，所以未实现
    :param large_image:
    :param small_image:
    :param alignment_point:
    :return: large_image
    """
    # 小尺寸图像的中心
    small_center = np.array(small_image.shape) // 2

    if small_image.shape[0] % 2 == 0:
        small_center[0] -= 1
    if small_image.shape[1] % 2 == 0:
        small_center[1] -= 1

    # 在小尺寸图像的外围加1圈0
    rows, cols = small_image.shape
    image = np.zeros((rows + 2, cols + 2), dtype=np.float64)
    image[1:-1, 1:-1] = small_image
    small_image = image

    # 计算相对偏差
    alignment_point = tuple(elem - 0.5 for elem in alignment_point)
    bias_x = alignment_point[0] - np.floor(alignment_point[0])
    bias_y = alignment_point[1] - np.floor(alignment_point[1])

    # 计算每个像素贡献权重
    w1 = bias_x * bias_y
    w2 = bias_x * (1 - bias_y)
    w3 = (1 - bias_x) * bias_y
    w4 = (1 - bias_x) * (1 - bias_y)

    # 计算在小尺寸图像在大尺寸图像对应的的起始坐标
    start_position = np.floor(alignment_point) - small_center
    # 将起始坐标限制在（0，large_image.shape）范围
    start_position = np.clip(start_position, 0, np.array(large_image.shape) - np.array((1, 1))).astype(int)
    # 计算在小尺寸图像在大尺寸图像对应的的终止坐标
    end_position = start_position + np.array((rows, cols))
    # 将终止坐标限制在（0，large_image.shape）范围
    end_position = np.clip(end_position, 0, np.array(large_image.shape) - np.array((1, 1))).astype(int)

    # 遍历计算大尺寸在起始坐标到终止坐标的像素值
    for i in range(start_position[0], end_position[0] + 1):
        for j in range(start_position[1], end_position[1] + 1):
            x = i - start_position[0] + 1
            y = j - start_position[1] + 1
            large_image[i, j] = large_image[i, j] + (w1 * small_image[x - 1, y - 1] + w2 * small_image[x - 1, y]
                                                     + w3 * small_image[x, y - 1] + w4 * small_image[x, y])
    # 返回大尺寸图像
    return large_image


def average_downsample(image, stride):
    """
    将image的stride*stride个像素求平均，作为下采样的1个像素的值。
    :param image:
    :param stride:
    :return:
    """
    # 获取原图像的高度和宽度
    height, width = image.shape[:2]

    # 计算下采样后的图像大小
    new_height = height // stride
    new_width = width // stride

    # 初始化下采样后的图像
    downsampled_image = np.zeros((new_height, new_width), dtype=np.float64)

    # 遍历原图像，每3x3的像素相加得到一个新的像素
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            roi = image[i:i + stride, j:j + stride]
            average_pixel_value = np.sum(roi)  # / (stride * stride)
            downsampled_image[i // stride, j // stride] = average_pixel_value
    return downsampled_image


def sum_downsample(image, stride):
    """
    将image的stride*stride个像素求平均，作为下采样的1个像素的值。
    :param image:
    :param stride:
    :return:
    """
    # 获取原图像的高度和宽度
    height, width = image.shape[:2]

    # 计算下采样后的图像大小
    new_height = height // stride
    new_width = width // stride

    # 初始化下采样后的图像
    downsampled_image = np.zeros((new_height, new_width), dtype=np.float64)

    # 遍历原图像，每3x3的像素相加得到一个新的像素
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            roi = image[i:i + stride, j:j + stride]
            average_pixel_value = np.sum(roi)
            downsampled_image[i // stride, j // stride] = average_pixel_value
    return downsampled_image


def superpixelize_image(image, new_shape, r):
    """
    将图像细化r倍到new_shape尺寸大小
    :param image:
    :param new_shape:
    :param r:
    :return:
    """
    # 初始化细化后的图像
    refined_image = np.zeros(new_shape, dtype=np.float64)
    for y in range(image.shape[0]):  # 高
        for x in range(image.shape[1]):  # 宽
            pixel = image[y, x]
            refined_image[y * r:(y + 1) * r, x * r:(x + 1) * r] = pixel / (r * r)
    return refined_image


def add_poisson_noise(image, intensity):
    """
    将强度为intensity的泊松噪声添加到image中
    :param image:
    :param intensity:
    :return:
    """
    # 计算泊松噪声
    noise = np.random.poisson(lam=intensity, size=image.shape)
    # 将噪声添加到图像中
    noisy_image = np.clip(image + noise, 0, 255)

    return noisy_image


def gauss_noisy(image, mean, stddev):
    """
    将均值为mean，标准差为stddev的高斯噪声添加到image上
    :param image:
    :param mean:
    :param stddev:
    :return:
    """
    # 计算正态分布噪声
    gauss = np.random.normal(mean, stddev, image.shape)
    noisy_image = cv2.add(image, gauss)
    # noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image


def angle_deviation_error(x, y):
    """
    计算x，y两个图像的角度偏差误差
    :param x:
    :param y:
    :return:
    """
    # 计算内积
    inner_product = np.sum(x * y)

    # 计算模
    magnitude_x = np.linalg.norm(x)
    magnitude_y = np.linalg.norm(y)

    # 取arcsin
    arcsin_xy = np.arcsin(inner_product/(magnitude_x * magnitude_y))

    return np.abs(arcsin_xy)


def find_max_coordinates(matrix):
    """
    寻找二维数组matrix中最大值的坐标，可能是单个，也可能是多个
    :param matrix:
    :return:
    """
    max_value = float('-inf')
    max_coordinates = []

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > max_value:
                max_value = matrix[i][j]
                max_coordinates = [(i, j)]
            elif matrix[i][j] == max_value:
                max_coordinates.append((i, j))

    return max_coordinates


def chart(*PSF):
    """
    输入多个相同大小的二维数组，展示中心水平切片和对角线切片图像
    :param PSF:
    :return:
    """
    # （水平）创建图像
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in PSF:
        for name, value in p.items():
            # 沿水平方向切片
            horizontal_slice = value[int(value.shape[1] / 2), :]
            # 绘制水平切片
            # ax.plot(horizontal_slice, label=f'{name}')
            ax.plot(horizontal_slice,
                    label=f'{name} (Max: {np.max(horizontal_slice):.6f}, Min: {np.min(horizontal_slice):.6f})')

    # 添加图例
    ax.legend()
    ax.set_title('Horizontal Cut Comparison')
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Normalized Intensity')
    # 显示图像
    plt.show()

    # (对角)创建图像
    fig1, ax = plt.subplots(figsize=(10, 5))
    for p in PSF:
        for name, value in p.items():
            # 沿对角线（左上-右下）切片
            diagonal_slice = np.diagonal(value)
            # 绘制对角线切片
            # ax.plot(diagonal_slice, label=f'{name}')
            ax.plot(diagonal_slice,
                    label=f'{name} (Max: {np.max(diagonal_slice):.6f}, Min: {np.min(diagonal_slice):.6f})')

    # 添加图例
    ax.legend()
    ax.set_title('Diagonal Cut Comparison')
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Normalized Intensity')
    # 显示图像
    plt.show()
    return


def get_star_list(file_name):
    """
    读取file_name里的星源中心坐标，其中文件为QMPF文件
    :param file_name:
    :return:
    """
    # 打开文件
    with open(file_name, 'r', encoding='utf-8') as file:
        # 读取文件的所有行到一个列表中
        lines = file.readlines()
    star_list = []

    for line in lines:
        data = line.split()
        star_list.append((float(data[6]), float(data[7]), float(data[12])))
        # print(line)
    return star_list


def calculate_optimal_photometry(PSF, image, star, r):
    """
    星源的最佳光度测定
    :param PSF: 高分辨率点扩散函数
    :param image: 严重欠采样图像
    :param star: 单一星源坐标（细分像素后的坐标）
    :param r: 细化的倍数
    :return: 该星源的最佳光度测定
    """
    PSF = PSF / np.sum(PSF)
    # 移位
    P = np.zeros(((PSF.shape[0] + 2 * r), (PSF.shape[1] + 2 * r)))
    PSF_center = np.array(P.shape) // 2
    if P.shape[0] % 2 == 0:
        PSF_center[0] -= 1
    if P.shape[1] % 2 == 0:
        PSF_center[1] -= 1
    # 中间大像素（r，r）的左上角坐标
    left_top = PSF_center - (np.ceil(r / 2) - 1, np.ceil(r / 2) - 1)
    # 计算细化后的偏移量
    lr_x = star[0] / r
    lr_y = star[1] / r
    bias_x = (lr_x - int(lr_x)) * r
    bias_y = (lr_y - int(lr_y)) * r
    center_xy = left_top + (bias_x, bias_y)
    # 计算偏移后的PSF（尺寸扩大r圈像素，防止放置超出像素范围）
    P = map_to_large_image(P, PSF, center_xy)
    # 以中心大橡塑为中间，截取出最终的范围
    lr_size = int(PSF.shape[0] / r) if int(PSF.shape[0] / r) % 2 != 0 else int(PSF.shape[0] / r) + 1
    half_size = int(lr_size / 2)
    left = int(left_top[0] - half_size * r)
    right = int(left_top[0] + half_size * r + r)
    top = int(left_top[1] - half_size * r)
    bottom = int(left_top[1] + half_size * r + r)
    shifted_PSF = P[top:bottom, left:right]
    # 下采样至欠采样像素大小
    Pp = sum_downsample(shifted_PSF, r)
    # 归一化
    Pp = Pp / np.sum(Pp)
    # 计算每个像素的权重
    Wp = Pp / np.sum(np.square(Pp))

    # 根据权重的尺寸截取星源区域
    T = np.zeros_like(Wp)
    # 加0.5是因为希望T像素中心和图像像素中心对齐，而不是和图像像素的左上角对齐
    T = map_to_small_image(image, T, (int(lr_x) + 0.5, int(lr_y) + 0.5))
    a = np.sum(T)
    # 计算通量
    Flux = np.sum(Wp * T)
    return Flux


def moment_method(image):

    image = image - (np.sum(image) / (image.shape[0] * image.shape[1]))
    # 计算行的质心
    row_center_of_mass = np.sum(image, axis=1) / np.sum(image)
    y0 = np.sum(np.arange(image.shape[0]) * row_center_of_mass)

    # 计算列的质心
    col_center_of_mass = np.sum(image, axis=0) / np.sum(image)
    x0 = np.sum(np.arange(image.shape[1]) * col_center_of_mass)
    # x_m = 0.0   # x坐标分子
    # y_m = 0.0   # y坐标分子
    # x_d = 0.0   # x坐标分母
    # y_d = 0.0   # y坐标分母
    #
    # # image = np.log(image)
    # # image = image + np.full(image.shape, 1)
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         x_m = x_m + image[i, j] * (i + 1)
    #         y_m = y_m + image[i, j] * (j + 1)
    #         x_d = x_d + image[i, j]
    #         y_d = y_d + image[i, j]
    # x0 = x_m / x_d
    # y0 = y_m / y_d
    return x0, y0
