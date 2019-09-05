import sys
import cv2
from tqdm import trange
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
def carve_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)
    for i in trange(c - new_c):  #显示进度条
        img = carve(img)

    return img

def carve_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1)) #旋转
    img = carve_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def carve(img):#进行分割
    r, c, _ = img.shape
    M, track = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])  # 找到最后一行的最小值
    for i in reversed(range(r)):  # 从后向前遍历
        mask[i, j] = False
        j = track[i, j]

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3)) #共用内存
    return img


def minimum_seam(img): #筛选出能量最小的一条线
    r, c, _ = img.shape
    energy_map = get_energy(img)

    M = energy_map.copy()
    track = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:  # 处理图像的左侧边缘，确保不会索引-1
                tmp = M[i-1, j:j + 2] #取上一行相邻元素的最小值
                tmplist = tmp.tolist()
                min_e = min(tmplist)
                min_e_index = tmplist.index(min(tmplist))
                #print("index",min_e,min_e_index)
                track[i,j] = min_e_index + j -1
                M[i,j] += min_e
            else:
                tmp = M[i - 1, j-1:j + 2]  # 取相邻元素的最小值
                tmplist = tmp.tolist()
                min_e = min(tmplist)
                min_e_index = tmplist.index(min(tmplist))
                #print("index", min_e, min_e_index)
                track[i, j] = min_e_index + j -1
                M[i, j] += min_e

    return M, track

def get_energy(img): #得到能量图
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # 这会将它从2D滤波转换为3D滤波器
    # 为每个通道：R，G，B复制相同的滤波器
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # 这会将它从2D滤波转换为3D滤波器
    # 为每个通道：R，G，B复制相同的滤波器
    filter_dv = np.stack([filter_dv] * 3, axis=2)
    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))
    # 我们计算红，绿，蓝通道中的能量值之和
    energy_map = convolved.sum(axis=2)
    return energy_map


def main():
    which_axis = " "
    scale_r = 0.9
    scale_c = 0.8
    in_filename = "tree.jpg"
    out_filename = "resize_tree.jpg"

    img = imread(in_filename)

    if which_axis == 'r':
        out = carve_r(img, scale_r)
    elif which_axis == 'c':
        out = carve_c(img, scale_c)
    else:
        out = carve_r(img, scale_r)
        out = carve_c(out, scale_c)

    imwrite(out_filename, out)

if __name__ == '__main__':
    main()