"""
Project ：Image-Enhancement 
File    ：common_lighting_algorithm.py
Author  ：MangoloD
Date    ：2022/3/9 14:15 
"""

import cv2
import sys
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def grey_world(image):
    """
    :param image:
    :return:
    灰度世界算法
    灰度世界算法以灰度世界假设为基础，该假设认为：对于一幅有着大量色彩变化的图像，R,G,B三个分量的平均值趋于同一灰度值Gray。从物理意义上讲，灰色世界法假设自然界景物对于光线的平均反射的均值在总体上是个定值，这个定值近似地为“灰色”。颜色平衡算法将这一假设强制应用于待处理图像，可以从图像中消除环境光的影响，获得原始场景图像。

    一般有两种方法确定Gray值

    1) 使用固定值，对于8位的图像(0~255)通常取128作为灰度值

    2) 计算增益系数,分别计算三通道的平均值avgR，avgG，avgB，则：

    Avg=(avgR+avgG+avgB)/3

    kr=Avg/avgR , kg=Avg/avgG , kb=Avg/avgB

    利用计算出的增益系数，重新计算每个像素值，构成新的图片

    ② 算法优缺点

    这种算法简单快速，但是当图像场景颜色并不丰富时，尤其出现大块单色物体时，该算法常会失效。

    vegetable.png效果明显， sky.png效果不明显
    """
    img = image.transpose(2, 0, 1).astype(np.uint32)
    avg_B = np.average(img[0])
    avg_G = np.average(img[1])
    avg_R = np.average(img[2])

    avg = (avg_B + avg_G + avg_R) / 3

    img[0] = np.minimum(img[0] * (avg / avg_B), 255)
    img[1] = np.minimum(img[1] * (avg / avg_G), 255)
    img[2] = np.minimum(img[2] * (avg / avg_R), 255)

    return img.transpose(1, 2, 0).astype(np.uint8)


def his_equ_color(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])  # equalizeHist(in, out)
    cv2.merge(channels, ycrcb)
    img_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img_eq


def whiteBalance(img):
    rows = img.shape[0]
    cols = img.shape[1]
    final = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(final[:, :, 1])
    avg_b = np.average(final[:, :, 2])
    for x in range(final.shape[0]):
        for y in range(final.shape[1]):
            l, a, b = final[x, y, :]
            # fix for cv correction
            l *= 100 / 255.0
            final[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            final[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)

    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)
    return final


class MSRCR_Method:
    # 单尺度计算
    def singleScaleRetinex(self, img, sigma):
        temp = cv2.GaussianBlur(img, (0, 0), sigma)
        gaussian = np.where(temp == 0, 0.01, temp)
        retinex = np.log10(img + 0.01) - np.log10(gaussian)

        return retinex

    # 多尺度计算
    def multiScaleRetinex(self, img, sigma_list):
        retinex = np.zeros_like(img * 1.0)
        for sigma in sigma_list:
            retinex += self.singleScaleRetinex(img, sigma)
            retinex = retinex / len(sigma_list)
        return retinex

    # 颜色恢复
    def colorRestoration(self, img, alpha, beta):
        img_sum = np.sum(img, axis=2, keepdims=True)
        color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

        return color_restoration

    # 在rgb三通道上分别统计每个像素值的出现次数
    # 将1%的最天值和最小值设孟为255和0,其余值映射到(0, 255)
    # low_clip:0.01
    # high_clip:0.99
    def simplestColorBalance(self, img, low_clip, high_clip):
        total = img.shape[0] * img.shape[1]

        for i in range(img.shape[2]):
            unique, counts = np.unique(img[:, :, i], return_counts=True)
            current = 0
            for u, c in zip(unique, counts):
                if float(current) / total < low_clip:
                    low_val = u
                if float(current) / total < high_clip:
                    high_val = u
                current += c
            # 将img中元素规整到1ow_val~high_val区间中
            img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

        return img

    # SSR算法
    def SSR(self, img, sigma):
        img_ssr = self.singleScaleRetinex(img, sigma)
        # 量化到0-255
        for i in range(img_ssr.shape[2]):
            img_ssr[:, :, i] = (img_ssr[:, :, i] - np.min(img_ssr[:, :, i])) / (
                    np.max(img_ssr[:, :, i]) - np.min(img_ssr[:, :, i])) * 255

        img_ssr = np.uint8(np.minimum(np.maximum(img_ssr, 0), 255))
        return img_ssr

    # MSR算法
    def MSR(self, img, sigma_list):
        img_msr = self.multiScaleRetinex(img, sigma_list)
        for i in range(img_msr.shape[2]):
            img_msr[:, :, i] = (img_msr[:, :, i] - np.min(img_msr[:, :, i])) / (
                    np.max(img_msr[:, :, i]) - np.min(img_msr[:, :, i])) * 255

        img_msr = np.uint8(np.minimum(np.maximum(img_msr, 0), 255))
        return img_msr

    # MSRCR算法
    def MSRCR(self, img, sigma_list, G, b, alpha, beta, low_clip, high_clip):

        img = np.float64(img) + 1.0
        img_retinex = self.multiScaleRetinex(img, sigma_list)
        img_color = self.colorRestoration(img, alpha, beta)
        img_msrcr = G * (img_retinex * img_color + b)
        for i in range(img_msrcr.shape[2]):
            img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / (
                    np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * 255

        img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
        img_msrcr = self.simplestColorBalance(img_msrcr, low_clip, high_clip)
        return img_msrcr


class ACE:
    # 饱和函数

    def calc_saturation(self, diff, slope, limit):
        ret = diff * slope
        if ret > limit:
            ret = limit
        elif ret < (-limit):
            ret = -limit
        return ret

    def automatic_color_equalization(self, img, slope=10, limit=1000, samples=500):
        img = img.transpose(2, 0, 1)
        # Convert input to an ndarray with column-major memory order(仅仅是地址连续，内容和结构不变)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        width = img.shape[2]
        height = img.shape[1]
        cary = []
        # 随机产生索引
        for i in range(0, samples):
            _x = random.randint(0, width) % width
        _y = random.randint(0, height) % height
        dict = {"x": _x, "y": _y}
        cary.append(dict)
        mat = np.zeros((3, height, width), float)
        r_max = sys.float_info.min
        r_min = sys.float_info.max
        g_max = sys.float_info.min
        g_min = sys.float_info.max
        b_max = sys.float_info.min
        b_min = sys.float_info.max
        for i in range(height):
            for j in range(width):
                r = img[0, i, j]
                g = img[1, i, j]
                b = img[2, i, j]
                r_rscore_sum = 0.0
                g_rscore_sum = 0.0
                b_rscore_sum = 0.0
                denominator = 0.0

                for _dict in cary:
                    _x = _dict["x"]  # width
                    _y = _dict["y"]  # height

                    # 计算欧氏距离
                    dist = np.sqrt(np.square(_x - j) + np.square(_y - i))
                    if dist < height / 5:
                        continue

                    _sr = img[0, _y, _x]
                    _sg = img[1, _y, _x]
                    _sb = img[2, _y, _x]

                    r_rscore_sum += self.calc_saturation(int(r) - int(_sr), slope, limit) / dist
                    g_rscore_sum += self.calc_saturation(int(g) - int(_sg), slope, limit) / dist
                    b_rscore_sum += self.calc_saturation(int(b) - int(_sb), slope, limit) / dist
                    denominator += limit / dist

                r_rscore_sum = r_rscore_sum / denominator
                g_rscore_sum = g_rscore_sum / denominator
                b_rscore_sum = b_rscore_sum / denominator
                mat[0, i, j] = r_rscore_sum

                mat[1, i, j] = g_rscore_sum

                mat[2, i, j] = b_rscore_sum
                if r_max < r_rscore_sum:
                    r_max = r_rscore_sum
                if r_min > r_rscore_sum:
                    r_min = r_rscore_sum
                if g_max < g_rscore_sum:
                    g_max = g_rscore_sum
                if g_min > g_rscore_sum:
                    g_min = g_rscore_sum
                if b_max < b_rscore_sum:
                    b_max = b_rscore_sum

                if b_min > b_rscore_sum:
                    b_min = b_rscore_sum

        for i in range(height):
            for j in range(width):
                img[0, i, j] = (mat[0, i, j] - r_min) * 255 / (r_max - r_min)
                img[1, i, j] = (mat[1, i, j] - g_min) * 255 / (g_max - g_min)
                img[2, i, j] = (mat[2, i, j] - b_min) * 255 / (b_max - b_min)

        return img.transpose(1, 2, 0).astype(np.uint8)


def paint(img, process_img, sign):
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(process_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(sign)

    plt.show()


if __name__ == '__main__':
    image_path = 'images/vegetable.png'
    image = cv2.imread(image_path)
    msrcr = MSRCR_Method()

    sign = 'awb'

    process_image = None
    if sign == 'grey_world':
        process_image = grey_world(image)
        paint(image, process_image, sign)
    elif sign == 'his_equ_color':
        process_image = his_equ_color(image)
        paint(image, process_image, sign)
    elif sign == 'awb':
        process_image = whiteBalance(image)
        paint(image, process_image, sign)
    elif sign == 'ace':
        ace = ACE()
        process_image = ace.automatic_color_equalization(image)
        paint(image, process_image, sign)
    else:
        ssr_img = msrcr.SSR(img=image, sigma=200)
        msr_img = msrcr.MSR(img=image, sigma_list=[15, 80, 200])
        msrcr_img = msrcr.MSRCR(img=image, sigma_list=[15, 80, 200], G=5, b=25, alpha=125, beta=46, low_clip=0.01,
                                high_clip=0.99)
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Image')

        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(ssr_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('SSR')

        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(msr_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('MSR')

        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(msrcr_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('MSRCR')

        plt.show()
