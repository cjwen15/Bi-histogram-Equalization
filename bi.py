import cv2
import numpy as np
import matplotlib.pyplot as plt


# 建立原始图像各灰度级的灰度值与像素个数对应表
def Origin_histogram(img):
    histogram = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k = img[i][j]
            if k in histogram:
                histogram[k] += 1
            else:
                histogram[k] = 1

    sorted_histogram = {}  # 建立排好序的映射表
    sorted_list = sorted(histogram)  # 根据灰度值进行从低至高的排序

    for j in range(len(sorted_list)):
        sorted_histogram[sorted_list[j]] = histogram[sorted_list[j]]

    return sorted_histogram


# 直方图均衡化
def equalization_histogram(histogram, img):
    pr = {}  # 建立概率分布映射表

    for i in histogram.keys():
        pr[i] = histogram[i] / (img.shape[0] * img.shape[1])

    tmp = 0
    for m in pr.keys():
        tmp += pr[m]
        pr[m] = max(histogram) * tmp

    new_img = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)

    for k in range(img.shape[0]):
        for l in range(img.shape[1]):
            new_img[k][l] = pr[img[k][l]]

    return new_img


# 双直方图均衡化
def equalization_bihistogram(img):
    height, width = img.shape[:2]
    size = img.size

    # 计算图像灰度均值
    X_mean = 0
    for i in range(height):
        for j in range(width):
            X_mean += img[i][j] / size

    X_mean = int(np.mean(img))
    X_min = int(np.min(img))
    X_max = int(np.max(img))
    # X_mean = int(X_mean)
    # X_min = int(img.min())
    # X_max = int(img.max())

    Xl = np.zeros((X_mean + 1))  # 记录图像在（X_min, X_mean）范围内的灰度值
    Xu = np.zeros((256))  # 记录图像在（X_mean, X_max）范围内的灰度值
    nl = 0
    nu = 0
    for i in range(height):
        for j in range(width):
            if (img[i, j] <= X_mean).all():  # 统计≤平均值的各级灰度值数量及总数
                Xl[img[i, j]] = Xl[img[i, j]] + 1
                nl = nl + 1
            else:  # 统计>平均值的各级灰度值数量及总数
                Xu[img[i, j]] = Xu[img[i, j]] + 1
                nu = nu + 1

    Xa = {}
    Xa[height + 1] = X_mean + 1
    while Xu[Xa[height + 1]] == 0:
        Xa[height + 1] = Xa[height + 1] + 1

    # 记录对应各级灰度值的概率密度
    Pl = Xl / nl
    Pu = Xu / nu
    # 累计密度函数
    Cl = Xl
    Cu = Xu
    Cl[0] = Pl[0]
    Cu[Xa[height + 1]] = Pu[Xa[height + 1]]
    for i in range(0, X_mean):
        Cl[i] = Pl[i] + Cl[i]
    for i in range(X_mean + 1, 256):
        Cu[i] = Pu[i] + Cu[i]

    # 灰度转换函数
    fl = Cl
    fu = Cu
    for i in range(0, X_mean):
        fl[i] = X_min + Cl[i] * (X_mean - X_min)
    for i in range(Xa[height + 1], 256):
        fu[i] = X_mean + Cu[i] * (X_max - Xa[height + 1])

    # 两个子图像合并
    new_img = img
    for i in range(height):
        for j in range(width):
            if (img[i, j] <= X_mean).all():
                new_img[i, j] = fl[img[i, j]]
            else:
                new_img[i, j] = fu[img[i, j]]
    return new_img


# 计算灰度直方图
def GrayHist(img):
    height, width = img.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(height):
        for j in range(width):
            grayHist[img[i][j]] += 1
    return grayHist


if __name__ == '__main__':
    '''直方图均衡化'''
    # # 读取原始图像
    # img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    # # 计算原图灰度直方图
    # origin_histogram = Origin_histogram(img)
    #
    # # 直方图均衡化
    # new_img = equalization_histogram(origin_histogram, img)
    #
    # origin_grayHist = GrayHist(img)
    # equaliza_grayHist = GrayHist(new_img)
    # x = np.arange(256)
    # # 绘制灰度直方图
    # plt.figure(num=1)
    # plt.subplot(2, 2, 1)
    # plt.plot(x, origin_grayHist, 'r', linewidth=2, c='black')
    # plt.title("Origin")
    # plt.ylabel("number of pixels")
    # plt.subplot(2, 2, 2)
    # plt.plot(x, equaliza_grayHist, 'r', linewidth=2, c='black')
    # plt.title("Equalization")
    # plt.ylabel("number of pixels")
    # plt.subplot(2, 2, 3)
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.title('Origin')
    # plt.subplot(2, 2, 4)
    # plt.imshow(new_img, cmap=plt.cm.gray)
    # plt.title('Equalization')
    # plt.show()

    '''双直方图均衡化'''
    img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)

    # 双直方图均衡化
    # new_img = equalization_bihistogram(img)

    (B, G, R) = cv2.split(img)
    IB = equalization_bihistogram(B)
    IG = equalization_bihistogram(G)
    IR = equalization_bihistogram(R)
    IN = np.concatenate((IB, IG, IR))
    plt.subplot(1, 2, 1)
    plt.title("A).Histogram Equalization")
    plt.imshow(img)
    plt.show()

    plt.subplot(1, 2, 2)
    plt.title("B).Bi-histogram Equalization")
    plt.imshow(cv2.normalize(IN), cmap='gray')
    plt.show()
