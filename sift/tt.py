import numpy as np
import cv2 as cv
import imutils

def stitch(img1,img2, ratio, reprojThresh,showMatches = False):
    #获取关键点和描述符
    (kp1, des1) = detectAndDescribe(img1)
    (kp2, des2) = detectAndDescribe(img2)
    print(len(kp1),len(des1))
    print(len(kp2), len(des2))
    R = matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh)

    if R is None:  #如果没有足够的最佳匹配点返回none
        return  None
    (good, M, mask) = R

    #对img1透视变换，M是ROI区域矩阵， 变换后的大小是(img1.w+img2.w, img1.h)
    print(M.shape)
    wrap = cv.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    wrap[0:img2.shape[0], 0:img2.shape[1]] = img2  #将img2的值赋给结果图像
    rows, cols = np.where(wrap[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    result = wrap[min_row:max_row, min_col:max_col, :]  # 去除黑色无用部分

    # 是否需要显示ROI区域
    if showMatches:
        vis = drawMatches(img1, img2, kp1, kp2, good, mask)
        return (result, vis)

    return result


def detectAndDescribe(img):  #返回关键点和描述符
    sift = cv.xfeatures2d.SIFT_create()
    kps, des = sift.detectAndCompute(img, None)
    kps = np.float32([kp.pt for kp in kps])
    return (kps, des)


def matchKeyPoints(kp1, kp2, des1, des2, ratio, reprojThresh):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2) #暴力匹配 knn

    good = []
    for m in matches:  #获取理想匹配
        if m[0].distance < ratio * m[1].distance: #如果最近的距离除以次近的距离得到的比率ratio少于某个阈值T，则接受这一对匹配点
            good.append((m[0].trainIdx, m[0].queryIdx))

    #最少要有四个点才能做透视变换
    if len(good) > 4:
        #获取关键点的坐标
        src_pts = np.float32([kp1[i] for (_, i) in good])
        dst_pts = np.float32([kp2[i] for (i, _) in good])

        #通过两个图像的关键点计算变换矩阵
        (M, mask) = cv.findHomography(src_pts, dst_pts, cv.RANSAC, reprojThresh)
        # 其中M为求得的单应性矩阵矩阵
        # mask则返回一个列表来表征匹配成功的特征点。
        # ptsA,ptsB为关键点
        # cv2.RANSAC, 表示使用RANSAC方法去掉污染的数据
        # ransacReprojThreshold 则表示一对内群点所能容忍的最大投影误差

        return (good, M, mask)#返回最佳匹配点、变换矩阵和掩模

    return None #如果不满足最少四个 就返回None



def drawMatches(img1, img2, kp1, kp2, metches,mask):
    (hA,wA) = img1.shape[:2]
    (hB,wB) = img2.shape[:2]
    vis = np.zeros((max(hA,hB), wA+wB, 3), dtype='uint8')
    vis[0:hA, 0:wA] = img1
    vis[0:hB, wA:] = img2
    for ((trainIdx, queryIdx),s) in zip(metches, mask):
        if s == 1:
            ptA = (int(kp1[queryIdx][0]), int(kp1[queryIdx][1]))
            ptB = (int(kp2[trainIdx][0])+wA, int(kp2[trainIdx][1]))
            cv.line(vis, ptA, ptB, (0, 255, 0), 1)

    return vis

