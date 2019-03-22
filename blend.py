import math, time
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    # TODO 8
    # TODO-BLOCK-BEGIN

    # img is a numpyarray with 3 dims and the shape of (10, 10, 3)
    x, y = img.shape[:2]
    pts = np.array([[0, 0, 1], [0, x-1, 1], [y-1, 0, 1], [y-1, x-1, 1]]).T
    #四个顶点的坐标
    res = np.dot(M, pts)
    res = res / res[-1]
    minX = res[0][np.argmin(res[0])]
    minY = res[1][np.argmin(res[1])]
    maxX = res[0][np.argmax(res[0])]
    maxY = res[1][np.argmax(res[1])] 

    # TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    #img与acc相加
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    # TODO-BLOCK-BEGIN

    # The shape of acc is (accHeight, accWidth, channels + 1)
    
    acc_rows, acc_cols, _ = acc.shape    
    rows, cols = img.shape[:2]     
    img = cv2.copyMakeBorder(img, 0, acc_rows - rows, 0, acc_cols - cols, cv2.BORDER_CONSTANT, value=0)   #使acc和img的尺寸一样

    row, col, _ = img.shape
    x_range = np.arange(col)
    y_range = np.arange(row)
    (x_mesh, y_mesh) = np.meshgrid(x_range, y_range)
    tmp = np.ones((row, col))
    coord = np.dstack((x_mesh, y_mesh, tmp)).reshape((col * row, 3)).T
    

    loca = np.linalg.inv(M).dot(coord)
    loca = loca / loca[2]  #笛卡尔坐标为1

    map_x = loca[0].reshape((row, col)).astype(np.float32)  #这里要进行类型转换
    map_y = loca[1].reshape((row, col)).astype(np.float32)
    img_warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    (minX, minY, maxX, maxY) = imageBoundingBox(img, M)

    mat_one = np.ones((img_warped.shape[0], img_warped.shape[1], 1))  #img_warped图像加权重
    weight_img = np.dstack((img_warped, mat_one))
    p = 1 / blendWidth
    img_right = np.clip(np.linspace(-p * minX, p * (acc_cols - 1 - minX), acc_cols), 0, 1).reshape((1, acc_cols, 1))
    img_left = np.ones((1, acc_cols, 1)) - img_right

    img_feathered = img_right * weight_img    #逐个相乘
    acc *= img_left  #在源位置上进行改变acc
    #三个RGB通道都为0
    gray_img = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
    gray_acc = cv2.cvtColor(acc[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    change_img = (gray_img != 0).reshape((acc_rows, acc_cols, 1))
    change_acc = (gray_acc != 0).reshape((acc_rows, acc_cols, 1))

    img_masked = change_img * img_feathered
    acc *= change_acc     #得到带权重通道的acc
    acc += img_masked
    
    '''
    acc_height, acc_width, _ = acc.shape
    img_height, img_width = img.shape[:2]
    img = cv2.copyMakeBorder(img, 0, acc_height - img_height, 0, acc_width - img_width, cv2.BORDER_CONSTANT,value=0)

    row, col, _ = img.shape
    x_range = np.arange(col)
    y_range = np.arange(row)
    (x_grid, y_grid) = np.meshgrid(x_range, y_range)
    ones = np.ones((row, col))
    coordinates = np.dstack((x_grid, y_grid, ones))
    coordinates = coordinates.reshape((col * row, 3))
    coordinates = coordinates.T

    location = np.linalg.inv(M).dot(coordinates)
    location = location / location[2]

    map_x = location[0].reshape((row, col)).astype(np.float32)
    map_y = location[1].reshape((row, col)).astype(np.float32)
    img_warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    (minX, minY, maxX, maxY) = imageBoundingBox(img, M)

    dst_one = np.ones((img_warped.shape[0], img_warped.shape[1], 1))
    dst_img = np.dstack((img_warped, dst_one))
    k = 1 / blendWidth
    feather_right = np.clip(np.linspace(-k * minX, k * (acc_width - 1 - minX), acc_cols), 0, 1).reshape((1, acc_width, 1))
    feather_left = np.ones((1, acc_width, 1)) - feather_right

    img_feathered = feather_right * dst_img
    acc *= feather_left

    grayimg = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
    grayacc = cv2.cvtColor(acc[:, :, :3].astype('uint8'), cv2.COLOR_BGR2GRAY)
    maskimg = (grayimg != 0).reshape((acc_height, acc_width, 1))
    maskacc = (grayacc != 0).reshape((acc_height, acc_width, 1))

    img_masked = maskimg * img_feathered
    acc *= maskacc
    acc += img_masked
    '''
    # TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    # TODO-BLOCK-BEGIN
    
    acc[:, :, 3][acc[:, :, 3] == 0] = 1
    img = acc / acc[:, :, 3].reshape((acc.shape[0], acc.shape[1], 1))   #归一化
    img = img[:, :, :3].astype(np.uint8)
    
    '''
    下面的有问题
    isweight = (acc[:,:,3] == 0)
    acc[:,:,3] = acc[:,:,3] + isweight   #权重为0的点权重要加1
    img = acc / acc[:,:,3].reshape((acc.shape[0],acc.shape[1],1))   #归一化
    img = acc[:,:,3].astype('int8')
    '''
    # TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accHeight: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        # TODO-BLOCK-BEGIN

        minx, miny, maxx, maxy = imageBoundingBox(img, M)
        if minx < minX:
            minX = minx
        if miny < minY:
            minY = miny
        if maxx > maxX:
            maxX = maxx
        if maxy > maxY:
            maxY = maxy
        # TODO-BLOCK-END
        # END TODO
    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    #关于TODO12在图像中心画分割线有关
    #因为在全景图中，第一张图要放在开始和结束两个位置，当匹配完成后，第一张和最后一张的中心都进行切割
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function   混合函数宽度
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)  #调用TODO9 得到图像画框的信息
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels    #图片拼接
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)    #当为全景图的时候，移动了width/2
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    # TODO-BLOCK-BEGIN

    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, width)

    # TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage
