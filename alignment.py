import math
import random

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)    #A矩阵是2N*9的

    for i in range(len(matches)):   #N次循环
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt    #注意！！！
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #TODO-BLOCK-BEGIN

        A[2 * i] = np.array([a_x, a_y, 1, 0, 0, 0, -b_x*a_x, -b_x*a_y, -b_x])
        A[2 * i + 1] = np.array([0, 0, 0, a_x, a_y, 1, -b_y*a_x, -b_y*a_y, -b_y])

        # raise Exception("TODO in alignment.py not implemented")
        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)  #A分解为U，S，V 。S只有对角线上有不为0的值，返回是一个以为矩阵

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD
    #TODO-BLOCK-BEGIN
    '''
    n1 = np.dot(A.T,A) #矩阵转置元素相乘
    eg,vec = np.linalg.eig(n1)  #特征值，特征向量
    eg1 = eg.tolist()  #index函数要转成list才可以用
    min1 = eg1.index(min(eg1)) #最小特征值的索引
    H = np.reshape(vec[:,min1],(3,3)) #最小特征值的一列向量为特征向量
    '''
    min_eig = np.array(Vt[-1])/np.sum(Vt[-1])   #两种写法都可以
    H = min_eig.reshape(3, 3)

    # raise Exception("TODO in alignment.py not implemented")
    #TODO-BLOCK-END
    #END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslation) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call compute_homography.
    #This function should also call get_inliers and, at the end,
    #least_squares_fit.
    #TODO-BLOCK-BEGIN
    #下面两种写法都可以
    
    len_m = len(matches) - 1
    max_ = []
    for i in range(nRANSAC):
        if m == eTranslate:
            j = random.randint(0, len_m)
            match = matches[j]
            (a_x, a_y) = f1[match.queryIdx].pt  #这个匹配第一个特征点的坐标
            (b_x, b_y) = f2[match.trainIdx].pt
            x = b_x - a_x
            y = b_y - a_y
            #x, y = np.array(f2[match.trainIdx].pt) - np.array(f1[match.queryIdx].pt)
            #M = np.array([1, 0, x, 0, 1, y, 0, 0, 1]).reshape(3, 3)
            M = np.array([[1, 0, x],[ 0, 1, y], [0, 0, 1]]) #变换矩阵
        elif m == eHomography:
            match = []
            while len(match) < 4:
                j = random.randint(0, len_m)
                if matches[j] not in match:
                    match.append(matches[j])
            M = computeHomography(f1, f2, match)

        inlier_indices = getInliers(f1, f2, matches, M, RANSACthresh)
        if len(inlier_indices) > len(max_):
            max_ = inlier_indices
    M = leastSquaresFit(f1, f2, matches, m, max_)
    
    '''
    total = len(matches)   #匹配对的数量
    totallist = list(range(total)) #返回一个列表
    numlist = [[] for j in range(nRANSAC)]
    number = []
    #total1 = random

    for i in range(nRANSAC):   #RANSAC迭代次数
        if(m==eTranslate):
            index1 = random.sample(totallist,1) #从列表中选取1个元素
            peer = matches[index1]  #选这一个对
            (a_x, a_y) = f1[peer.queryIdx].pt  #这个匹配第一个特征点的坐标
            (b_x, b_y) = f2[peer.trainIdx].pt
            xt = b_x - a_x
            yt = b_y - b_y
            H = np.array([[1,0,xt],[0,1,yt],[0,0,1]])  #变换矩阵
            numlist[i] = getInliers(f1,f2,matches,H,RANSACthresh)  #内点特征匹配对的索引 这个第三个参数是啥
            number.append(len(numlist[i]))        #????????????
        else:
            index2 = random.sample(totallist,4)
            peer1 = []
            peer1.append(matches[index2[0]])    #列表添加
            peer1.append(matches[index2[1]])
            peer1.append(matches[index2[2]])
            peer1.append(matches[index2[3]])
            H = computeHomography(f1,f2,peer1)
            numlist[i] = getInliers(f1,f2,matches,H,RANSACthresh)
            number.append(len(numlist[i]))
    index = number.index(max(number)) #最多的索引

    M  = leastSquaresFit(f1,f2,matches,m,numlist[index])  #注意最后一个参数是索引
    '''
    '''
    number = 0
    final_inliers = []

    for i in range(nRANSAC):
        if m == eTranslate:
            match = [matches[random.randint(0, len(matches) - 1)]]
            temp_M = leastSquaresFit(f1, f2, match, eTranslate, [0])

        else:
            match = random.sample(matches, 4)
            temp_M = computeHomography(f1, f2, match)

        inlier_indices = getInliers(f1, f2, matches, temp_M, RANSACthresh)

        if len(inlier_indices) > number:
            number = len(inlier_indices)
            final_inliers = inlier_indices
    M = leastSquaresFit(f1, f2, matches, m, final_inliers)
    '''


    # raise Exception("TODO in alignment.py not implemented")
    #TODO-BLOCK-END
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        #Determine if the ith matched feature f1[id1], when transformed
        #by M, is within RANSACthresh of its match in f2.
        #If so, append i to inliers
        #TODO-BLOCK-BEGIN
        
        peer = matches[i]

        pt1 = f1[peer.queryIdx].pt
        pt2 = np.array(f2[peer.trainIdx].pt)
        pt3 = np.array([pt1[0], pt1[1], 1]).T
        pt4 = M.dot(pt3)
        x, y = [pt4[0], pt4[1]] / pt4[2]
        dist = np.linalg.norm(np.array([x, y]) - pt2)
        if dist < RANSACthresh:
            inlier_indices.append(i)
        
        '''
        pt1 = np.array(f1[i].pt)   #np数组
        pt2 = np.array(f2[i].pt)
        pt3 = np.array([pt1[0],pt1[1],1]).T
        x,y,_ = np.dot(M,pt3)
        dist = np.linalg.norm(np.array([x,y])-pt2)
        if dist < RANSACthresh:
            inlier_indices.append(i)
        '''
        #raise Exception("TODO in alignment.py not implemented")
        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in inlier_indices:
            #BEGIN TODO 6
            #Use this loop to compute the average translation vector
            #over all inliers.
            #TODO-BLOCK-BEGIN
            
            peer = matches[i]
            k1 = f1[peer.queryIdx]
            k2 = f2[peer.trainIdx]
            x, y = np.array(k2.pt) - np.array(k1.pt)
            u += x
            v += y
           
            #raise Exception("TODO in alignment.py not implemented")
            #TODO-BLOCK-END
            #END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN
        in_matches = []
        for i in inlier_indices:
            in_matches.append(matches[i])
        M = computeHomography(f1, f2, in_matches)
        #raise Exception("TODO in alignment.py not implemented")
        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

