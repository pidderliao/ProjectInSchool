# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from itertools import permutations
import numpy as np
import random
import copy
import cv2
import os
import glob
import sys
import logging


class DBTool:
    def __init__(self):
        #請你給我一個 Log 的分身，他的名字叫做.... __name__ (function 的名稱)！！
        self.logger = logging.getLogger( 'ForgeryPSO.py' )
        #設定這個log 分身他要處理的情報等級
        self.logger.setLevel(logging.INFO)
        #關於 log 將要輸出的檔案，請你按照下面的設定，幫我處理一下
        fh = logging.FileHandler('receive.log', 'w', 'utf-8')
        #設定這個檔案要處理的情報等級，只要是 debug 等級或以上的就寫入檔案
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        #關於 console(也就是cmd 那個黑黑的畫面)，請你按照下面的設定，幫我處理一下
        ch = logging.StreamHandler()
        #設定 Console 要處理的情報等級，只要是 Info 等級的就印出來
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        # log 印出來的格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #將 印出來的格式和 File Handle, Console Handle 物件組合在一起
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        #log 的分身組合 File Handle 和 Console Handle
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)


def unique_rows(arr):
    ''' Return unique rows of array a '''
    return  np.unique(arr.view(np.dtype((np.void, arr.dtype.itemsize*arr.shape[1])))).view(arr.dtype).reshape(-1, arr.shape[1])

def detectKepyPoints(img, octLayer, consTh, sigm):
    # find the keypoints and descriptors with SIFT
    # OcLayer = int(octLayer)
    # consTh = round(consTh, 4)
    # sigm = round(sigm, 2)

    detector = cv2.xfeatures2d.SIFT_create(nOctaveLayers = octLayer, contrastThreshold = consTh, sigma = sigm)
    kp = detector.detect(img)
    return kp

def computeDescriptors(img, kp, octLayer, consTh, sigm):
    # OcLayer = int(octLayer)
    # consTh = round(consTh, 4)
    # sigm = round(sigm, 2)

    descriptor = cv2.xfeatures2d.SIFT_create(nOctaveLayers = octLayer, contrastThreshold = consTh, sigma = sigm)
    kp, des = descriptor.compute(img, kp)
    return kp, des

def matchFeatures(kp, des, keypointsMatch, pairsMatchDistance):
    # BFMatcher with default params
    # cv2.NORM_L2 - SIFT, SURF
    # cv2.NORM_HAMMING - ORB, BRIEF, BRISK
    bf = cv2.BFMatcher(cv2.NORM_L2)
    #bf = cv2.BFMatcher(norm)

    # Calculate matches between a keypoint and k=3 more close keypoints
    # The first match is invalid, because is going to be the same Keypoint
    matches = bf.knnMatch(des, des, k=10)

    #matches = bf.knnMatch(np.asarray(des, np.float32), np.asarray(des, np.float32), k = 2)

    # We apply the Lowe and Amerini method to select good matches
    ratio  = keypointsMatch
    dismin = pairsMatchDistance
    # print(ratio, dismin)

    mkp1, mkp2 = [], []

    for m in matches:
        j = 1
        if(m[j].distance < ratio * m[j + 1].distance):


            if matches.index(m) == len(matches)-1:
                break
            else:
                j = j + 1

        for k in range(1, j):
            temp = m[k]

            # Check if the keypoints are spatial separated
            if pdist(np.array([kp[temp.queryIdx].pt,
                               kp[temp.trainIdx].pt])) > dismin:
                mkp1.append(kp[temp.queryIdx])
                mkp2.append(kp[temp.trainIdx])

    p1 = np.float32([kp1.pt for kp1 in mkp1])
    p2 = np.float32([kp2.pt for kp2 in mkp2])

    if len(p1) != 0:
        # Remove non-unique pairs of points
        p = np.hstack((p1, p2))
        p = unique_rows(p)
        p1 = np.float32(p[:, 0:2])
        p2 = np.float32(p[:, 2:4])

            #    t1 = time.time()
            #    print('Match Features: %f s' % (t1-t0))

    return p1, p2

def hierarchical_clustering(p, metric, th):
    ''' Compute the Hierarchical Agglomerative Cluster '''
    distance_p = pdist(p)
    Z = linkage(distance_p, metric)
    C = fcluster(Z, th, 'inconsistent', 4)

    return C

def compute_transformations(C, p, p1, min_cluster_pts, maxInlier):
    ''' Consider the cluster information,
        compute the number of transformations '''
    num_gt = 0
    num_ft = 0
    num_missft = 0

    c_max = np.max(C) # number of cluster
    if c_max > 1:
        for k, j in permutations(range(1, c_max+1), 2):
            z1 = []
            z2 = []
            for r in range(1, p1.shape[0]):
                if (C[r] == k) and (C[r + p1.shape[0]] == j):
                    z1.append(p[r, :])
                    z2.append(p[r+p1.shape[0], :])
                if (C[r] == j) and (C[r + p1.shape[0]] == k):
                    z1.append(p[r+p1.shape[0], :])
                    z2.append(p[r, :])

            z1 = np.array(z1)
            z2 = np.array(z2)

            # maxInlier = int(maxInlier)
            # print(maxInlier)
            if (len(z1) > min_cluster_pts) and (len(z2) > min_cluster_pts):
                (M, mask) = cv2.findHomography(z1, z2, cv2.RANSAC, ransacReprojThreshold = maxInlier) #, maxIters = 10

                # 要拿來跑只有inlier結果圖
                # matchesMask = mask.ravel().tolist()
                # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                #                    singlePointColor=None,
                #                    matchesMask=matchesMask,  # draw only inliers
                #                    flags=2)
                # img2 = cv2.drawMatches(img, p, img, p1, p1, None, **draw_params)
                # plt.imshow(img2), plt.show


                #----------output inlier and outlier ----------
                #print(mask),
                #print(mask.size)
                for i in range(mask.size):
                    if mask[i] != 0:
                        num_gt = num_gt + 1
                    elif mask[i] == 0:
                        num_ft = num_ft + 1
                    else:
                        num_missft = num_missft + 1

    return num_gt, num_ft

def plot_image(img, p1, p2, C):
    ''' Plot the image with keypoints and theirs matches '''

    plt.imshow(img, cmap=plt.get_cmap('gray'), interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.scatter(p1[:, 0],p1[:, 1], c=C[0:p1.shape[0]], s=30)

    for (x1, y1), (x2, y2) in zip(p1, p2):
        img_result = plt.plot([x1, x2], [y1, y2], 'c')

    plt.show()



    # MINE_MATCH_COUNT = 10
    # if (len(p1) > MINE_MATCH_COUNT) and (len(p2) > MINE_MATCH_COUNT):
    #     M, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    #     matchMask = mask.ravel().tolist()
    #     draw_params = dict(matchColor = (0, 255, 0), singlePointColor = None, matchMask = matchMask, flags = 2)
    #     img = cv2.drawMatches(img, p1, img, p2, )



def experiment():
    '''run experimet'''
    #Dataset




class PSOinit:
    def __init__(self):
        # Init PSO
        self.pop_pos = []
        self.velocity = []
        self.p_best_fitness = 0.0
        self.g_best_pos = []
        self.g_best_fitness = 0.0


    # Problem Definition

    # maximum ( Pmatch )
    # => minimum (1 / Pmatch)
    # Pmatch max = GoodMatchingKeypoints / GoodMatchingKeypoints + Threshold


    # mininum fitness( Pmatch )
    def minPmatch(self, img, metric, th, min_cluster_pts, plot):

        kp = detectKepyPoints(img, int(self.pop_pos[0]), round(self.pop_pos[1], 4), round(self.pop_pos[2], 2))
        kp, des = computeDescriptors(img, kp, int(self.pop_pos[0]), round(self.pop_pos[1], 4), round(self.pop_pos[2], 2))
        p1, p2 = matchFeatures(kp, des, round(self.pop_pos[3], 2), int(self.pop_pos[4]))

        # No matches - no geometric transformations - no tampering
        if len(p1) == 0:
            return 0
        else:

            p = np.vstack((p1, p2))

            # Hierarchical Agglomerative Clustering
            C = hierarchical_clustering(p, metric, th)

            # Compute number of transformations
            num_gt, num_ft= compute_transformations(C, p, p1, min_cluster_pts, int(self.pop_pos[5]))



            print('-------Pmatch----------')
            print(num_gt)

            print(num_ft)
            print('-------End Pmatch----------')
            if num_ft <= 10:
                    temp = num_gt + 10
                    Pmatch = num_gt / (float(temp))

            else:
                if num_gt != 0:
                    Pmatch = num_gt / (float(num_gt + num_ft))
                else:
                    Pmatch = 0

            if plot == True:
                plot_image(img, p1, p2, C)

            return Pmatch



  # assign random positions and velocities to the particles
    def Init(self, Lower_pos, Upper_pos, vmax, D, img, metric, th, min_cluster_pts, plot):

        for i in range(D):
            self.pop_pos.append(random.uniform(Lower_pos[i], Upper_pos[i]))  #可能需要限定幾位數
            self.velocity.append(random.uniform(round(-vmax, 1), round(vmax, 1)))
            #print(self.pop_pos)
        #print(vmax)
        #print(int(self.pop_pos[0]),  round(self.pop_pos[1], 4),  round(self.pop_pos[2], 2), round(self.pop_pos[3], 2), int(self.pop_pos[4]),  int(self.pop_pos[5]))
        self.minPmatch(img, metric, th, min_cluster_pts, plot)
        self.g_best_pos = copy.deepcopy(self.pop_pos)
        self.g_best_fitness = self.p_best_fitness

def Find_Best(Pparent, nPop):
    Best = copy.deepcopy(Pparent[0])
    for i in range(nPop):
        if Best.p_best_fitness < Pparent[i].p_best_fitness:
            Best = copy.deepcopy(Pparent[i])
    return Best

def load_dict(filename):
    ''' Load a python dictionary from a file '''
    temp = {}

    with open(filename, 'r') as f:

        for line in f:
            if '\n' in line:
                (key, val) = line.split('\n')
                #if not line.strip():
                    #continue
                temp[key] = val
    return temp

###    Demo   ###

def result():


    # #-------- 連 續 Run --------
    # DB = 'MICC-F220'
    # #DB_dir = './MICC-F220/*.jpg'
    # FILE_NAME = 'MICFF-220_list.txt'
    # retval = os.getcwd()
    # print("Current root: ", retval)
    # os.chdir(DB)
    # retval = os.getcwd()
    # print("Successfully change root:  %s" % retval)


    # img_path = load_dict(FILE_NAME)
    #
    # for imagefile in img_path.keys():
    #     img = cv2.imread(imagefile)

        #------ 單 張 Run ------
        img = cv2.imread('MICC-F220/CRW_4853tamp1.jpg')
        filename = 'CRW_4853tamp1.jpg'

        #Init parameters
        metric ='single'
        th = 2.2
        min_cluster_pts = 4

        # Init parameters for PSO
        w = 1
        c1 = 0.729
        c2 = 0.729
        nPop = 20           # number of particle
        D = 6               # dimension
        max_iter = 1
        Lower_pos = [3, 0.0001, 1.00, 0.01, 10,  1] # octLayer consTh sigma keyPointMatching minDistancePairs RANSAC
        Upper_pos = [6, 0.1000, 2.00, 1.00, 50, 10] # octLayer consTh sigma keyPointMatching minDistancePairs RANSAC

        vmax = 0.5


        Pparent = [PSOinit() for _ in range (nPop)]
        for i in range(nPop):
            Pparent[i].Init(Lower_pos, Upper_pos, vmax, D, img, metric, th, min_cluster_pts, False)
        Best = Find_Best(Pparent, nPop)
        for i in range(max_iter):
            Bestcurrent = Find_Best(Pparent, nPop)
            if Bestcurrent.p_best_fitness > Best.p_best_fitness:
                Best = copy.deepcopy(Bestcurrent)
            #print(Best.g_best_fitness)
            print('------------Updating Copy-Move Parameter in PSO----------------')
            for j in range(nPop):
                for k in range(D):
                     # calculate new velocity and set it in the [min, max] range
                    Pparent[j].velocity[k] = w * Pparent[j].velocity[k] + c1 * random.random() * (Best.pop_pos[k] - Pparent[j].pop_pos[k]) + c2 * random.random() * (Pparent[j].g_best_pos[k] - Pparent[j].pop_pos[k])
                    if abs(Pparent[j].velocity[k]) > vmax:
                        if Pparent[j].velocity[k] > 0:
                            Pparent[j].velocity[k] = vmax
                        else:
                            Pparent[j].velocity[k] = -vmax

                    if abs(Pparent[j].velocity[k]) < -vmax:
                        if Pparent[j].velocity[k] < 0:
                            Pparent[j].velocity[k] = -vmax
                        else:
                            Pparent[j].pop_pos[k] = -vmax

                         # calculate new positions and set it in the [min, max] range
                    Pparent[j].pop_pos[k] += Pparent[j].velocity[k]
                    #print(Pparent[j].pop_pos[k])

                    if Pparent[j].pop_pos[k] > Upper_pos[k] or Pparent[j].pop_pos[k] < Lower_pos[k]:
                             Pparent[j].pop_pos[k] = random.uniform(Lower_pos[k], Upper_pos[k])
                #print('\n')
                temp = copy.deepcopy(Pparent[j])
                Pparent[j].p_best_fitness = Pparent[j].minPmatch(img, metric, th, min_cluster_pts, True)
                # plt.show()
                # ResultFileName = str(j) + '_' + str(filename)
                #plt.savefig(ResultFileName, bbox_inches = 'tight')
                if Pparent[j].p_best_fitness > Pparent[j].g_best_fitness:
                               Pparent[j].g_best_fitness = Pparent[j].p_best_fitness
                               Pparent[j].g_best_pos = copy.deepcopy(Pparent[j].pop_pos)
                print('---------------PSO Gbest------------------')
                #print(Pparent[j].g_best_pos)
                print(int(Pparent[j].g_best_pos[0]), round(Pparent[j].g_best_pos[1], 4), round(Pparent[j].g_best_pos[2], 2),
                      round(Pparent[j].g_best_pos[3], 2), int(Pparent[j].g_best_pos[4]), int(Pparent[j].g_best_pos[5]))
                print(Pparent[j].g_best_fitness)


if __name__ == '__main__':
    result()