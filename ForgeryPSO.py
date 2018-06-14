from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from itertools import permutations
import numpy as np
import random
import copy
import cv2
import sys


def unique_rows(arr):
    ''' Return unique rows of array a '''
    return  np.unique(arr.view(np.dtype((np.void, \
                arr.dtype.itemsize*arr.shape[1])))) \
                .view(arr.dtype).reshape(-1, arr.shape[1])

def detectKepyPoints(img, octLayer, consTh, sigm):
    # find the keypoints and descriptors with SIFT
    OcLayer = int (octLayer)
    detector = cv2.xfeatures2d.SIFT_create(nOctaveLayers = OcLayer, contrastThreshold = consTh, sigma = sigm)
    kp = detector.detect(img)
    return kp

def computeDescriptors(img, kp, octLayer, consTh, sigm):
    OcLayer = int(octLayer)
    descriptor = cv2.xfeatures2d.SIFT_create(nOctaveLayers=OcLayer, contrastThreshold=consTh, sigma=sigm)
    kp, des = descriptor.compute(img, kp)
    return kp, des

def matchFeatures(kp, des):
    # BFMatcher with default params
    # cv2.NORM_L2 - SIFT, SURF
    # cv2.NORM_HAMMING - ORB, BRIEF, BRISK
    bf = cv2.BFMatcher(cv2.NORM_L2)
    #bf = cv2.BFMatcher(norm)

    # Calculate matches between a keypoint and k=3 more close keypoints
    # The first match is invalid, because is going to be the same Keypoint
    matches = bf.knnMatch(des, des, k=10)

    # We apply the Lowe and Amerini method to select good matches
    ratio = 0.5
    mkp1, mkp2 = [], []
    for m in matches:
        j = 1
        while (m[j].distance < ratio * m[j + 1].distance):
            j = j + 1

        for k in range(1, j):
            temp = m[k]

            # Check if the keypoints are spatial separated
            if pdist(np.array([kp[temp.queryIdx].pt,
                               kp[temp.trainIdx].pt])) > 10: #dis min
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

def compute_transformations(C, p, p1, min_cluster_pts):
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
            for r in range(1,p1.shape[0]):
                if (C[r] == k) and (C[r + p1.shape[0]] == j):
                    z1.append(p[r, :])
                    z2.append(p[r+p1.shape[0], :])
                if (C[r] == j) and (C[r + p1.shape[0]] == k):
                    z1.append(p[r+p1.shape[0], :])
                    z2.append(p[r, :])

            z1 = np.array(z1)
            z2 = np.array(z2)


            if (len(z1) > min_cluster_pts) and (len(z2) > min_cluster_pts):
                M,_ = cv2.findHomography(z1, z2, cv2.RANSAC, 5.0)

               # print('\n')
               # print(M)

                if (np.any(M) != None)  and (len(M) != 0):
                        num_gt = num_gt + 1
                elif (np.any(M) != None) and (len(M) == 0):
                        num_ft = num_ft + 1
                else: num_missft = num_missft + 1

    return num_gt, num_ft

def plot_image(img, p1, p2, C):
    ''' Plot the image with keypoints and theirs matches '''
    plt.imshow(img, cmap=plt.get_cmap('gray'), interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.scatter(p1[:, 0],p1[:, 1], c=C[0:p1.shape[0]], s=30)

    for (x1, y1), (x2, y2) in zip(p1, p2):
        plt.plot([x1, x2],[y1, y2], 'c')

    plt.show()






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
    # Pmatch min = GoodMatchingKeypoints / GoodMatchingKeypoints + Threshold


    # mininum fitness( Pmatch )
    def minPmatch(self, img, metric, th, min_cluster_pts):

        kp = detectKepyPoints(img, self.pop_pos[0],  self.pop_pos[1],  self.pop_pos[2])
        kp, des = computeDescriptors(img, kp, self.pop_pos[0],  self.pop_pos[1],  self.pop_pos[2])
        p1, p2 = matchFeatures(kp, des)

        # No matches - no geometric transformations - no tampering
        if len(p1) == 0:
            return 0
        else:
            p = np.vstack((p1, p2))

            # Hierarchical Agglomerative Clustering
            C = hierarchical_clustering(p, metric, th)

            # Compute number of transformations
            num_gt, num_ft = compute_transformations(C, p, p1, min_cluster_pts)
            print(num_gt)
            print(num_ft)
            Pmatch = 0
        if num_ft <= 10:
                temp = num_gt + 10
                Pmatch = float(num_gt / temp)
        else:   Pmatch = float(num_gt / num_gt + num_ft)



        return Pmatch


  # assign random positions and velocities to the particles
    def Init(self, Lower_pos, Upper_pos, vmax, D, img, metric, th, min_cluster_pts):

        for i in range(D):
            self.pop_pos.append(random.uniform(Lower_pos[i], Upper_pos[i]))
            self.velocity.append(random.uniform(-vmax, vmax))
            #print(self.pop_pos)
        self.minPmatch(img, metric, th, min_cluster_pts)
        self.g_best_pos = copy.deepcopy(self.pop_pos)
        self.g_best_fitness = self.p_best_fitness

def Find_Best(Pparent, nPop):
    Best = copy.deepcopy(Pparent[0])
    for i in range(nPop):
        if Best.p_best_fitness < Pparent[i].p_best_fitness:
            Best = copy.deepcopy(Pparent[i])
    return Best


###    Demo   ###

def result():

    #Init parameters

    img = cv2.imread("F.png")
    metric ='single'
    th = 2.2
    min_cluster_pts = 4

    # Init parameters for PSO
    w = 1
    c1 = 0.729
    c2 = 0.729
    nPop = 20           # number of particle
    D = 3               # dimension
    max_iter = 100
    Lower_pos = [3, 0.0001, 1.00] # octLayer consTh sigm
    Upper_pos = [6, 0.1000, 2.00] # octLayer consTh sigm

    vmax = 0.5


    Pparent = [PSOinit() for _ in range (nPop)]
    for i in range(nPop):
        Pparent[i].Init(Lower_pos, Upper_pos, vmax, D, img, metric, th, min_cluster_pts)
    Best = Find_Best(Pparent, nPop)
    for i in range(max_iter):
        Bestcurrent = Find_Best(Pparent, nPop)
        if Bestcurrent.p_best_fitness > Best.p_best_fitness:
            Best = copy.deepcopy(Bestcurrent)
        print(Best.g_best_fitness)
        for j in range(nPop):
            for k in range(D):
                 # calculate new velocity and set it in the [min, max] range
                Pparent[j].velocity[k] = w * Pparent[j].velocity[k] + c1 * random.random() * (Best.pop_pos[k] - Pparent[j].pop_pos[k]) + c2 * random.random() * (Pparent[j].g_best_pos[k] - Pparent[j].pop_pos[k])
                if abs(Pparent[j].velocity[k]) > vmax:
                    if Pparent[j].velocity[k] > 0:
                        Pparent[j].velocity[k] = vmax

                if abs(Pparent[j].velocity[k]) < -vmax:
                    if Pparent[j].velocity[k] < 0:
                        Pparent[j].velocity[k] = -vmax

                     # calculate new positions and set it in the [min, max] range
                Pparent[j].pop_pos[k] += Pparent[j].velocity[k]

                if Pparent[j].pop_pos[k] > Upper_pos[k] or Pparent[j].pop_pos[k] < Lower_pos[k]:
                         Pparent[j].pop_pos[k] = random.uniform(Lower_pos[k], Upper_pos[k])
            temp = copy.deepcopy(Pparent[j])
            Pparent[j].minPmatch(img, metric, th, min_cluster_pts)
            if Pparent[j].p_best_fitness > Pparent[j].g_best_fitness:
                           Pparent[j].g_best_fitness = Pparent[j].p_best_fitness
                           Pparent[j].g_best_pos = copy.deepcopy(Pparent[j].pop_pos)
                           print(Pparent[j].g_best_pos)


if __name__ == '__main__':
    result()