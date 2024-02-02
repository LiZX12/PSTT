
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt

import numpy as np
import random

from joblib.numpy_pickle_utils import xrange


import kmedoids


def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in xrange(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C

def distance(point1, point2):  # 计算距离（欧几里得距离）
    return np.sqrt(np.sum((point1 - point2) ** 2))


def k_means(data, k, max_iter=10000):
    centers = {}  # 初始聚类中心
    # 初始化，随机选k个样本作为初始聚类中心。 random.sample(): 随机不重复抽取k个值
    n_data = data.shape[0]  # 样本个数
    for idx, i in enumerate(random.sample(range(n_data), k)):
        # idx取值范围[0, k-1]，代表第几个聚类中心;  data[i]为随机选取的样本作为聚类中心
        centers[idx] = data[i]

        # 开始迭代
    for i in range(max_iter):  # 迭代次数
        print("开始第{}次迭代".format(i + 1))
        clusters = {}  # 聚类结果，聚类中心的索引idx -> [样本集合]
        labels = {}  # 聚类结果，聚类中心的索引idx -> [样本集合]
        for j in range(k):  # 初始化为空列表
            clusters[j] = []
            labels[j] = []

        for iter,sample in enumerate(data):   # 遍历每个样本
            distances = []  # 计算该样本到每个聚类中心的距离 (只会有k个元素)
            for c in centers:  # 遍历每个聚类中心
                # 添加该样本点到聚类中心的距离
                distances.append(distance(sample, centers[c]))
            idx = np.argmin(distances)  # 最小距离的索引
            clusters[idx].append(sample)  # 将该样本添加到第idx个聚类中心
            labels[idx].append(iter)

        pre_centers = centers.copy()  # 记录之前的聚类中心点

        for c in clusters.keys():
            # 重新计算中心点（计算该聚类中心的所有样本的均值）
            centers[c] = np.mean(clusters[c], axis=0)

        is_convergent = True
        for c in centers:
            if distance(pre_centers[c], centers[c]) > 1e-8:  # 中心点是否变化
                is_convergent = False
                break
        if is_convergent == True:
            # 如果新旧聚类中心不变，则迭代停止
            break
    return centers, clusters ,labels


def predict(p_data, centers):  # 预测新样本点所在的类
    # 计算p_data 到每个聚类中心的距离，然后返回距离最小所在的聚类。
    distances = [distance(p_data, centers[c]) for c in centers]
    return np.argmin(distances)

from sklearn.metrics.pairwise import pairwise_distances
if __name__ =='__main__':
    result = scipy.io.loadmat(r'H:\program\Spatial-Temporal-Re-identification-master\model\fs_ResNet50_pcb_market_e\pytorch_result_unalbel_label.mat')
    samples = result['query_f']
    # # samples = np.loadtxt("kmeansSamples.txt")
    # clusterCents,SSE,labels = k_means(samples[:,200],751,5)
    # plt.show()
    # # print(clusterCents)
    # # print(SSE)
    # print(labels)

    # 3 points in dataset
    data = np.array([[1, 1],
                     [1, 2],
                     [3, 1],
                     [2, 2],
                     [5, 5],
                     [40, 40],
                     [35, 45],
                     [1, 1]])

    # distance matrix
    D = pairwise_distances(data, metric='euclidean')

    c = kmedoids.fasterpam(D, 3)
    print("Labels is:", c.labels)
    #
    # # split into 2 clusters
    # M, C =kMedoids(D, 3)
    #
    # print('medoids:')
    # for point_idx in M:
    #     print(data[point_idx])
    #
    # print('')
    # print('clustering result:')
    # for label in C:
    #     for point_idx in C[label]:
    #         print('label {0}:　{1}'.format(label, data[point_idx]))

