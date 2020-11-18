from sklearn import manifold
from sko.PSO import PSO
import math
import pandas as pd
from DensityPeakCluster.distance.distance import *
from DensityPeakCluster.distance.distence_builder_iron import data_process
from DensityPeakCluster.plot_utils import *
from sklearn import __version__ as sklearn_version


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


class Builder(object):
    def __init__(self, ncomponents, neighbor):
        super().__init__()
        self.neighbor = neighbor
        self.ncomponents = ncomponents
        self.distance = {}

    def load_points(self, filename):
        data = pd.read_excel(filename).drop('时间戳', axis=1)  # 读取数据
        result = data['聚类结果']
        vecoters = data.drop('聚类结果', axis=1).values
        self.vectors = vecoters
        self.result = result
        self.data_preprocess()

    def data_preprocess(self):
        processer = data_process(self.vectors)
        processer.PCA_dim_reduce(self.ncomponents)
        # processer.Isomap_dim_reduce(self.ncomponents, self.neighbor)
        processer.normalize()
        self.vectors = processer.data

    def build_distance_for_cluster(self, distance_obj):
        n = len(self.vectors)
        for i in range(n - 1):  # 前一个向量
            self.distance[(i + 1, i + 1)] = 0.0
            for j in range(i, n):  # 后一个向量
                dis = distance_obj.distance(self.vectors[i], self.vectors[j])
                self.distance[(i + 1, j + 1)] = dis
                self.distance[(j + 1, i + 1)] = dis
        self.distance[(n, n)] = 0


def select_dc(max_id, max_dis, min_dis, distances, auto=False):
    if auto:
        return autoselect_dc(max_id, max_dis, min_dis, distances)
    percent = 2.0  # 百分之几
    position = int(max_id * (max_id + 1) / 2 * percent / 100)
    dc = sorted(distances.values())[position * 2 + max_id]  # 前max_id个值为0
    return dc


def autoselect_dc(max_id, max_dis, min_dis, distances):
    dc = (max_dis + min_dis) / 2

    while True:
        nneighs = sum([1 for v in distances.values() if v < dc]) / max_id ** 2
        if 0.01 <= nneighs <= 0.02:
            break
        # binary search
        if nneighs < 0.01:
            min_dis = dc
        else:
            max_dis = dc
        dc = (max_dis + min_dis) / 2
        if max_dis - min_dis < 0.0001:
            break
    return dc


def local_density(max_id, distances, dc, guass=True, cutoff=False):
    assert guass ^ cutoff  # 断言 若异常返回错误
    guass_func = lambda dij, dc: math.exp(- (dij / dc) ** 2)
    cutoff_func = lambda dij, dc: 1 if dij < dc else 0
    func = guass and guass_func or cutoff_func
    rho = [-1] + [0] * max_id
    for i in range(1, max_id):
        for j in range(i + 1, max_id + 1):
            rho[i] += func(distances[(i, j)], dc)
            rho[j] += func(distances[(i, j)], dc)
    return np.array(rho, np.float32)


def min_distance(max_id, max_dis, distances, rho):
    sort_rho_idx = np.argsort(-rho)
    delta, nneigh = [0.0] + [float(max_dis)] * (len(rho) - 1), [0] * len(rho)
    delta[sort_rho_idx[0]] = -1.
    for i in range(1, max_id):
        for j in range(0, i):
            old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]
            if distances[(old_i, old_j)] < delta[old_i]:
                delta[old_i] = distances[(old_i, old_j)]
                nneigh[old_i] = old_j
    delta[sort_rho_idx[0]] = max(delta)
    nneigh[sort_rho_idx[0]] = sort_rho_idx[0]
    gamma = rho * delta
    return np.array(delta, np.float32), np.array(nneigh, int), gamma


class DensityPeakCluster(object):

    def local_density(self, data, dc=None, auto_select_dc=False):
        assert not (dc is not None and auto_select_dc)
        distances = data
        max_dis = max(distances.values())
        min_dis = min(distances.values())
        max_id = 0
        for i, j in distances:
            max_id = max(max_id, i, j)
        if dc is None:
            dc = select_dc(max_id, max_dis, min_dis,
                           distances, auto=auto_select_dc)
        rho = local_density(max_id, distances, dc)
        return distances, max_dis, min_dis, max_id, rho

    def cluster(self, data, n_clusters, dc=None, auto_select_dc=False):
        assert not (dc is not None and auto_select_dc)
        distances, max_dis, min_dis, max_id, rho = self.local_density(
            data, dc=dc, auto_select_dc=auto_select_dc)
        delta, nneigh, gamma = min_distance(max_id, max_dis, distances, rho)
        sort_rho_idx = np.argsort(-rho)
        ccluster, ccenter = {}, {}  # cl/icl in cluster_dp.m
        sort_gamma_idx = np.argsort(-gamma)
        for i in range(n_clusters):
            idx = sort_gamma_idx[i]
            ccenter[idx] = idx
            ccluster[idx] = i + 1
        for idx in sort_rho_idx:
            nneigh_item = nneigh[idx]
            if idx == 0 or idx in ccluster:
                continue
            ccluster[idx] = ccluster[nneigh_item]

        self.ccluster, self.ccenter = ccluster, ccenter
        self.distances = distances
        self.max_id = max_id
        return rho, delta, nneigh, gamma, sort_gamma_idx


def plot_cluster(cluster):
    dp = np.zeros((cluster.max_id, cluster.max_id), dtype=np.float32)
    cls = []
    for i in range(1, cluster.max_id):
        for j in range(i + 1, cluster.max_id + 1):
            dp[i - 1, j - 1] = cluster.distances[(i, j)]
            dp[j - 1, i - 1] = cluster.distances[(i, j)]
        cls.append(cluster.ccluster[i])
    cls.append(cluster.ccluster[cluster.max_id])
    version = versiontuple(sklearn_version)
    if version[0] > 0 or version[1] > 14:
        mds = manifold.MDS(max_iter=200, eps=1e-4, n_init=1,
                           dissimilarity='precomputed')
    else:
        mds = manifold.MDS(max_iter=200, eps=1e-4, n_init=1)
    dp_mds = mds.fit_transform(dp)
    plot_scatter_diagram(1, dp_mds[:, 0], dp_mds[:, 1], title='cluster', style_list=cls)
    result = []
    for p in dp_mds:
        if p[0]>0:
            result.append(2)
        else:
            result.append(1)
    return result



def cluster(data, n_clusters, auto_select_dc=False):
    dpcluster = DensityPeakCluster()
    dpcluster.cluster(data, n_clusters, auto_select_dc=auto_select_dc)
    result = dpcluster.ccluster
    result2 = plot_cluster(dpcluster)
    return result,result2


def optimize_func(ncomponents, neighbor):
    builder = Builder(ncomponents, neighbor)
    builder.load_points(r'./data/data_iron/data.xlsx')
    builder.build_distance_for_cluster(PearsonDistance())
    result_true = builder.result
    distences = builder.distance
    result_predict, result2 = cluster(distences, 2, auto_select_dc=True)
    n = len(result_true)
    count = 0
    count2 = 0
    for i in range(n):
        if result_true[i] != result_predict[i + 1]:
            count += 1
        if result_true[i] != result2[i]:
            count2 += 1
    return count / n, count2/n


def PSO_func(x):
    x1 = x[0]
    x1 = int(round(x1 * 63) + 2)
    # x2 = int(round(x2 * 199) + 1)
    print((x1))
    return optimize_func(x1, 20)


if __name__ == '__main__':
    # pso = PSO(func=PSO_func, dim=1, pop=20, max_iter=40, lb=[0], ub=[1])
    # pso.record_mode = True
    # pso.run()
    # print('best_x is ', list(pso.gbest_x), 'best_y is ', pso.gbest_y)
    min_err = 1
    idx_min_err = 0
    err1,err2 = optimize_func(2, 0)
    # 0.9480968858131488
    print(err1,1-err2)
