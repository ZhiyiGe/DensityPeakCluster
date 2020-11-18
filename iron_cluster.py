#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import numpy as np
from DensityPeakCluster.plot import *
from DensityPeakCluster.cluster import *


def load_result(filename):
    s = []
    f = open(filename, 'r')  # 由于我使用的pycharm已经设置完了路径，因此我直接写了文件名
    for lines in f:
        result = int(lines.strip())
        s.append(result)
    f.close()
    return np.array(s)


def cluster(data, n_clusters, auto_select_dc=False):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dpcluster = DensityPeakCluster()
    rho, delta, nneigh, gamma, sort_gamma_idx = dpcluster.cluster(
        load_paperdata, data, n_clusters, auto_select_dc=auto_select_dc)
    logger.info(str(len(dpcluster.ccenter)) + ' center as below')
    for idx, center in dpcluster.ccenter.items():
        logger.info('%d %f %f' % (idx, rho[center], delta[center]))
    plot_rho_delta(rho, delta)  # plot to choose the threthold
    plot_scatter_diagram(0, np.arange(len(gamma)), gamma[sort_gamma_idx], x_label='index', y_label='vlaue',
                         title='gamma')
    result = dpcluster.ccluster
    plot_cluster(dpcluster)
    return result


if __name__ == '__main__':
    # plot('./data/data_in_paper/example_distances.dat', 20, 0.1)
    result_predict = cluster(r'./data/data_iron/distence.dat', 2, auto_select_dc=True)
    result_true = load_result(r'./data/data_iron/result.txt')
    n = len(result_true)
    count = 0
    for i in range(n):
        if result_true[i] == result_predict[i + 1]:
            count += 1
        else:
            print(i+1,result_true[i],result_predict[i + 1])
    print(count / n)
