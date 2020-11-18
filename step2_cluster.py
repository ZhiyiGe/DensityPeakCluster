#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
from DensityPeakCluster.plot import *
from DensityPeakCluster.cluster import *


def plot(data, n_clusters, auto_select_dc=False):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dpcluster = DensityPeakCluster()
    rho, delta, nneigh, gamma, sort_gamma_idx = dpcluster.cluster(
        load_paperdata, data, n_clusters, auto_select_dc=auto_select_dc)
    logger.info(str(len(dpcluster.ccenter)) + ' center as below')
    for idx, center in dpcluster.ccenter.items():
        logger.info('%d %f %f' % (idx, rho[center], delta[center]))
    plot_rho_delta(rho, delta)   #plot to choose the threthold
    plot_scatter_diagram(0, np.arange(len(gamma)), gamma[sort_gamma_idx], x_label='index', y_label='vlaue', title='gamma')
    plot_cluster(dpcluster)


if __name__ == '__main__':
    plot('./data/data_in_paper/example_distances.dat',5, auto_select_dc=True)
    # plot(r'./data/data_iris_flower/iris.forcluster', 2, auto_select_dc=True)
