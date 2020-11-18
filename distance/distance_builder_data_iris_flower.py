#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# data reference : R. A. Fisher (1936). "The use of multiple measurements
# in taxonomic problems"

from DensityPeakCluster.distance.distance_builder import *
from DensityPeakCluster.distance.distance import *


if __name__ == '__main__':
    builder = DistanceBuilder()
    builder.load_points(r'../data/data_iris_flower/iris.data')
    # 计算余弦距离变更保存
    builder.build_distance_file_for_cluster(
        PearsonDistance(), r'../data/data_iris_flower/iris.forcluster')

