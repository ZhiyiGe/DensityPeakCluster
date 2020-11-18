#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class DistanceBuilder(object):
    """
    为聚类创建距离文件
    """

    def __init__(self):
        self.vectors = []

    def load_points(self, filename):
        """
        从文件中加载所有点(x dimension vectors)

        Args:
            filename : file's name that contains all points. Format is a vector one line, 每个维度的值用空格分开
        """
        with open(filename, 'r') as fp:
            for line in fp:
                self.vectors.append(list(map(float, line.strip().split(' '))))
        self.vectors = np.array(self.vectors)


    def build_distance_file_for_cluster(self, distance_obj, filename):
        """
        Save distance and index into file

        Args:
            distance_obj : distance.Distance object for 计算两点之间的距离
            filename     : file to save the result for cluster
        """
        fo = open(filename, 'w')
        for i in range(len(self.vectors) - 1):  # 前一个向量
            for j in range(i, len(self.vectors)): # 后一个向量
                fo.write(str(i + 1) + ' ' + str(j + 1) + ' ' +
                         str(distance_obj.distance(self.vectors[i], self.vectors[j])) + '\n')
        fo.close()
# end DistanceBuilder
