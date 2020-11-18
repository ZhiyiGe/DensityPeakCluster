#! /usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt
from abc import ABCMeta, abstractmethod
from .error_wrongvec import WrongVecError
from scipy.spatial import distance

import numpy as np
from numpy import linalg


class Distance():
    """
      抽象的类，展示两个向量的距离

      Attributes:
      """

    __metaclass__ = ABCMeta

    @abstractmethod
    def distance(self, vec1, vec2):
        """
        计算两个向量之间的距离(一个线性的 numpy 数组)
        如果你用 scipy 存储稀疏矩阵, 请使用 s.getrow(line_num).toarray() 建立一维数组

        Args:
            vec1: 第一个行向量, 一个数组实例
            vec2: 第二个行向量，一个数组实例

        Returns:
            计算的距离

        Raises:
            TypeError: if vec1 or vec2 is not numpy.ndarray and one line array
        """
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            raise TypeError("type of vec1 or vec2 is not numpy.ndarray")
        if vec1.ndim is not 1 or vec2.ndim is not 1:
            raise WrongVecError("vec1 or vec2 is not one line array")
        if vec1.size != vec2.size:
            raise WrongVecError("vec1 or vec2 is not same size")
        pass


# end Distance


class PearsonDistance(Distance):
    """
    皮尔森距离

    Distance 的一个子类
    """

    def distance(self, vec1, vec2):
        """
        Compute distance of two vector by pearson distance
        """
        super(PearsonDistance, self).distance(vec1, vec2)  # super method
        return distance.correlation(vec1, vec2)
# end PearsonDistance


class ConsineDistance(Distance):
    """
    consine distance  余弦距离

    a sub class of Distance
    """

    def distance(self, vec1, vec2):
        """
        Compute distance of two vector by consine distance
        """
        super(ConsineDistance, self).distance(vec1, vec2)  # super method
        return distance.cosine(vec1, vec2)
# end ConsineDistance
