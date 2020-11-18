#! /usr/bin/env python
# -*- coding: utf-8 -*-


class WrongVecError(Exception):
    '''
    Raised when an operation use empty or not same size vector.

    Attributes:
        value: error info
    '''

    def __init__(self, value):
        self.value = value

    def __str__(self):
        # 返回对象的标准字符串形式
        return repr(self.value)
