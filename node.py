# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 08:52:46 2021

@author: Khaliladib
"""

import sys

class Node:
    
    def __init__(self, i, j, weight):
        self.i = i
        self.j = j
        self.weight = weight
        self.pre = None
        self.distance = sys.maxsize
        