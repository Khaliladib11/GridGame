# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 08:52:46 2021

@author: Khaliladib
"""

import sys


# This is class for node, it helps me to store the nodes data in array of node object
class Node:
    
    def __init__(self, i, j, weight):
        self.i = i
        self.j = j
        self.weight = weight
        self.pre = None
        self.distance = sys.maxsize
    
    def printInfo(self):
        print(f"node[{self.i}][{self.j}]. weight: {self.weight}. distance: {self.distance}. previous cell: {self.pre}")
    
    def print_pre(self):
        if self.pre is not None:
            self.pre.printInfo()
        