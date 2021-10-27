# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 08:53:22 2021

@author: Khaliladib
"""


class PQ():
    
    def __init__(self):
        self.queue = []
    
    def put(self, node):
        self.queue.append(node)
    
    def get(self):
        min = 0
        for i in range(len(self.queue)):
            current_node = self.queue[i]
            min_node = self.queue[min]
            if current_node.distance < min_node.distance:
                min = i
            
        node = self.queue[min]
        del self.queue[min]
        return node
    
    def isEmpty(self):
        return len(self.queue) == 0
    
    def qSize(self):
        return len(self.queue)
    
    def getQueue(self):
        return self.queue
    





