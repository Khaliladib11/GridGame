# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 08:53:22 2021

@author: Khaliladib
"""


"""
Priority class queue
"""

class PQ():
    
    # the constructor of the class, just intialize an empty array called queue 
    def __init__(self):
        self.queue = []
    
    # append new node to the array
    def put(self, node):
        self.queue.append(node)
    
    # search for the minimum node distance in the queue, delete it from the queue and return the node
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
    
    # method to check if the node is empty
    def isEmpty(self):
        return len(self.queue) == 0
    
    # return the size of the queue
    def qSize(self):
        return len(self.queue)
    
    # return the queue 
    def getQueue(self):
        return self.queue
    





