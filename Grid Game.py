# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:57:03 2021

@author: Khaliladib
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from numpy.random import default_rng
from queue import PriorityQueue

class GridGame(ABC):
    
    def __init__(self, height=5, width=5, n=9):
        
        self.height = height
        self.width = width
        self.n = n
        self.grid = []
        
    
    def buildTheGame(self):
        
        rng = default_rng()
        self.grid = rng.integers(0, self.n, (self.height, self.width))
        self.grid[0][0] = 0
        return self.grid
    
    def visualizeTheGrid(self, grid):
        height = len(grid)
        width = len(grid[0])
        title = "Grid Game"
        xlabel = "Width"
        ylabel = "Heigth"
        plt.figure(figsize=(height, width))
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tick_params(labelsize=14)
        img = plt.imshow(grid)
        plt.colorbar(img)
        fontSize=14
        if width > 10 and height > 10:
            fontSize = 25
        for y in range(height):
            for x in range(width):
                plt.text(x , y, '%.f' % self.grid[y, x],
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=fontSize
              )
        plt.show()
    
    def get_adjacent(self, node):
        
        i = node[0]
        j = node[1]
        neighbors = []
        
        if i == 0 and j == 0:
            neighbors.append([i, j+1, self.grid[i][j+1]])
            neighbors.append([i+1, j, self.grid[i+1][j]])
        
        elif i > 0 and i < self.height-1 and j == 0:   
            neighbors.append([i-1, j, self.grid[i-1][j]])
            neighbors.append([i+1, j, self.grid[i+1][j]])
            neighbors.append([i, j+1, self.grid[i][j+1]])
        
        elif i == self.height-1 and j == 0:
            neighbors.append([i-1, j, self.grid[i-1][j]])
            neighbors.append([i, j+1, self.grid[i][j+1]])
        
        elif i == 0 and j > 0 and j < self.width-1:
            neighbors.append([i+1, j, self.grid[i+1][j]])
            neighbors.append([i, j-1, self.grid[i][j-1]])
            neighbors.append([i, j+1, self.grid[i][j+1]])
        
        elif i == 0 and j == self.width-1:
            neighbors.append([i+1, j, self.grid[i+1][j]])
            neighbors.append([i, j-1, self.grid[i][j-1]])
        
        elif i > 0 and i < self.height-1 and j == self.width-1:
            neighbors.append([i-1, j, self.grid[i-1][j]])
            neighbors.append([i+1, j, self.grid[i+1][j]])
            neighbors.append([i, j-1, self.grid[i][j-1]])
        
        elif i == self.height-1 and j > 0 and j < self.width-1:
            neighbors.append([i-1, j, self.grid[i-1][j]])
            neighbors.append([i, j+1, self.grid[i][j+1]])
            neighbors.append([i, j-1, self.grid[i][j-1]])
        
        elif i == self.height-1 and j == self.width-1:
            neighbors.append([i-1, j, self.grid[i-1][j]])
            neighbors.append([i, j-1, self.grid[i][j-1]])
        
        else:
            neighbors.append([i-1, j, self.grid[i-1][j]])
            neighbors.append([i+1, j, self.grid[i+1][j]])
            neighbors.append([i, j-1, self.grid[i][j-1]])
            neighbors.append([i, j+1, self.grid[i][j+1]])
        
        return neighbors
    
    @abstractmethod
    def findPath(self, grid, i=0, j=0, path=[[0, 0]]):
        pass
    
    @abstractmethod
    def computePath(self, grid, path):
        pass
    
    @abstractmethod
    def dijkstra(self, grid):
        pass


class Node:
    
    def __init__(self, i, j, w):
        self.i = i
        self.j = j
        self.w = w
        self.pre = None
        self.distance = sys.maxsize
        



class numberInCellMode(GridGame):
    
    def findPath(self, grid, i=0, j=0, path=[[0, 0]]):
        if i < len(grid)-1 and j < len(grid[0])-1:
            if grid[i+1][j] >= grid[i][j+1]:
                path.append([i, j+1])
                return self.findPath(grid, i, j+1, path)
            else:
                path.append([i+1, j])
                return self.findPath(grid, i+1, j, path)
        
        elif i == len(grid)-1 and j < len(grid[0])-1:
            path.append([i, j+1])
            return self.findPath(grid, i, j+1, path)
            
        elif j == len(grid[0])-1 and i < len(grid)-1:
            path.append([i+1, j])
            return self.findPath(grid, i+1, j, path)
        else:
            return path
    
    def computePath(self, grid, path):
        newGrid = np.copy(grid)
        cost = 0
        
        for cell in path:
            i = cell[0]
            j = cell[1]
            newGrid[i][j] = -10
            cost += grid[i][j]
            
        print(f"The total cost is: {cost}")
        self.visualizeTheGrid(newGrid)
    

    def dijkstra(self, grid):
        
        distances = np.ones((self.height, self.width), dtype=int)*sys.maxsize
        distances[0][0] = 0
        visited = []
        path_to_destination= []
        path = np.empty( (self.height, self.width), dtype=object)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                node = Node(i, j, grid[i][j])
                path[i][j] = node
               
       
        pq = PriorityQueue()
        pq.put((0, (0, 0, 0)))
        
        while not pq.empty():
            
                distance, current_cell = pq.get()
                visited.append(current_cell)
            
                current_i = current_cell[0]
                current_j = current_cell[1]
                
                for neighbor in self.get_adjacent([current_i, current_j]):
                    i = neighbor[0]
                    j = neighbor[1]
                    newDistance = neighbor[2]
                    if neighbor not in visited:
                        old_cost = distances[i][j]
                        new_cost = distances[current_i][current_j] + newDistance
                        if new_cost < old_cost:
                                pq.put((new_cost, [i, j, newDistance]))
                                distances[i][j] = new_cost
                                path[i][j].pre = [current_i, current_j]
        
        
        print(distances)
        for i in range(len(path)):
            for j in range(len(path[0])):
                           print(f"before [{i}, {j}]: {path[i][j].pre}")
        
        i = len(grid)-1
        j = len(grid[0])-1
        complate = False
        while not complate:
            path_to_destination.append([i, j])
            if path[i][j].pre is not None:
                k = i
                x = j
                i = path[k][x].pre[0]
                j = path[k][x].pre[1]
            else: complate = True
        
        return path_to_destination
    
class AbsoluteValueMode(GridGame):

    def findPath(self, grid, i=0, j=0, path=[[0, 0]]):
        if i < len(grid)-1 and j < len(grid[0])-1:
            if abs(grid[i][j] - grid[i+1][j]) >= abs(grid[i][j] - grid[i][j+1]):
                path.append([i, j+1])
                return self.findPath(grid, i, j+1, path)
            else:
                path.append([i+1, j])
                return self.findPath(grid, i+1, j, path)
        
        elif i == len(grid)-1 and j < len(grid[0])-1:
           path.append([i, j+1])
           return self.findPath(grid, i, j+1, path)
            
        elif j == len(grid[0])-1 and i < len(grid)-1:
            path.append([i+1, j])
            return self.findPath(grid, i+1, j, path)
        else:
            return path
    
    def computePath(self, grid, path):
        newGrid = np.copy(grid)
        prevCost = grid[0][0]
        newGrid[0][0] = -10
        currentCost = 0
        cost = 0
        for cell in range(1, len(path)):
            prevI = path[cell-1][0]
            prevJ = path[cell-1][1]
            currentI = path[cell][0]
            currentJ = path[cell][1]
            
            prevCost = grid[prevI][prevJ]
            currentCost = grid[currentI][currentJ]
            
            cost += abs(prevCost - currentCost)
            newGrid[currentI][currentJ] = -10
            
        print(f"The total cost is: {cost}")
        self.visualizeTheGrid(newGrid)
                
    def dijkstra(self, grid):
        distances = np.ones((self.height, self.width), dtype=int)*sys.maxsize
        distances[0][0] = 0
        visited = []
        path_to_destination= []
        path = np.empty( (self.height, self.width), dtype=object)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                node = Node(i, j, grid[i][j])
                path[i][j] = node
               
       
        pq = PriorityQueue()
        pq.put((0, (0, 0, 0)))
        
        while not pq.empty():
            
                distance, current_cell = pq.get()
                visited.append(current_cell)
            
                current_i = current_cell[0]
                current_j = current_cell[1]
                
                for neighbor in self.get_adjacent([current_i, current_j]):
                    i = neighbor[0]
                    j = neighbor[1]
                    newDistance = neighbor[2]
                    if neighbor not in visited:
                        old_cost = distances[i][j]
                        new_cost = abs(distances[current_i][current_j] - newDistance)
                        if new_cost < old_cost:
                                pq.put((new_cost, [i, j, newDistance]))
                                distances[i][j] = new_cost
                                path[i][j].pre = [current_i, current_j]
        
        
        print(distances)
        for i in range(len(path)):
            for j in range(len(path[0])):
                           print(f"before [{i}, {j}]: {path[i][j].pre}")
        
        i = len(grid)-1
        j = len(grid[0])-1
        complate = False
        while not complate:
            path_to_destination.append([i, j])
            if path[i][j].pre is not None:
                k = i
                x = j
                i = path[k][x].pre[0]
                j = path[k][x].pre[1]
            else: complate = True
        
        return path_to_destination


"""
modeA = numberInCellMode()
modeB = AbsoluteValueMode()
grid = modeA.buildTheGame()
print(grid)
path = modeA.findPath(grid)
modeA.computePath(grid, path)
path = modeB.findPath(grid)
modeB.computePath(grid, path)
"""


gridGame = numberInCellMode(10, 10)
grid = gridGame.buildTheGame()
print(grid)
gridGame.visualizeTheGrid(grid)
path = gridGame.dijkstra(grid)
path = list(reversed(path))
print(path)
gridGame.computePath(grid, path)

#path = gridGame.dijkstra(grid)



"""
i = 4
j = 4
complate = False
test = []
while not complate:
    test.append([i, j])
    if path[i][j].pre is not None:
        k = i
        x = j
        i = path[k][x].pre[0]
        j = path[k][x].pre[1]
    else: complate = True

result=list(reversed(test))
print(result)

gridGame.computePath(grid, result)   
"""
        
        