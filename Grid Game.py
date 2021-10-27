# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:57:03 2021

@author: Khaliladib
"""

# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from numpy.random import default_rng
from node import Node
from priorityQueue import PQ


# The main class, which contains the necessary methods and properties, it is an abstract class for the game's two modes.
class GridGame(ABC):
    
    # constructor method to iniialize the game with the height and width an the max number a cell can take, finally create an empty array
    def __init__(self, height=5, width=5, n=9):
        
        self.height = height
        self.width = width
        self.n = n
        self.grid = []
        
    
    # This method is responsible for build the game, essentially filling the grid with numbers
    def buildTheGame(self):
        
        #usign the default distribution
        rng = default_rng()
        self.grid = rng.integers(0, self.n, (self.height, self.width))
        #assign the value in the cell[0][0] to 0 and return the grid
        self.grid[0][0] = 0
        return self.grid
    
    # This method is reponsible for visualize the gird using matplotlib library, it takes as parameter the grid
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
        #changing the font according to the size of the grid, and also visalize the number it each cell
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
    
    
    # this methoc checks for boundaris and return the adjacent cell to each cell, it takes as input the node it self
    def get_adjacent(self, node):
        
        # getting i and j from the node
        i = node[0]
        j = node[1]
        neighbors = []
        
        #start loking for neighbors of each cell, add them to the neighbors array, then return the array at the end
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
    
    
    # The abstract method needs to be implemented in the subclasses, it is represents the naive approach for the task. more details in the subclasses
    @abstractmethod
    def findPath(self, grid, i=0, j=0, path=[[0, 0]]):
        pass
    
    
    # Second abstract method to be implemented, it is reponsible for compute the pasth and return the cost.
    @abstractmethod
    def computePath(self, grid, path):
        pass
    
    
    # Last abstract method for this class, the Dijkstra algorithm 
    @abstractmethod
    def dijkstra(self, grid):
        pass



# The first subclass, this class will inherit the properties and methods from the GridGame class. 
#In this mode, the cost of each cell is the number inside the cell itself
class numberInCellMode(GridGame):
    
    """
    findPath method to find the path in the naive approach. Basically, it uses the recursion approach to find the shortest path. 
    It takes as input the grid, the postion i j of the current cell, and the path
    """
    def findPath(self, grid, i=0, j=0, path=[[0, 0]]):
        
        """
        Basically, this method starts from cell[0][0], and start comparing the value of adjacents right and buttom cells, and choose the smalles one
        after it finds the smalles value in adjacents cells, it appends the values of this cell to the path array
        and recall itself with the value of the new cell also the path to keep track the cells selected and return at the end
        """
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
    
    # This method takes as input the gird and also the path to return the total cost
    def computePath(self, grid, path):
        
        # copy the grid to newGrid and initialize the cost to 0
        newGrid = np.copy(grid)
        cost = 0
        
        # for each cell in the path, it will assign -10 to it in the newGrid, and it will add the value to the cost from the original one
        for cell in path:
            i = cell[0]
            j = cell[1]
            newGrid[i][j] = -10
            cost += grid[i][j]
            
        # finally, it will print the cost and visualeze the grid with the path
        # the value -10 in the cells path helps to visualize the path.
        print(f"The total cost is: {cost}")
        self.visualizeTheGrid(newGrid)
    

    def dijkstra(self, grid):
        height = len(grid)
        width = len(grid[0])
        visited = []
        path = []
        nodes = np.empty((height, width), dtype=object)
        
        for i in range(len(grid)):
                for j in range(len(grid[0])):
                    node = Node(i, j, grid[i][j])
                    nodes[i][j] = node
    
        nodes[0][0].distance = 0
        pq = PQ()
        node_0 = nodes[0][0]
        pq.put(node_0)
        
        while not pq.isEmpty():
            
            current_node = pq.get()
            visited.append(current_node)
            current_i = current_node.i
            current_j = current_node.j
            

            for neighbor in self.get_adjacent([current_i, current_j]):
                i = neighbor[0]
                j = neighbor[1]
                distance = neighbor[2]
                if neighbor not in visited:
                    newNode = nodes[i][j]
                    old_cost = newNode.distance
                    new_cost = current_node.distance + distance
                    if new_cost < old_cost:
                        pq.put(newNode)
                        nodes[i][j].distance = new_cost
                        nodes[i][j].pre = [current_i, current_j]
            
            
        i = len(grid)-1
        j = len(grid[0])-1
        complate = False
        while not complate:
            path.append([i, j])
            if nodes[i][j].pre is not None:
                k = i
                x = j
                i = nodes[k][x].pre[0]
                j = nodes[k][x].pre[1]
            else: complate = True
            
        return list(reversed(path))



class AbsoluteValueMode(GridGame):

    # findPath method works same way as in mode one, but instead of comparing the values, it compares the absolute differences
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
    
    #computePath method works same way as in mode one, but instead to just adding the value from cell's path, it adds the absolute differences
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
        height = len(grid)
        width = len(grid[0])
        visited = []
        path = []
        nodes = np.empty((height, width), dtype=object)
        
        for i in range(len(grid)):
                for j in range(len(grid[0])):
                    node = Node(i, j, grid[i][j])
                    nodes[i][j] = node
    
        nodes[0][0].distance = 0
        pq = PQ()
        node_0 = nodes[0][0]
        pq.put(node_0)
        
        while not pq.isEmpty():
            
            current_node = pq.get()
            visited.append(current_node)
            current_i = current_node.i
            current_j = current_node.j
            

            for neighbor in self.get_adjacent([current_i, current_j]):
                i = neighbor[0]
                j = neighbor[1]
                distance = neighbor[2]
                if neighbor not in visited:
                    newNode = nodes[i][j]
                    old_cost = newNode.distance
                    new_cost = abs( current_node.distance - distance)
                    if new_cost < old_cost:
                        pq.put(newNode)
                        nodes[i][j].distance = new_cost
                        nodes[i][j].pre = [current_i, current_j]
            
            
        i = len(grid)-1
        j = len(grid[0])-1
        complate = False
        while not complate:
            path.append([i, j])
            if nodes[i][j].pre is not None:
                k = i
                x = j
                i = nodes[k][x].pre[0]
                j = nodes[k][x].pre[1]
            else: complate = True
            
        return list(reversed(path))



mode_A = numberInCellMode(5, 5)
grid_A = mode_A.buildTheGame()
mode_A.visualizeTheGrid(grid_A)
path_A = mode_A.findPath(grid_A)
mode_A.computePath(grid_A, path_A)


path = mode_A.dijkstra(grid_A)
mode_A.computePath(grid_A, path)
print(path)

