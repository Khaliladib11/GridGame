# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 12:27:31 2022

@author: Khaliladib
"""

#%% Import Libraries

import time
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from numpy.random import default_rng
from node import Node
from priorityQueue import PQ

# Classes

# The main class, which contains the necessary methods and properties, it is an abstract class for the game's two modes.
class GridGame(ABC):
    
    # constructor method to iniialize the game with the height and width an the max number a cell can take, finally create an empty array
    def __init__(self, height=5, width=5, n=9, grid=None):
        
        self.__height = height
        self.__width = width
        self.__n = n
        if grid is not None:
             self.__grid = grid
        else:
            self.__grid = []
    
    
    @property 
    def height(self):
        return self.__height
    
    @height.setter
    def height(self, height):
        self.__height = height
        
    @property 
    def width(self):
        return self.__width
    
    @width.setter
    def width(self, width):
        self.__width = width   
    
    @property 
    def n(self):
        return self.__n
    
    @n.setter
    def n(self, n):
        self.__n = n  
    
    @property 
    def grid(self):
        return self.__grid
    
    @grid.setter
    def grid(self, grid):
        self.__grid = grid  
        
    
    # This method is responsible for build the game, essentially filling the grid with numbers
    def buildTheGame(self, distribution="normal"):
        # fix the seeds
        np.random.seed(702)
        #check the distribution and act according to it
        if distribution == "lognormal":
            self.grid = np.random.lognormal(self.n/2, self.n/4, (self.height, self.width))
            
        elif distribution == 'uniform':
            self.grid = np.random.uniform(0, self.n, (self.height, self.width))
            
        elif distribution == 'gamma':
            self.grid = np.random.gamma(self.n/2, self.n/4, (self.height, self.width))
        
        elif distribution == "poisson":
            self.grid = np.random.poisson(self.n/2, (self.height, self.width))
            
        else:
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
    
    
    
    def mean(self, path):
        weight_path = []
        
        for p in range(len(path)):
            i = path[p][0]
            j = path[p][1]
            weight_path.append(self.grid[i][j])
        
        return round(np.mean(weight_path), 2)
    
    def variance(self, path):
        weight_path = []
        
        for p in range(len(path)):
            i = path[p][0]
            j = path[p][1]
            weight_path.append(self.grid[i][j])
        
        return round(np.var(weight_path), 2)
    
    
            
    
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
            path_temp = path
            path = []
            return path_temp
    
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
        #print(f"The total cost is: {cost}")
        #self.visualizeTheGrid(newGrid)
        return cost, newGrid
    

    # Implementation of dijkstra's algorithm to find the shortest path
    def dijkstra(self, grid):
        
        # First initialize 2 variables for the height and width of the grid
        height = len(grid)
        width = len(grid[0])
        # initialize two empty arrays, one to keep track visited node, the other is to store the path and return it
        visited = []
        path = []
        # initialize a numpy array of object (Node)
        nodes = np.empty((height, width), dtype=object)
        
        # fill the nodes np-array with the actual node from the grid
        # Node class is imported from another python file
        for i in range(len(grid)):
                for j in range(len(grid[0])):
                    node = Node(i, j, grid[i][j])
                    nodes[i][j] = node
        
        nodes[0][0].distance = 0
        
        # initialize the priority Queue, also it is imported from another python file
        # put the first node to the priority Queue
        pq = PQ()
        node_0 = nodes[0][0]
        pq.put(node_0)
        
        # iterate over the pq until its empty
        while not pq.isEmpty():
            
            # get the top of pq
            current_node = pq.get()
            visited.append(current_node)
            current_i = current_node.i
            current_j = current_node.j
            
            # get all neighbors for the current cell using get_adjacent method 
            for neighbor in self.get_adjacent([current_i, current_j]):
                i = neighbor[0]
                j = neighbor[1]
                distance = neighbor[2]
                
                # if the cell not visited, check the old cost and the new one
                # compare them
                if neighbor not in visited:
                    newNode = nodes[i][j]
                    old_cost = newNode.distance
                    # the new cost here will be the sum of the of the distance to the current node in the PQ plus the weight of the adjacent cell
                    new_cost = current_node.distance + distance
                    
                    # if the new cost smaller then old cost, update the distance of the node
                    # put the node in the PQ, and assign the pre attribute to the node behind
                    if new_cost < old_cost:
                        pq.put(newNode)
                        nodes[i][j].distance = new_cost
                        nodes[i][j].pre = [current_i, current_j]
            
        
        # after we iterate over all the grid, now we need to append the path from destination to source in the path array
        i = len(grid)-1
        j = len(grid[0])-1
        
        # basically, here we appedn the last cell(destination cell), then start looking for the previous cell
        # in this way, we can track the path from destination to source
        complate = False
        while not complate:
            path.append([i, j])
            if nodes[i][j].pre is not None:
                k = i
                x = j
                i = nodes[k][x].pre[0]
                j = nodes[k][x].pre[1]
            else: complate = True
            
        # finally return the reversed path, because the path will be from destination to source
        # we want it to be from source to destination
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
            path_temp = path
            path = []
            return path_temp
    
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
            
        #print(f"The total cost is: {cost}")
        #self.visualizeTheGrid(newGrid)
        return cost, newGrid
             
    
    
    # same implementation like before, but here new cost will be the sum of the destination for the cell 
    # plus the absolute value between the current cell's weight and adjacent cell  
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
                    new_cost = current_node.distance + abs( current_node.weight - distance)
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


#%% Analysis

# method for evaluate how the cost rises as the grid size increases
# ofcourse by running the same size multiple time and take the mean value
def analyze_the_size(mode="A", distribution='normal'):
    sizes = [5, 10, 25, 50]
    costs = []
    #variances = []
    #means = []
    if mode == 'A':
        for size in sizes:
            mean_cost = []
            for i in range(1, 10):
                gridGame = numberInCellMode(size, size)
                grid = gridGame.buildTheGame(distribution=distribution)
                path = gridGame.dijkstra(grid)
                cost, newGrid = gridGame.computePath(grid, path)
                #variance = gridGame.variance(path)
                #mean = gridGame.mean(path)
                mean_cost.append(cost)
                
            costs.append(np.mean(mean_cost))
        #print(costs)
                #means.append(mean)
                #variances.append(variance)
        plt.figure()
        plt.plot(sizes, costs)
        plt.xlabel("Size")
        plt.ylabel("Cost")   
        plt.title(f"Mode: {mode}, distribution: {distribution}")
        plt.show()
    else:
        for size in sizes:
            mean_cost = []
            for i in range(1, 10):

                gridGame = AbsoluteValueMode(size, size)
                grid = gridGame.buildTheGame(distribution=distribution)
                path = gridGame.dijkstra(grid)
                cost, newGrid = gridGame.computePath(grid, path)
                #variance = gridGame.variance(path)
                #mean = gridGame.mean(path)
                mean_cost.append(cost)
                
            costs.append(np.mean(mean_cost))
                #means.append(mean)
                #variances.append(variance)
        plt.figure()
        plt.plot(sizes, costs)
        plt.xlabel("Size")
        plt.ylabel("Cost")   
        plt.title(f"Mode: {mode}, distribution: {distribution}")
        plt.show()



# method for drawing a histogram to highlight the cost differences in various types of distribution
def distribution_hist(mode="A", size=20, distribution=['normal', 'uniform', 'gamma', 'poisson']):
    costs = []
    if mode == 'A':
        for dist in distribution:
            dist_cost = []
            for i in range(10):
                gridGame = numberInCellMode(size, size)
                grid = gridGame.buildTheGame(distribution=dist)
                path = gridGame.dijkstra(grid)
                cost, newGrid = gridGame.computePath(grid, path)
                dist_cost.append(cost)
            costs.append(np.mean(dist_cost))
    else:
        for dist in distribution:
            dist_cost = []
            for i in range(10):
                gridGame = AbsoluteValueMode(size, size)
                grid = gridGame.buildTheGame(distribution=dist)
                path = gridGame.dijkstra(grid)
                cost, newGrid = gridGame.computePath(grid, path)
                dist_cost.append(cost)
            costs.append(np.mean(dist_cost))
            
    plt.figure()
    plt.bar(distribution, costs)  
    plt.xlabel("Distribution")
    plt.ylabel("Costs")
    plt.title(f"Costs of diffrent types of distributions using a grid with {size}x{size} in mode {mode}")
    plt.show()
        

# method for calculating and comparing the execution time of various sizes and modes
def check_execution_time(mode="A", distribution='normal', sizes=[5, 10, 20, 30, 40, 50]):
    times = []
    if mode == "A":
        
        for size in sizes:
            time_mean = []
            for i in range(1, 10):
                gridGame = numberInCellMode(size, size)
                grid = gridGame.buildTheGame(distribution=distribution)
                start_time = time.time()
                path = gridGame.dijkstra(grid)
                finish_time = time.time()
                seconds = finish_time - start_time
                #print(seconds)
                time_mean.append(seconds)

            times.append(np.mean(time_mean))
            
        plt.figure()
        index = np.arange(len(sizes))
        plt.bar(index, times) 
        plt.xticks(index, sizes)
        plt.xlabel("Sizes")
        plt.ylabel("Execution time in seconds")
        plt.title(f"Execution time in: Mode: {mode}, distribution: {distribution}")
        plt.show()
        
    elif mode=='B':
        for size in sizes:
            time_mean = []
            for i in range(1, 10):
                gridGame = AbsoluteValueMode(size, size)
                grid = gridGame.buildTheGame(distribution=distribution)
                start_time = time.time()
                path = gridGame.dijkstra(grid)
                finish_time = time.time()
                seconds = finish_time - start_time
                #print(seconds)
                time_mean.append(seconds)

            times.append(np.mean(time_mean))
            
        plt.figure()
        index = np.arange(len(sizes))
        plt.bar(index, times) 
        plt.xticks(index, sizes)
        plt.xlabel("Sizes")
        plt.ylabel("Execution time in seconds")
        plt.title(f"Execution time in: Mode: {mode}, distribution: {distribution}")
        plt.show()
    
    else:
        times_A = []
        times_B = []
        for size in sizes:
            time_mean_A = []
            time_mean_B = []
            for i in range(1, 10):
                # Mode A
                gridGame = numberInCellMode(size, size)
                grid = gridGame.buildTheGame(distribution=distribution)
                start_time = time.time()
                path = gridGame.dijkstra(grid)
                finish_time = time.time()
                seconds = finish_time - start_time
                #print(seconds)
                time_mean_A.append(seconds)
                # Mode B
                gridGame = AbsoluteValueMode(size, size)
                grid = gridGame.buildTheGame(distribution=distribution)
                start_time = time.time()
                path = gridGame.dijkstra(grid)
                finish_time = time.time()
                seconds = finish_time - start_time
                #print(seconds)
                time_mean_B.append(seconds)

            times_A.append(np.mean(time_mean_A))
            times_B.append(np.mean(time_mean_B))
    
        index = np.arange(len(sizes))
        bar_width = 0.2
        plt.figure()
        plt.bar(index+bar_width, times_A, 0.4, label ='Mode A')  
        plt.bar(index-bar_width, times_B, 0.4, label='Mode B')
        plt.xlabel("Sizes")
        plt.ylabel("Execution time in seconds")
        plt.xticks(index, sizes)
        #plt.set_xticklabels(sizes)
        plt.title(f"compare Execution time in: Mode: A and B, distribution: {distribution}")
        plt.show()

