import numpy as np
import random
import matplotlib.pyplot as plt


def readTSP(filename):
    nodelist = []
    with open(filename,'r') as file:
        # Read header
        file.readline()# NAME
        file.readline() # TYPE
        file.readline() # COMMENT
        dimension = file.readline().strip().split()[1] # DIMENSION
        file.readline() # EDGE_WEIGHT_TYPE
        file.readline()

        # Read node list
        
        N = int(dimension)
        for i in range(0, int(dimension)):
            x,y = file.readline().strip().split()[1:]
            nodelist.append([int(x), int(y)])

        # Close input file

    return nodelist

def count_dist(v, u):

    return np.sqrt(((v[0] - u[0]) ** 2)+((v[1] - u[1]) ** 2))

def create_dst_matrix(nodes):
    dst_matrix = []
    for i in range(len(nodes)):
        dst_matrix.append([])
        for j in range(len(nodes)):
            dst_matrix[i].append(count_dist(nodes[i],nodes[j]))
        
    return dst_matrix

def min_index_value(indexes , node_neighbors):
    index = 0
    while index in indexes:
        index +=1
    for i in range(len(node_neighbors)):
        if i not in indexes:
            if node_neighbors[i] <= node_neighbors[index]:
                index=i

    return index , node_neighbors[index]


    
def nearest_neighbour(dst_matrix):
    # start from first node
    
    length =  len(dst_matrix[0])
    size_of_cycle = length//2
    first_cycle =[random.randint(0, length-1)]
  
    while len(first_cycle) < size_of_cycle:
        
        # check on which end is closer node
        first_index, first_index_value = min_index_value(first_cycle,dst_matrix[first_cycle[0]])
        last_index, last_index_value =  min_index_value(first_cycle,dst_matrix[first_cycle[-1]])
        
        # add node at beginnig
        if first_index_value > last_index_value:
            first_cycle.insert(0,first_index)
        # add node at end
        else:
            first_cycle.append(last_index)
    
    second_cycle = [0]

    # looking for first node outside of first cycle
    while second_cycle[0] in first_cycle:
        second_cycle[0]+=1
    size_of_second_cycle = length - size_of_cycle
    
    while len(second_cycle) < size_of_second_cycle:
        # check on which end is closer node
        first_index, first_index_value = min_index_value(first_cycle+second_cycle,dst_matrix[second_cycle[0]])
        last_index, last_index_value =  min_index_value(first_cycle+second_cycle,dst_matrix[second_cycle[-1]])
        
        # add node at beginnig
        if first_index_value > last_index_value:
            second_cycle.insert(0,first_index)
        # add node at end
        else:
            second_cycle.append(last_index)

    return [first_cycle , second_cycle]

def cycle_expansion(dst_matrix):
    pass

def cycle_expansion_regret(dst_matrix):
    pass
def draw_lines(nodes,indexes):
    x_cords =[]
    y_cords=[]
    for i in indexes:
        x_cords.append(nodes[i][0])
        y_cords.append(nodes[i][1])

    x_cords.append(x_cords[0])
    y_cords.append(y_cords[0])
    plt.plot(x_cords,y_cords) 

def display(nodes, indexes):
    nodess = np.array(nodes)
    plt.scatter(nodess[:,0],nodess[:,1])
    draw_lines(nodes,indexes[0])
    draw_lines(nodes,indexes[1])
    plt.show()
    # pass
nodes = readTSP("kroA100.tsp")
dst_matrix = create_dst_matrix(nodes)
indexes = nearest_neighbour(dst_matrix)
display(nodes, indexes)
