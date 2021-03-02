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
            dst_matrix[i].append([count_dist(nodes[i],nodes[j]),j])
        #nearest 
        dst_matrix[i].sort()
    return dst_matrix

def min_index_value(indexes , node_neighbors):
    index = 0
    while node_neighbors[index][1] in indexes:
        index +=1
    return node_neighbors[index][1] , node_neighbors[index]

def nearest_neighbour(dst_matrix):
    # start from first node
    
    length =  len(dst_matrix[0])
    size_of_cycle = length//2
    first_cycle = [random.randint(0, length-1)]
  
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

def best_new(cycle,first,second,first_neighbours,second_neighbours):
    new = 0
    while new in cycle:
        new+=1
    new_dst = first_neighbours[-1][0]+second_neighbours[-1][0]
    for i in range(len(first_neighbours)):
        if first_neighbours[i][1] not in cycle:
            for j in range(len(first_neighbours)):
                if second_neighbours[j][1] not in cycle and first_neighbours[i][1] + second_neighbours[j][1]<new_dst:
                    if first_neighbours[i][1] == second_neighbours[j][1]:
                        new = first_neighbours[i][1]
                        new_dst = first_neighbours[i][1] + second_neighbours[j][1]
                    else:
                        break
                
    return [new, new_dst]  
def lowest_cost(nearest,cycle_dst):
    # print(nearest)
    # print(cycle_dst)
    # print()
    cost = nearest[0][1] - cycle_dst[0]

    index=0
    for i in range(len(nearest)):
        if cost > nearest[i][1] - cycle_dst[i]:
            index = i
            cost = nearest[i][1] - cycle_dst[i]
    return index

def cycle_expansion(dst_matrix):
    length =  len(dst_matrix[0])
    size_of_cycle = length//2
    first_cycle = [0]#[random.randint(0, length-1)]
    first_cycle_dst =[]
    # first node is the nearest one
    first_index, first_index_value = min_index_value(first_cycle,dst_matrix[first_cycle[0]])
    first_cycle.append(first_index)
    first_cycle_dst.append(first_index_value[0])

    while len(first_cycle) < size_of_cycle:
        nearest = []
        for i in range(len(first_cycle)-1):
            nearest.append(best_new(first_cycle,first_cycle[i],first_cycle[i+1],dst_matrix[first_cycle[i]],dst_matrix[first_cycle[i+1]]))
        #find lowest cost
        index = lowest_cost(nearest,first_cycle_dst)

        first_cycle.insert(index,nearest[index][0])
        first_cycle_dst.insert(index,nearest[index][1])
    second_cycle = [0]

    # looking for first node outside of first cycle
    while second_cycle[0] in first_cycle:
        second_cycle[0]+=1
    size_of_second_cycle = length - size_of_cycle
    second_cycle_dst =[]
    # second node is the nearest one
    second_index, second_index_value = min_index_value(second_cycle,dst_matrix[second_cycle[0]])
    second_cycle.append(second_index)
    second_cycle_dst.append(second_index_value[0])
    while len(second_cycle) < size_of_second_cycle:
        nearest = []
        for i in range(len(second_cycle)-1):
            nearest.append(best_new(first_cycle+second_cycle,second_cycle[i],second_cycle[i+1],dst_matrix[second_cycle[i]],dst_matrix[second_cycle[i+1]]))
        #find lowest cost
        index = lowest_cost(nearest,second_cycle_dst)

        second_cycle.insert(index,nearest[index][0])
        second_cycle_dst.insert(index,nearest[index][1])

    return [first_cycle, second_cycle]

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
# indexes = nearest_neighbour(dst_matrix)
indexes = cycle_expansion(dst_matrix)
display(nodes, indexes)
