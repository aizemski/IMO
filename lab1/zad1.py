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
nodes = readTSP("kroA100.tsp")
def count_dist(v, u):

    return np.sqrt(((v[0] - u[0]) ** 2)+((v[1] - u[1]) ** 2))

def create_dst_matrix(nodes):
    dst_matrix_sorted = []
    dst_matrix = []
    for i in range(len(nodes)):
        dst_matrix_sorted.append([])
        dst_matrix.append([])
        for j in range(len(nodes)):
            dst_matrix_sorted[i].append([count_dist(nodes[i],nodes[j]),j])
        #nearest 
        dst_matrix[i].append(dst_matrix_sorted[i])
        dst_matrix_sorted[i].sort()
    return dst_matrix_sorted ,dst_matrix

def min_index_value(indexes , node_neighbors):
    index = 0
    while node_neighbors[index][1] in indexes:
        index +=1
    return node_neighbors[index][1] , node_neighbors[index][0]

def nearest_neighbour_execute(cycle,size_of_cycle,dst_matrix,second_cycle):
    while len(cycle) < size_of_cycle:
    
        # check on which end is closer node
        index_list =[]
        index_value_list=[]
        for i in cycle:
            index, index_value = min_index_value(cycle+second_cycle,dst_matrix[i])
            index_list.append(index)
            index_value_list.append(index_value)
        min_val = min(index_value_list)
        
        for i in range(len(index_value_list)):
            if index_value_list[i] == min_val:
                cycle.insert(i+1,index_list[i])  
                break
    return cycle

def nearest_neighbour(dst_matrix):
    # start from first node
    length =  len(dst_matrix[0])
    size_of_cycle = length//2
    first_cycle = [0]#[random.randint(0, length-1)]
    first_cycle = nearest_neighbour_execute(first_cycle,size_of_cycle,dst_matrix,[])
    second_cycle = [0]

    # looking for first node outside of first cycle
    while second_cycle[0] in first_cycle:
        second_cycle[0]+=1
    size_of_second_cycle = length - size_of_cycle
    second_cycle = nearest_neighbour_execute(second_cycle,size_of_second_cycle,dst_matrix,first_cycle)

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
    # input()

    cost = nearest[0][1] - cycle_dst[0]

    index=0
    for i in range(len(nearest)):
        if cost > nearest[i][1] - cycle_dst[i]:
            index = i
            cost = nearest[i][1] - cycle_dst[i]
    return index

def cycle_expansion_execute(cycle,cycle_dst,size_of_cycle,dst_matrix,second_cycle):
    while len(cycle) < size_of_cycle:
        nearest = []
        for i in range(len(cycle)-1):
            nearest.append(best_new(second_cycle+cycle,cycle[i],cycle[i+1],dst_matrix[cycle[i]],dst_matrix[cycle[i+1]]))
        #find lowest cost
        index = lowest_cost(nearest,cycle_dst)

        cycle.insert(index+1,nearest[index][0])
        cycle_dst.insert(index+1,nearest[index][1])
    return cycle
def cycle_expansion(dst_matrix):
    length =  len(dst_matrix[0])
    size_of_cycle = length//2
    first_cycle = [0]#[random.randint(0, length-1)]
    first_cycle_dst =[]
    # first node is the nearest one
    first_index, first_index_value = min_index_value(first_cycle,dst_matrix[first_cycle[0]])
    first_cycle.append(first_index)
    first_cycle_dst.append(first_index_value)
    first_cycle = cycle_expansion_execute(first_cycle,first_cycle_dst,size_of_cycle,dst_matrix,[])

    second_cycle = [0]

    # looking for first node outside of first cycle
    while second_cycle[0] in first_cycle:
        second_cycle[0]+=1
    size_of_second_cycle = length - size_of_cycle
    second_cycle_dst =[]
    # second node is the nearest one
    second_index, second_index_value = min_index_value(second_cycle,dst_matrix[second_cycle[0]])
    second_cycle.append(second_index)
    second_cycle_dst.append(second_index_value)
    second_cycle = cycle_expansion_execute(second_cycle,second_cycle_dst,size_of_second_cycle,dst_matrix,first_cycle)
    return [first_cycle, second_cycle]

def regret_2():
    pass

def cycle_expansion_regret_2(dst_matrix):
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

dst_matrix_sorted, dst_matrix = create_dst_matrix(nodes)

for i in range(10):
    # indexes = nearest_neighbour(dst_matrix_sorted)
    indexes = cycle_expansion(dst_matrix_sorted)
    display(nodes, indexes)
