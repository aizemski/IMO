import numpy as np
import random
import matplotlib.pyplot as plt

KROA100_FILENAME ="kroA100.tsp"
KROB100_FILENAME ="kroB100.tsp"
class TSP:
    def __init__(self,filename):
        self.nodes = self.readTSP(filename)
        self.dst_matrix_sorted, self.dst_matrix = self.create_dst_matrix()

    def readTSP(self,filename):
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

    def count_dist(self,v, u):
        return np.sqrt(((v[0] - u[0]) ** 2)+((v[1] - u[1]) ** 2))

    def create_dst_matrix(self):
        dst_matrix_sorted = []
        dst_matrix = []
        for i in range(len(self.nodes)):
            dst_matrix_sorted.append([])
            dst_matrix.append([])
            for j in range(len(self.nodes)):
                dst_matrix_sorted[i].append([self.count_dist(self.nodes[i],self.nodes[j]),j])
            #nearest 
            dst_matrix[i].append(dst_matrix_sorted[i].copy())
            dst_matrix_sorted[i].sort()
        return dst_matrix_sorted ,dst_matrix

    def group_nodes(self,first,second):
        dst = []
        for node in range(len(self.dst_matrix_sorted[0])):
            if node != first and node != second:
                dst1 = self.dst_matrix[node][0][first][0]
                dst2 = self.dst_matrix[node][0][second][0]
                dst.append((node,dst1/dst2))
        dst.sort(key=lambda x: x[1])

        return [first]+[p[0] for p in dst[:round(len(dst)/2)]], [second]+[p[0] for p in dst[round(len(dst)/2):]]
        
    def min_index_value(self,indexes, node_neighbors,cluster):
        index = 1 
        while node_neighbors[index][1] in indexes or node_neighbors[index][1] not in cluster:
            index +=1
        return node_neighbors[index][1] , node_neighbors[index][0]

    def nearest_neighbour_execute(self,cycle,cluster):
        while len(cycle) < len(cluster):
            # check on which end is closer node
            index_list =[]
            index_value_list=[]
            for i in cycle:
                index, index_value = self.min_index_value(cycle,self.dst_matrix_sorted[i],cluster)
                index_list.append(index)
                index_value_list.append(index_value)
            min_val = min(index_value_list)
            for i in range(len(index_value_list)):
                if index_value_list[i] == min_val:
                    cycle.insert(i+1,index_list[i])  
                    break
        return cycle

    def nearest_neighbour(self):
        # start from first node
        length =  len(self.dst_matrix_sorted[0])
        size_of_cycle = length//2
        first_cycle = [random.randint(0, length-1)]
        # farest node to first_cycle first node
        second_cycle = [self.dst_matrix_sorted[first_cycle[0]][-1][1]] 
        #divide nodes in two clusters
        cluster1, cluster2 = self.group_nodes(first_cycle[0],second_cycle[0])
        
        first_cycle = self.nearest_neighbour_execute(first_cycle,cluster1)
        second_cycle = self.nearest_neighbour_execute(second_cycle,cluster2)     
        # self.save_fig([first_cycle,second_cycle])
        return [first_cycle , second_cycle]


    def lowest_cost(self,cycle,cluster):
        candidat_lowest = []
        for i in range(len(cluster)):
            if cluster[i] not in cycle:
                tmp_lowest = []
                for j in range(-1,len(cycle)-1):
                    old_edge = self.dst_matrix[cycle[j]][0][cycle[j+1]][0]
                    new_edge_1 =self.dst_matrix[cycle[j]][0][cluster[i]][0]
                    new_edge_2 = self.dst_matrix[cluster[i]][0][cycle[j+1]][0]
                    tmp_lowest.append([new_edge_1+new_edge_2-old_edge,j+1])
                tmp_lowest.sort(key=lambda x : x[0])
                candidat_lowest.append([tmp_lowest[0][0],tmp_lowest[0][1],cluster[i]])
                
        candidat_lowest.sort(key=lambda x : x[0])
        # print(candidat_lowest)
        return candidat_lowest[0][1], candidat_lowest[0][2]

    def count_new_dist(self,cycle):
        new_dst = []
        for i in range(-1,len(cycle)-1):

            new_dst.append(self.dst_matrix[ cycle[i] ][0][ cycle[i+1] ][0])
        return new_dst

    def cycle_expansion_execute(self,cycle,cluster):
        while len(cycle) < len(cluster):
            

            index, point = self.lowest_cost(cycle,cluster)
            cycle.insert(index,point)
            
      
        return cycle

    def cycle_expansion(self):
        length =  len(self.dst_matrix_sorted[0])
        size_of_cycle = length//2
        first_cycle = [random.randint(0, length-1)]
        second_cycle = [self.dst_matrix_sorted[first_cycle[0]][-1][1]]
        cluster1, cluster2 = self.group_nodes(first_cycle[0],second_cycle[0])
        # first node is the nearest one
        first_index, _ = self.min_index_value(first_cycle,self.dst_matrix_sorted[first_cycle[0]],cluster1)
        second_index, _ = self.min_index_value(second_cycle,self.dst_matrix_sorted[second_cycle[0]],cluster2)

        first_cycle.append(first_index)
        second_cycle.append(second_index)
        # divide nodes in two clusters
        
        
        first_cycle = self.cycle_expansion_execute(first_cycle,cluster1)
        second_cycle = self.cycle_expansion_execute(second_cycle,cluster2)
        # self.save_fig([first_cycle, second_cycle])
        return [first_cycle, second_cycle]
    
    def regret_2(self,cycle,cluster):
        candidat_regret = []
        for i in range(len(cluster)):
            if cluster[i] not in cycle:
                tmp_regret = []
                for j in range(-1,len(cycle)-1):
                    old_edge = self.dst_matrix[cycle[j]][0][cycle[j+1]][0]
                    new_edge_1 =self.dst_matrix[cycle[j]][0][cluster[i]][0]
                    new_edge_2 = self.dst_matrix[cluster[i]][0][cycle[j+1]][0]
                    tmp_regret.append([new_edge_1+new_edge_2-old_edge,j+1])
                tmp_regret.sort(key=lambda x : x[0])
                candidat_regret.append([tmp_regret[1][0]-tmp_regret[0][0],tmp_regret[0][1],cluster[i]])
        candidat_regret.sort(key=lambda x : x[0])
        return candidat_regret[-1][1], candidat_regret[-1][2]
        
    def cycle_expansion_regret_2_execute(self,cycle,cluster):

        while len(cycle) < len(cluster):
          
            index, point = self.regret_2(cycle,cluster)
            cycle.insert(index,point)
        
        return cycle

    def cycle_expansion_regret_2(self):
        length =  len(self.dst_matrix_sorted[0])
        size_of_cycle = length//2
        first_cycle = [random.randint(0, length-1)]
        second_cycle = [self.dst_matrix_sorted[first_cycle[0]][-1][1]]
        cluster1, cluster2 = self.group_nodes(first_cycle[0],second_cycle[0])
        # first node is the nearest one
        first_index, _ = self.min_index_value(first_cycle,self.dst_matrix_sorted[first_cycle[0]],cluster1)
        second_index, _ = self.min_index_value(second_cycle,self.dst_matrix_sorted[second_cycle[0]],cluster2)

        first_cycle.append(first_index)
        second_cycle.append(second_index)
        # divide nodes in two clusters
        
        first_cycle = self.cycle_expansion_regret_2_execute(first_cycle,cluster1)
        
        second_cycle = self.cycle_expansion_regret_2_execute(second_cycle,cluster2)
        return [first_cycle, second_cycle]

    def draw_lines(self,indexes):
        if indexes:
            x_cords =[]
            y_cords=[]
            for i in indexes:
                x_cords.append(self.nodes[i][0])
                y_cords.append(self.nodes[i][1])

            x_cords.append(x_cords[0])
            y_cords.append(y_cords[0])
            plt.plot(x_cords,y_cords) 

    def save_fig(self, indexes,filename='a'):
        nodess = np.array(self.nodes)
        plt.scatter(nodess[:,0],nodess[:,1])
        self.draw_lines(indexes[0])
        self.draw_lines(indexes[1])
        plt.savefig(filename+'.jpg')
        plt.close()
    


solver = TSP(KROA100_FILENAME)
# solver = TSP(KROB100_FILENAME)
iterations = 100
dst_array_cycle = []
min_len = float('inf')
for i in range(iterations):
    indexes = solver.cycle_expansion()
    cycle_length=sum(solver.count_new_dist(indexes[0]))+sum(solver.count_new_dist(indexes[1]))
    if cycle_length < min_len:
        solver.save_fig(indexes,'cycle_expansion')
        min_len = cycle_length
    dst_array_cycle.append(cycle_length)

print('cycle expansion',sum(dst_array_cycle)/len(dst_array_cycle),min(dst_array_cycle),max(dst_array_cycle))
dst_array_nn = []
min_len =float('inf')
for i in range(iterations):
    indexes = solver.nearest_neighbour()
    cycle_length=sum(solver.count_new_dist(indexes[0]))+sum(solver.count_new_dist(indexes[1]))
    if cycle_length < min_len:
        solver.save_fig(indexes,'nearest_neighbour')
        min_len = cycle_length
    dst_array_nn.append(cycle_length)

print('nearest neighbour',sum(dst_array_nn)/len(dst_array_nn),min(dst_array_nn),max(dst_array_nn))
dst_array_cycle_r2 = []
min_len = float('inf')
for i in range(iterations):
    indexes = solver.cycle_expansion_regret_2()
    cycle_length=sum(solver.count_new_dist(indexes[0]))+sum(solver.count_new_dist(indexes[1]))
    if cycle_length < min_len:
        solver.save_fig(indexes,'cycle_expansion_regret_2')
        min_len = cycle_length
    dst_array_cycle_r2.append(cycle_length)

print('cycle expansion regret 2',sum(dst_array_cycle_r2)/len(dst_array_cycle_r2),min(dst_array_cycle_r2),max(dst_array_cycle_r2))

