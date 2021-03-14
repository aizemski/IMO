import numpy as np
import csv
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

    def best_new(self,cycle,first,second,first_neighbours,second_neighbours,cluster):
        new = 0
        while new in cycle or new not in cluster:
            new+=1
        new_dst = first_neighbours[-1][0]+second_neighbours[-1][0]
        for i in range(len(first_neighbours)):
            if first_neighbours[i][1] not in cycle and first_neighbours[i][1] in cluster:
                for j in range(len(first_neighbours)):
                    if second_neighbours[j][1] not in cycle and second_neighbours[j][1] in cluster and first_neighbours[i][1] + second_neighbours[j][1]<new_dst:
                        if first_neighbours[i][1] == second_neighbours[j][1]:
                            new = first_neighbours[i][1]
                            new_dst = first_neighbours[i][1] + second_neighbours[j][1]
                        else:
                            break
                    
        return [new, new_dst]  
    def lowest_cost(self,nearest,cycle_dst):
        cost = nearest[0][1] - cycle_dst[0]
        index=0
        for i in range(len(nearest)):
            if cost > nearest[i][1] - cycle_dst[i]:
                index = i
                cost = nearest[i][1] - cycle_dst[i]
        return index

    def count_new_dist(self,cycle):
        new_dst = []
        for i in range(-1,len(cycle)-1):

            new_dst.append(self.dst_matrix[ cycle[i] ][0][ cycle[i+1] ][0])
        return new_dst

    def cycle_expansion_execute(self,cycle,cluster):
        while len(cycle) < len(cluster):
            nearest = []
            cycle_dst = self.count_new_dist(cycle)
            for i in range(-1,len(cycle)-1):
                nearest.append(self.best_new(cycle,cycle[i],cycle[i+1],self.dst_matrix_sorted[cycle[i]],self.dst_matrix_sorted[cycle[i+1]],cluster))

            index = self.lowest_cost(nearest,cycle_dst)
            cycle.insert(index,nearest[index][0])
            
      
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
    
    def find_second(self,candidats,cycle,cluster,cycle_dst,nearest):
        print('xd')
        candidat_regret = []
        for i in range(len(candidats)):
            tmp_regret = []
            for j in range(-1,len(cycle)-1):
                old_edge = self.dst_matrix[cycle[j]][0][cycle[j+1]][0]
                new_edge_1 =self.dst_matrix[cycle[j]][0][candidats[i]][0]
                new_edge_2 = self.dst_matrix[candidats[i]][0][cycle[j+1]][0]
                tmp_regret.append(new_edge_1+new_edge_2-old_edge)
            tmp_regret.sort()
            candidat_regret.append([tmp_regret[1]-tmp_regret[0],candidats[i]])
        candidat_regret.sort(key=lambda x : x[0])

        
        return candidat_regret[-1][1]
        


    def regret_2(self,nearest,cycle,cluster,cycle_dst):
        candidats = []
        for i in range(len(nearest)):
            nearest[i][1] = nearest[i][1] - cycle_dst[i]
        for i in nearest:
            if i[0] not in candidats:
                candidats.append(i[0])
        # print(candidats)
        if len(candidats)>1:
            best = self.find_second(candidats,cycle,cluster,cycle_dst,nearest)
        else:
            best = candidats[0]
        mini_nearest = float('inf')
        index = 0
        for i in range(nearest):
            if nearest[i][0]== best and nearest[i][1]<mini_nearest:
                mini_nearest=nearest[i][1]
                index = i
        return index

    def cycle_expansion_regret_2_execute(self,cycle,cluster):

        while len(cycle) < len(cluster):
            nearest = []
            cycle_dst = self.count_new_dist(cycle)
            for i in range(-1,len(cycle)-1):
                nearest.append(self.best_new(cycle,cycle[i],cycle[i+1],self.dst_matrix_sorted[cycle[i]],self.dst_matrix_sorted[cycle[i+1]],cluster))
            index = self.regret_2(nearest,cycle,cluster,cycle_dst)
            cycle.insert(index,nearest[index][0])
            
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
        self.save_fig([first_cycle, second_cycle])
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
            # plt.savefig(str(len(indexes))+str(indexes))

    def save_fig(self, indexes,filename='a'):
        nodess = np.array(self.nodes)
        plt.scatter(nodess[:,0],nodess[:,1])
        self.draw_lines(indexes[0])
        self.draw_lines(indexes[1])
        plt.savefig(filename+'.pdf')
        plt.close()
        # pass
    
    def save_statistic(self,data):
        with open("result.csv","w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(data)

solver = TSP(KROA100_FILENAME)
dst_array_cycle = []
min_len = float('inf')
for i in range(1):
    indexes = solver.cycle_expansion_regret_2()
    cycle_length=sum(solver.count_new_dist(indexes[0]))+sum(solver.count_new_dist(indexes[1]))
    if cycle_length < min_len:
        solver.save_fig(indexes,'cycle_expansion')
        min_len = cycle_length
    dst_array_cycle.append(cycle_length)

# print(sum(dst_array_cycle)/len(dst_array_cycle),min(dst_array_cycle),max(dst_array_cycle))
dst_array_nn = []
min_len =float('inf')
for i in range(0):
    indexes = solver.nearest_neighbour()
    cycle_length=sum(solver.count_new_dist(indexes[0]))+sum(solver.count_new_dist(indexes[1]))
    if cycle_length < min_len:
        solver.save_fig(indexes,'nearest_neighbour')
        min_len = cycle_length
    dst_array_nn.append(cycle_length)

# print(sum(dst_array_nn)/len(dst_array_nn),min(dst_array_nn),max(dst_array_nn))
# solver.save_statistic([dst_array_nn,dst_array_cycle])
# print(solver.group_nodes(1,2))
