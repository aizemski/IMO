import numpy as np
import random
import matplotlib.pyplot as plt
import time

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

    def rest_nodes(self, cycle):
        nodes =[]
        for i in range(len(self.nodes)):
            if i not in cycle:
                nodes.append(i)
        return nodes
    
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
        return candidat_lowest[0][1], candidat_lowest[0][2]

    def count_new_dist(self,cycle):
        new_dst = []
        for i in range(-1,len(cycle)-1):
            new_dst.append(self.dst_matrix[ cycle[i] ][0][ cycle[i+1] ][0])
        return new_dst

    def cycle_expansion_execute(self,cycle,cluster,size):
        while len(cycle) < size:
            index, point = self.lowest_cost(cycle,cluster)
            cycle.insert(index,point)

        return cycle

    def cycle_expansion(self):
        length =  len(self.dst_matrix_sorted[0])
        size = length//2
        first_cycle = [random.randint(0, length-1)]
        second_cycle = [self.dst_matrix_sorted[first_cycle[0]][-1][1]]
        # first node is the nearest one
        first_index, _ = self.min_index_value(first_cycle,self.dst_matrix_sorted[first_cycle[0]],self.rest_nodes(first_cycle))
        second_index, _ = self.min_index_value(second_cycle,self.dst_matrix_sorted[second_cycle[0]],self.rest_nodes(second_cycle))

        first_cycle.append(first_index)
        second_cycle.append(second_index)

        first_cycle = self.cycle_expansion_execute(first_cycle,self.rest_nodes(second_cycle),size)
        second_cycle = self.cycle_expansion_execute(second_cycle,self.rest_nodes(first_cycle),length-size)
        
        return [first_cycle, second_cycle]
    
    def get_random_sol(self):
        length = len(self.dst_matrix_sorted[0])
        first_cycle = random.sample(range(0, length), length//2)
        second_cycle = [x for x in range(0,length) if x not in first_cycle]
        return [first_cycle,second_cycle]

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

class LocalSearch():
    def __init__(self,cycles,nodes,dst_matrix,dst_matrix_sorted,solver):
        self.first , self.second = cycles
        self.nodes = nodes
        self.dst_matrix_sorted = dst_matrix_sorted
        self.dst_matrix = dst_matrix
        self.solver = solver
    def count_dst(self, x ,s1 ,s2):
        
        return self.dst_matrix[x][0][s1][0] + self.dst_matrix[x][0][s2][0]

    def count_change(self,x,xs1,xs2,y,ys1,ys2):

       
        old_x = self.count_dst(x,xs1,xs2)
        old_y = self.count_dst(y,ys1,ys2)
        if x == ys1:
            ys1 = y
            xs2 = x
        if y == xs1:
            xs1 = x
            ys2 = y
        
        new_x = self.count_dst(x,ys1,ys2)
        new_y = self.count_dst(y,xs1,xs2)
        
        
        return (new_x+new_y) - (old_x+old_y)  

    def greed(self):
        def inner(i,j,which):
            
            if which:
                length = len(self.first)
                
                if self.count_change(self.first[i],self.first[i-1],self.first[(i+1)%length],self.first[j],self.first[j-1],self.first[(j+1)%length]) <0:
                    
                    return True
                return False
            else:
                length = len(self.second)
                if  self.count_change(self.second[i],self.second[i-1],self.second[(i+1)%length],self.second[j],self.second[j-1],self.second[(j+1)%length]) <0:
                    return True
                return False       

            
        def outer(i,j):
            length1 = len(self.first)
            length2 = len(self.second)
            if self.count_change(self.first[i],self.first[i-1],self.first[(i+1)%length1],self.second[j],self.second[j-1],self.second[(j+1)%length2]) <0:
                return True
            return False
                       

        #random
        
        length1 = len(self.first)
        length2 = len(self.second)
        order1 = random.sample(range(0,length1 ),length1)
        order2 = random.sample(range(0,length2 ),length2)
        flag=True
        i = 0
        while flag:
            flag = False
            while i < length2 :
                k,l = 0,0
                while k < length2 or l<length2:
                    # between cycles
                    if  l<length2 and random.randint(0, 1):
                        if outer(order1[i],order2[l]):
                            flag = True
                            self.first[order1[i]],self.second[order2[l]] = self.second[order2[l]],self.first[order1[i]]
                            
                        l+=1
                    elif k < length2:
                        # first cycle
                        if random.randint(0, 1):
                            if inner(order1[i],order1[k],1):
                                flag = True
                                self.first[order1[i]],self.first[order1[k]] = self.first[order1[k]],self.first[order1[i]]
                              
                        # second cycle
                        else: 
                            if inner(order2[i],order2[k],1):
                                flag = True
                                self.second[order2[i]],self.second[order2[k]] = self.second[order2[k]],self.second[order2[i]]
                                
                            
                        k+=1
                i+=1
    def steeper(self):
        def inner():
            best = 0
            index , index2 = 0,0 
            # index, index2
            length = len(self.first)

            for i in range(length):
                for j in range(length):
                    current = self.count_change(self.first[i],self.first[i-1],self.first[(i+1)%length],self.first[j],self.first[j-1],self.first[(j+1)%length])
                    if current <best:
                        best = current
                        index = self.first[i]
                        index2 = self.first[j]
            length = len(self.second)
            for i in  range(length):
                for j in range(length):
                    currnet =self.count_change(self.second[i],self.second[i-1],self.second[(i+1)%length],self.second[j],self.second[j-1],self.second[(j+1)%length])
                    if current<best:
                        best = current
                        index = self.second[i]
                        index2 = self.second[j]    

            return best,index,index2

        def outer():
            length1 = len(self.first)
            length2 = len(self.second)
            best = 0
            index , index2 = 0,0
 
            for i in range(length1):
                for j in range(length2):
                    current = self.count_change(self.first[i],self.first[i-1],self.first[(i+1)%length1],self.second[j],self.second[j-1],self.second[(j+1)%length2])
                    
                    if current < best:
                        best = current
                        index = self.first[i]
                        index2 = self.second[j]
            return best,index,index2

        while True:
            
            
            i_best, i_index, i_index2 = inner()
            o_best, o_index, o_index2 = outer()
            
            if i_best == 0 and o_best == 0:
                break

            if i_best < o_best:
                if i_index in self.first:
                    i = self.first.index(i_index)
                    j = self.first.index(i_index2)
                    self.first[i],self.first[j] = self.first[j],self.first[i]
                else:
                    i = self.second.index(i_index)
                    j = self.second.index(i_index2)
                    self.second[i],self.second[j] = self.second[j],self.second[i]
            else:
                i = self.first.index(o_index)
                j = self.second.index(o_index2)
                self.first[i],self.second[j] = self.second[j],self.first[i]
        
    def random(self):
        strat = time.time()
        # while time.time() < 
        

if __name__ =='__main__':
    # name = KROB100_FILENAME
    name = KROA100_FILENAME
    solver = TSP(name)

    iterations = 100
    #cycle expansion
    dst_array_cycle = []
    time_array=[]
    min_len = float('inf')
    for i in range(iterations):
        indexs = solver.cycle_expansion()
        search = LocalSearch(indexs,solver.nodes,solver.dst_matrix,solver.dst_matrix_sorted,solver)

        start = time.time()
        search.greed()
        end = time.time()
        
        cycle_length=sum(solver.count_new_dist(search.first))+sum(solver.count_new_dist(search.second))
        if cycle_length < min_len:
            solver.save_fig([search.first,search.second],'cycle_expansion_greedy_'+name)
            min_len = cycle_length

        dst_array_cycle.append(cycle_length)
        time_array.append(end - start)
    print('cycle expansion greed '+name,sum(dst_array_cycle)/len(dst_array_cycle),min(dst_array_cycle),max(dst_array_cycle)) 
    print('cycle expansion greed time '+name,sum(time_array)/len(time_array),min(time_array),max(time_array)) 
    
    dst_array_cycle = []
    time_array=[]
    min_len = float('inf')
    for i in range(iterations):
        indexs = solver.cycle_expansion()
        search = LocalSearch(indexs,solver.nodes,solver.dst_matrix,solver.dst_matrix_sorted,solver)
    
        start = time.time()
        search.steeper()
        end = time.time()
    
        cycle_length=sum(solver.count_new_dist(search.first))+sum(solver.count_new_dist(search.second))
        if cycle_length < min_len:
            solver.save_fig([search.first,search.second],'cycle_expansion_steeper'+name)
            min_len = cycle_length

        time_array.append(end - start)
        dst_array_cycle.append(cycle_length)
    print('cycle expansion steeper '+name,sum(dst_array_cycle)/len(dst_array_cycle),min(dst_array_cycle),max(dst_array_cycle)) 
    print('cycle expansion steeper time '+name,sum(time_array)/len(time_array),min(time_array),max(time_array)) 
    
    # random sol
    dst_array_cycle = []
    time_array=[]
    min_len = float('inf')
    for i in range(iterations):
        indexs = solver.get_random_sol()
        search = LocalSearch(indexs,solver.nodes,solver.dst_matrix,solver.dst_matrix_sorted,solver)
        
        start = time.time()
        search.greed()
        end = time.time()
        
        cycle_length=sum(solver.count_new_dist(search.first))+sum(solver.count_new_dist(search.second))
        if cycle_length < min_len:
            solver.save_fig([search.first,search.second],'random_greed_'+name)
            min_len = cycle_length
        
        time_array.append(end - start)
        dst_array_cycle.append(cycle_length)
    print('random greed '+name,sum(dst_array_cycle)/len(dst_array_cycle),min(dst_array_cycle),max(dst_array_cycle)) 
    print('random greed '+name,sum(time_array)/len(time_array),min(time_array),max(time_array))
    
    dst_array_cycle = []
    time_array=[]
    min_len = float('inf')
    for i in range(iterations):
        indexs = solver.get_random_sol()
        search = LocalSearch(indexs,solver.nodes,solver.dst_matrix,solver.dst_matrix_sorted,solver)
    
        start = time.time()
        search.steeper()
        end = time.time()
    
        cycle_length=sum(solver.count_new_dist(search.first))+sum(solver.count_new_dist(search.second))
        if cycle_length < min_len:
            solver.save_fig([search.first,search.second],'random_steeper_'+name)
            min_len = cycle_length
    
        time_array.append(end - start)
        dst_array_cycle.append(cycle_length)
    print('random steeper '+name,sum(dst_array_cycle)/len(dst_array_cycle),min(dst_array_cycle),max(dst_array_cycle)) 
    print('random steeper time '+name,sum(time_array)/len(time_array),min(time_array),max(time_array))
    