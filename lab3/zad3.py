import numpy as np
import random
import matplotlib.pyplot as plt
import time

KROA200_FILENAME ="kroA200.tsp"
KROB200_FILENAME ="kroB200.tsp"

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

    def cycle_expansion(self,first,second):
        length =  len(self.dst_matrix_sorted[0])
        size = length//2
        first_cycle = [first[0]]
        second_cycle = [second[0]]
       

        first_cycle = self.cycle_expansion_execute(first_cycle,first,size)
        second_cycle = self.cycle_expansion_execute(second_cycle,second,length-size)
        
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
        self.candidats = []
    def count_candidats(self):
        size = 20
        for i in range(len(self.nodes)):
            self.candidats.append([])
            for j in range(size):
                
                
                self.candidats[i].append(self.dst_matrix_sorted[i][1:size+1][j][1])

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
    def count_dst_edge(self,x,s1):
        return self.dst_matrix[x][0][s1][0]

    def count_change_edge(self,x,xs1,y,ys1):
        old_x = self.count_dst_edge(x,xs1)
        old_y = self.count_dst_edge(y,ys1)
        new_x = self.count_dst_edge(x,ys1)
        new_y = self.count_dst_edge(y,xs1)
        return (new_x+new_y) - (old_x+old_y)  
    def revers(self,begin,end,which):
        if which==1:
            tmp = self.first.copy()
            for i in range(end-1,begin,-1):
                
                self.first[begin+(end-i)] = tmp[i]
        else:
            
            tmp = self.second.copy()
            for i in range(end-1,begin,-1):
                self.second[begin+(end-i)] = tmp[i]

    def steepest(self):
        def inner():
            best = 0
            index , index2 = 0,0 
            length = len(self.first)
          
            for i in range(length):
                for j in range(i+5,length):
                    current = self.count_change_edge(self.first[i],self.first[(i+1)%length],self.first[j],self.first[(j-1)%length])
                    if current <best:
                        best = current
                        index = self.first[i]
                        index2 = self.first[j]
                   
            length = len(self.second)
            for i in range(length):
                for j in range(i+5,length):
                    current = self.count_change_edge(self.second[i],self.second[(i+1)%length],self.second[j],self.second[(j-1)%length])
                    if current <best:
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
         
                    self.revers(i,j,1)
               
                else:
                    i = self.second.index(i_index)
                    j = self.second.index(i_index2)
                    
                    self.revers(i,j,2)
                    
            else:
                i = self.first.index(o_index)
                j = self.second.index(o_index2)
                
                self.first[i],self.second[j] = self.second[j],self.first[i]
            
            
    def steepest_list(self):
        def inner():
            moves=[]
            length = len(self.first)
            best =0
            for i in range(length):
                for j in range(i+5,length):
                    current = self.count_change_edge(self.first[i],self.first[(i+1)%length],self.first[j],self.first[(j-1)%length])
                    if current <0:
                        if current<best:
                            best=current
                           
                        if self.first[i] < self.first[j]:
                            moves.append((current,self.first[i],self.first[j]))
                        else:
                            moves.append((current,self.first[j],self.first[i]))
            length = len(self.second)
            for i in range(length):
                for j in range(i+5,length):
                    current = self.count_change_edge(self.second[i],self.second[(i+1)%length],self.second[j],self.second[(j-1)%length])
                    if current <0:
                        if current<best:
                            best=current
                           
                        if self.second[i] < self.second[j]:
                            moves.append((current,self.second[i],self.second[j]))
                        else:
                            moves.append((current,self.second[j],self.second[i]))
            
            return moves
            

        def outer():
            length1 = len(self.first)
            length2 = len(self.second)
            moves = []
 
            for i in range(length1):
                for j in range(length2):
                    current = self.count_change(self.first[i],self.first[i-1],self.first[(i+1)%length1],self.second[j],self.second[j-1],self.second[(j+1)%length2])
                    if current <0:
                        if self.first[i] < self.second[j]:
                            moves.append((current,self.first[i],self.second[j]))
                        else:
                            moves.append((current,self.second[j],self.first[i]))
                    
            return moves
        def count_new_value(x,y):
            index_x , index_y = 0,0
            x_cycle, y_cycle = 1,1
            if x in self.first:
                x_cycle =1
                index_x = self.first.index(x)
            else:
                x_cycle =2
                index_x = self.second.index(x)
            if y in self.first:
                y_cycle =1
                index_y = self.first.index(y)
            else:
                y_cycle =2
                index_y = self.second.index(y)
            maxi = max(index_x,index_y)
            mini = min(index_x,index_y)
            if x_cycle==1 and y_cycle ==1:
                if (maxi-mini >4):
                    return self.count_change_edge(self.first[mini],self.first[(mini+1)%len(self.first)],self.first[maxi],self.first[(maxi-1)%len(self.first)])
                else:
                    return 1
            elif x_cycle==2 and y_cycle ==2:
                if (maxi-mini >4):
                    return self.count_change_edge(self.second[mini],self.second[(mini+1)%len(self.second)],self.second[maxi],self.second[(maxi-1)%len(self.second)])
                else:
                    return 1
            else:
                if x_cycle == 1:
                    return self.count_change(self.first[index_x],self.first[index_x-1],self.first[(index_x+1)%len(self.first)],self.second[index_y],self.second[index_y-1],self.second[(index_y+1)%len(self.second)])
                else:
                    return self.count_change(self.second[index_x],self.second[index_x-1],self.second[(index_x+1)%len(self.second)],self.first[index_y],self.first[index_y-1],self.first[(index_y+1)%len(self.first)])
        def check_if_is_not_in_moves(move,x,y):
            for i in range(len(move)):
                if (moves[i][1]==x and moves[i][2]==y) or (moves[i][2]==x and moves[i][1]==y):
                    return False
            return True
        def count_new_values(moves,x,y,types):
        
            if types == 0:
                a,b=0,0
                length = len (self.first)
                for i in [self.first.index(x),self.first.index(y)]:
                    for j in range(length):  
                        if abs(j-i)>4:
                            a = min(i,j)
                            b = max(i,j)
                            current = self.count_change_edge(self.first[a],self.first[(a+1)%length],self.first[b],self.first[(b-1)%length])
                            if current <0:
                                if self.first[a] < self.first[b]:
                                    moves.append((current,self.first[a],self.first[b]))
                                else:
                                    moves.append((current,self.first[b],self.first[a]))
                        current = self.count_change(self.first[i],self.first[i-1],self.first[(i+1)%length],self.second[j],self.second[j-1],self.second[(j+1)%length])
                        if current <0:
                            if self.first[i] < self.second[j]:
                                moves.append((current,self.first[i],self.second[j]))
                            else:
                                moves.append((current,self.second[j],self.first[i]))
            if types == 1:
                length = len (self.second)
                for i in [self.second.index(x),self.second.index(y)]:
                    
                    for j in range(length):
                        if abs(j-i)>4:
                            current = self.count_change_edge(self.second[i],self.second[(i+1)%length],self.second[j],self.second[(j-1)%length])
                            if current <0:
                                if self.second[i] < self.second[j]:
                                    moves.append((current,self.second[i],self.second[j]))
                                else:
                                    moves.append((current,self.second[j],self.second[i]))
                        current = self.count_change(self.first[i],self.first[i-1],self.first[(i+1)%length],self.second[j],self.second[j-1],self.second[(j+1)%length])
                        if current <0:
                            if self.first[i] < self.second[j]:
                                moves.append((current,self.first[i],self.second[j]))
                            else:
                                moves.append((current,self.second[j],self.first[i]))
            if types ==2:
               
                try:
                    i = self.first.index(x)
                    length = len (self.second)
                    for j in range(length):
                        
                        current = self.count_change(self.first[i],self.first[i-1],self.first[(i+1)%length],self.second[j],self.second[j-1],self.second[(j+1)%length])
                        if current <0:
                            if self.first[i] < self.second[j]:
                                moves.append((current,self.first[i],self.second[j]))
                            else:
                                moves.append((current,self.second[j],self.first[i]))
                        if abs(j-i)>4:
                            a = min(i,j)
                            b = max(i,j)
                            current = self.count_change_edge(self.first[a],self.first[(a+1)%length],self.first[b],self.first[(b-1)%length])
                            if current <0:
                                if self.first[a] < self.first[b]:
                                    moves.append((current,self.first[a],self.first[b]))
                                else:
                                    moves.append((current,self.first[b],self.first[a]))
                except:
                    pass
                try:
                    j = self.second.index(y)
                    length = len (self.first)
                    for i in range(length):
                        
                        current = self.count_change(self.first[i],self.first[i-1],self.first[(i+1)%length],self.second[j],self.second[j-1],self.second[(j+1)%length])
                        if current <0:
                            if self.first[i] < self.second[j]:
                                moves.append((current,self.first[i],self.second[j]))
                            else:
                                moves.append((current,self.second[j],self.first[i]))
                        if abs(j-i)>4:
                            current = self.count_change_edge(self.second[i],self.second[(i+1)%length],self.second[j],self.second[(j-1)%length])
                            if current <0:
                                if self.second[i] < self.second[j]:
                                    moves.append((current,self.second[i],self.second[j]))
                                else:
                                    moves.append((current,self.second[j],self.second[i]))
                except:
                    pass
            return moves

        def update_moves(moves,x,y,types=0):
            k = 0
            
            
            moves = count_new_values(moves,x,y,types)
            
            
            while k<len(moves):
                new_value = count_new_value(moves[k][1],moves[k][2])
                if new_value<0:
                    moves[k]=(new_value,moves[k][1],moves[k][2])
                    k+=1
                else:
                    
                    moves.pop(k)
               
            
            return moves
        


        inner_moves = inner()
        outer_moves = outer()
        moves = inner_moves+outer_moves
        moves = list(dict.fromkeys(moves))
        
        moves.sort()
         
        while len(moves):
            
            if moves[0][1] in self.first and moves[0][2] in self.first :
              
                i = self.first.index(moves[0][1])
                j = self.first.index(moves[0][2])
                x1=self.first[(i+1)%len(self.first)]
                x2=self.first[(j+1)%len(self.first)]
                
                self.revers(min(i,j),max(i,j),1)
                
                moves = update_moves(moves,moves[0][1],moves[0][2],types=0)
                moves = update_moves(moves,x1,x2,types=0)
                
            
            elif moves[0][1] in self.second and moves[0][2] in self.second:
                i = self.second.index(moves[0][1])
                j = self.second.index(moves[0][2])
                x1=self.second[(i+1)%len(self.second)]
                x2=self.second[(j+1)%len(self.second)]
               
               
                self.revers(min(i,j),max(i,j),2)
                moves = update_moves(moves,moves[0][1],moves[0][2],types=1)
                moves = update_moves(moves,x1,x2,types=1)
                
                
            else:
                i =0
                j =0
                x1 =0
                x2=0
                y1=0
                y2=0
              
                if moves[0][1]in self.first:
                    i = self.first.index(moves[0][1])
                    j = self.second.index(moves[0][2])
                    x =moves[0][2]
                    y =moves[0][1]
                    x1 = self.first[(i+1)%len(self.first)]
                    x2=self.second[(j+1)%len(self.second)]
                    y1=self.first[i-1]
                    y2= self.second[j-1]
                    
                    
                   
                else:
                    j = self.second.index(moves[0][1])
                    i = self.first.index(moves[0][2])
                    x =moves[0][1]
                    y =moves[0][2]
                    x2=self.second[(i+1)%len(self.second)]
                    x1=self.first[(j+1)%len(self.first)]
                    y1=self.second[i-1]
                    y2=self.first[j-1]
                    
                   
                self.first[i],self.second[j] = self.second[j],self.first[i]
                moves = update_moves(moves,x,y,types=2)
                moves = update_moves(moves,x1,x2,types=2)
                moves = update_moves(moves,y1,y2,types=2)
             
           
            moves = list(dict.fromkeys(moves))
            
            moves.sort()
            
    def steepest_candidats(self):
        l =0
        while True:
            l+=1
            flag = -1
            best = 0
            x_,y_ = 0,0
            
            for i in range(len(self.candidats)):
                for j in range(len(self.candidats[0])):
                    if   i in self.first:
                        x,y=0,0
                        x = self.first.index(i)
                        if self.candidats[i][j] in self.first:
                       
                            y = self.first.index(self.candidats[i][j])
                            a = min(x,y)
                            b= max(x,y)
                            if b-a>4:
                                current = self.count_change_edge(self.first[a],self.first[(a+1)%len(self.first)],self.first[b],self.first[(b-1)%len(self.first)])
                                if current < best:
                                    best = current
                                    x_,y_ = x,y
                                    flag=0
                                    
                        else:
                            
                            y = self.second.index(self.candidats[i][j])
                            current = self.count_change(self.first[x],self.first[x-1],self.first[(x+1)%len(self.first)],self.second[y],self.second[y-1],self.second[(y+1)%len(self.second)])
                            if current < best:
                                best = current
                                x_,y_ = x,y
                                flag = 1

                       
                    else:
                        x,y=0,0
                        x = self.second.index(i)
                        if self.candidats[i][j] in self.second:
                            
                            y = self.second.index(self.candidats[i][j])
                            a = min(x,y)
                            b= max(x,y)
                            if b-a>4:
                                current = self.count_change_edge(self.second[a],self.second[(a+1)%len(self.second)],self.second[b],self.second[(b-1)%len(self.second)])
                                if current < best:
                                    best = current
                                    x_,y_ = x,y
                                    
                                    flag=2
                        else:
                           
                                y = self.first.index(self.candidats[i][j])
                                current = self.count_change(self.first[y],self.first[y-1],self.first[(y+1)%len(self.first)],self.second[x],self.second[x-1],self.second[(x+1)%len(self.second)])
                                if current < best:
                                    best = current
                                    x_,y_ = x,y
                                    flag=3
                                   
           
            if best==0:
                break
            
            
        
            if flag ==0:
                self.revers(min(x_,y_),max(x_,y_),1)
            
          
            if flag == 1:
                self.first[x_],self.second[y_] = self.second[y_],self.first[x_]
            if flag == 2:
              
                self.revers(min(x_,y_),max(x_,y_),2)
               
            if flag == 3:
                self.first[y_],self.second[x_] = self.second[x_],self.first[y_]
            
           


if  __name__ =='__main__':
    name = KROB200_FILENAME
    # name = KROA200_FILENAME
    solver = TSP(name)

    iterations = 100
    cycle expansion
    dst_array_cycle = []
    time_array=[]
  
    
    dst_array_cycle = []
    time_array=[]
    min_len = float('inf')
    for i in range(iterations):
        indexs = solver.get_random_sol()
    
        start = time.time()
        indexs = solver.cycle_expansion(indexs[0],indexs[1])
        end = time.time()
    
        cycle_length=sum(solver.count_new_dist(indexs[0]))+sum(solver.count_new_dist(indexs[1]))
        if cycle_length < min_len:
            solver.save_fig([indexs[0],indexs[1]],'cycle_expansion_'+name)
            min_len = cycle_length

        time_array.append(end - start)
        dst_array_cycle.append(cycle_length)
    print('cycle expansion  '+name,sum(dst_array_cycle)/len(dst_array_cycle),min(dst_array_cycle),max(dst_array_cycle)) 
    print('cycle expansion  time '+name,sum(time_array)/len(time_array),min(time_array),max(time_array)) 

    dst_array_cycle = []
    time_array=[]
    min_len = float('inf')
    for i in range(iterations):
        indexs = solver.get_random_sol()
        search = LocalSearch(indexs,solver.nodes,solver.dst_matrix,solver.dst_matrix_sorted,solver)
    
        start = time.time()
        search.steepest()
        end = time.time()
    
        cycle_length=sum(solver.count_new_dist(search.first))+sum(solver.count_new_dist(search.second))
        if cycle_length < min_len:
            solver.save_fig([search.first,search.second],'random_steepest_'+name)
            min_len = cycle_length
    
        time_array.append(end - start)
        dst_array_cycle.append(cycle_length)
    print('random steepest '+name,sum(dst_array_cycle)/len(dst_array_cycle),min(dst_array_cycle),max(dst_array_cycle)) 
    print('random steepest time '+name,sum(time_array)/len(time_array),min(time_array),max(time_array))

    dst_array_cycle = []
    time_array=[]
    min_len = float('inf')
    for i in range(iterations):
        indexs = solver.get_random_sol()
        search = LocalSearch(indexs,solver.nodes,solver.dst_matrix,solver.dst_matrix_sorted,solver)
    
        start = time.time()
        search.steepest_list()
        end = time.time()
    
        cycle_length=sum(solver.count_new_dist(search.first))+sum(solver.count_new_dist(search.second))
        if cycle_length < min_len:
            solver.save_fig([search.first,search.second],'random_steepest_list'+name)
            min_len = cycle_length
    
        time_array.append(end - start)
        dst_array_cycle.append(cycle_length)
    print('random steepest list '+name,sum(dst_array_cycle)/len(dst_array_cycle),min(dst_array_cycle),max(dst_array_cycle)) 
    print('random steepest list time '+name,sum(time_array)/len(time_array),min(time_array),max(time_array))
    dst_array_cycle = []
    time_array=[]
    min_len = float('inf')
    for i in range(iterations):
        indexs = solver.get_random_sol()
        solver.save_fig([indexs[0],[]],'test'+name)
        search = LocalSearch(indexs,solver.nodes,solver.dst_matrix,solver.dst_matrix_sorted,solver)
        search.count_candidats()
        start = time.time()
        search.steepest_candidats()
        end = time.time()
    
        cycle_length=sum(solver.count_new_dist(search.first))+sum(solver.count_new_dist(search.second))
        if cycle_length < min_len:
            solver.save_fig([search.first,search.second],'random_steepest_candidats'+name)
            min_len = cycle_length
    
        time_array.append(end - start)
        dst_array_cycle.append(cycle_length)
    print('random steepest candidats '+name,sum(dst_array_cycle)/len(dst_array_cycle),min(dst_array_cycle),max(dst_array_cycle)) 
    print('random steepest candidats time '+name,sum(time_array)/len(time_array),min(time_array),max(time_array))
    
    

