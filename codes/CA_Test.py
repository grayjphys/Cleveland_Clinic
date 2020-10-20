def get_line(start, end):
    #Bresenham's Line Algorithm
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()

    return points

import numpy as np

def return_edge_pos(node_pos1,node_pos2,width):
    x1 = node_pos1[0][0]
    y1 = node_pos1[0][1]
    x2 = node_pos2[0][0]
    y2 = node_pos2[0][1]
    y_ = []
    if x1 != x2:
        m = (y2-y1)/(x2-x1)
        c = round(width*(1+m**2)**0.5)
        for d in np.arange(-c,c+1):
            y = get_line((x1,y1+d),(x2,y2+d))
            for point in y:
                if point in node_pos1 or point in node_pos2 or ((point[0]-x1)**2+(point[1]-y1)**2)**0.5 > ((x2-x1)**2+(y2-y1)**2)**0.5 or ((point[0]-x2)**2+(point[1]-y2)**2)**0.5 > ((x2-x1)**2+(y2-y1)**2)**0.5:
                    y =[p for p in y if p != point]
            for point in y:
                y_.append(point)
            pass
    else:
        for d in np.arange(-width,width+1):
            y = get_line((x1+d,y1),(x2+d,y2))
            for point in y:
                if point in node_pos1 or point in node_pos2 or ((point[0]-x1)**2+(point[1]-y1)**2)**0.5 > ((x2-x1)**2+(y2-y1)**2)**0.5 or ((point[0]-x2)**2+(point[1]-y2)**2)**0.5 > ((x2-x1)**2+(y2-y1)**2)**0.5:
                    y =[p for p in y if p != point]
            for point in y:
                y_.append(point)
    return y_

'''Star Graph'''
def g(n_row,n_col,num_nodes,radius,length,width):
    shiftx = n_col//2
    shifty = n_row//2  
    internal_angle = 2*np.pi/(num_nodes -1)
    node_pos_dict = {}
    edge_pos_dict = {}
    for node in range(num_nodes):
        if node == 0:
            x = shiftx
            y = shifty
        else:
            x = int((length*np.cos((node-1)*internal_angle))) + shiftx
            y = int((length*np.sin((node-1)*internal_angle))) + shifty
        pos = [(x,y)]
        for x_ in np.arange(x-radius,x+radius+1):
            for y_ in np.arange(y-radius,y+radius+1):
                if (x-x_)**2 + (y-y_)**2 <= radius**2 and (x_,y_) not in pos:
                    pos.append((x_,y_))
        node_pos_dict[node] = pos
    node_pos = list(node_pos_dict.values())

    for edge in range(1,num_nodes):
        edge_pos_dict[edge] = return_edge_pos(node_pos[0],node_pos[edge],width)
    edge_pos = list(edge_pos_dict.values())
    num_edges = len(edge_pos)
    
    n_pts = []
    for n in range(num_nodes):
        n_pts.extend(node_pos[n])
    e_pts = []
    for e in range(num_edges):
        e_pts.extend(edge_pos[e])
    points = n_pts + e_pts
    count = 0
    for point in points:
        p = (point[0],point[1])
        points[count] = (p[0],p[1])
        count+=1
    return points, node_pos

import random

def build_neighbor_pos_dictionary(n_row, n_col,num_nodes,radius,length,width):
    list_of_all_pos_in_ca, list_node_pos = g(n_row,n_col,num_nodes,radius,length,width)
    dict_of_neighbors_pos_lists = {pos: build_neighbor_pos_list(pos, list_of_all_pos_in_ca) for pos in list_of_all_pos_in_ca}

    return dict_of_neighbors_pos_lists, list_node_pos

def build_neighbor_pos_list(pos, list_of_all_pos_in_ca):
    r, c = pos

    l = [(r+i, c+j)
         for i in [-1, 0, 1]
         for j in [-1, 0, 1]
         if (r+i,c+j) in list_of_all_pos_in_ca
         if not (j == 0 and i == 0)]

    return l

binomial = np.random.binomial
shuffle = np.random.shuffle
random_choice = random.choice


def locate_empty_neighbor_position(neighbor_pos_list,agent_dictionary):

    empty_neighbor_pos_list = [pos for pos in neighbor_pos_list if pos not in agent_dictionary]
    if empty_neighbor_pos_list:
        empty_pos = random_choice(empty_neighbor_pos_list)
        return empty_pos

    else:
        return None

def act(PLOT_ID, pos, agent_dictionary,dictionary_of_neighbor_pos_lists):
    neighbors = agent_dictionary[pos][1]
    good = 0
    bad = 0    
    for neighbor in neighbors:
        if neighbor in agent_dictionary:
            if agent_dictionary[neighbor][0] == PLOT_ID:
                good +=1/8
            else:
                bad += 1/8
    empty_pos = locate_empty_neighbor_position(dictionary_of_neighbor_pos_lists[pos], agent_dictionary)          
    if good >= bad and empty_pos is not None:
        agent_dictionary[empty_pos] = [PLOT_ID,dictionary_of_neighbor_pos_lists[empty_pos]]
        if good >= 0.5:
            del agent_dictionary[pos]
    else:
        del agent_dictionary[pos]

if __name__ == "__main__":
    import time
    start = time.time()
    import matplotlib.pyplot as plt
    MAX_REPS = 1000
    NUM_NODES = 6
    RADIUS = 5
    LENGTH = 20
    WIDTH = 3
    N_ROW = 2*RADIUS + 2*LENGTH
    N_COL = 2*RADIUS + 2*LENGTH
    DICT_NEIGHBOR_POS, NODE_POS = build_neighbor_pos_dictionary(N_ROW,N_COL,NUM_NODES,RADIUS,LENGTH,WIDTH)

    center_r = NODE_POS[0][0][0]
    center_c = NODE_POS[0][0][1]
    center_pos = (center_r,center_c)

    cell_dict = {center_pos:[2,DICT_NEIGHBOR_POS[center_pos]]}

    for i in range(1,NUM_NODES):
        center_r = NODE_POS[i][0][0]
        center_c = NODE_POS[i][0][1]
        center_pos = (center_r,center_c)

        cell_dict[center_pos] = [1,DICT_NEIGHBOR_POS[center_pos]]   
        
    rep = 0
    while rep < MAX_REPS:
        if rep % 5 == 0:
            visualization_matrix = np.zeros((N_ROW+1,N_COL+1))
            for cell in cell_dict:
                visualization_matrix[cell] = cell_dict[cell][0]
            plt.imshow(visualization_matrix,interpolation='none',cmap='seismic')
            plt.show()
            plt.close()
        cell_list = list(cell_dict.keys())
        shuffle(cell_list)
        for cell in cell_list:
            act(cell_dict[cell][0],cell,cell_dict,DICT_NEIGHBOR_POS)
        
    end = time.time()
    total = end-start
    print("total time", total)