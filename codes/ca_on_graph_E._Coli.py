import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#fig, axes = plt.subplots(1,cycle+star+complete,sharex=True,sharey = True)
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

'''Cycle Graph'''

def cycle(num_nodes,radius,length,width,n_row,n_col):
    shiftx = (n_col-1)//2 + length
    shifty = (n_row-1)//2
    internal_angle = 2*np.pi/num_nodes
    node_pos_dict = {}
    edge_pos_dict = {}
    for node in range(num_nodes):
        x = int((length/2*np.sin(np.pi/num_nodes))*np.cos(node*internal_angle+np.pi/2))
        y = int((length/2*np.sin(np.pi/num_nodes))*np.sin(node*internal_angle+np.pi/2))
        pos = [(x,y)]
        for x_ in np.arange(x-radius,x+radius+1):
            for y_ in np.arange(y-radius,y+radius+1):
                if (x-x_)**2 + (y-y_)**2 <= radius**2 and (x_,y_) not in pos:
                    pos.append((x_,y_))
        node_pos_dict[node] = pos
    node_pos = list(node_pos_dict.values())
    for edge in range(num_nodes):
        edge_pos_dict[edge] = return_edge_pos(node_pos[edge],node_pos[(edge+1)%num_nodes],width)
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
        p = (point[0]+shiftx,point[1]+shifty)
        points[count] = (p[0],p[1])
        count+=1
    # visualization_matrix = np.zeros((n_row,n_col))
    # for p in points:
    #     visualization_matrix[p] = 1
    # plt.imshow(visualization_matrix,interpolation='none',cmap='seismic',vmin=0,vmax=2)
    # plt.show()
    return points, node_pos


'''Star Graph'''
def star(num_nodes,radius,length,width,n_row,n_col):
    shiftx = (n_col-1)//2 + length
    shifty = (n_row-1)//2  
    internal_angle = 2*np.pi/(num_nodes -1)
    node_pos_dict = {}
    edge_pos_dict = {}
    for node in range(num_nodes):
        if node == 0:
            x = 0
            y = 0
        else:
            x = int((length/2*np.sin(np.pi/(num_nodes-1)))*np.cos((node-1)*internal_angle+np.pi/2))
            y = int((length/2*np.sin(np.pi/(num_nodes-1)))*np.sin((node-1)*internal_angle+np.pi/2))
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
        p = (point[0]+shiftx,point[1]+shifty)
        points[count] = (p[0],p[1])
        count+=1
    # visualization_matrix = np.zeros((n_row,n_col))
    # for p in points:
    #     visualization_matrix[p] = 1
    # plt.imshow(visualization_matrix,interpolation='none',cmap='seismic',vmin=0,vmax=2)
    # plt.show()
    return points, node_pos


'''Complete Graph'''
def complete(num_nodes,radius,length,width,n_row,n_col):
    shiftx = (n_col-1)//2 + length
    shifty = (n_row-1)//2
    internal_angle = 2*np.pi/num_nodes
    node_pos_dict = {}
    edge_pos_dict = {}
    for node in range(num_nodes):
        x = int((length/2*np.sin(np.pi/num_nodes))*np.cos(node*internal_angle+np.pi/2))
        y = int((length/2*np.sin(np.pi/num_nodes))*np.sin(node*internal_angle+np.pi/2))
        pos = [(x,y)]
        for x_ in np.arange(x-radius,x+radius+1):
            for y_ in np.arange(y-radius,y+radius+1):
                if (x-x_)**2 + (y-y_)**2 <= radius**2 and (x_,y_) not in pos:
                    pos.append((x_,y_))
        node_pos_dict[node] = pos
    node_pos = list(node_pos_dict.values())
    e = 0
    list_e = []
    for edge in range(num_nodes):
        for edge2 in range(num_nodes):
            if edge2 != edge:
                list_e.append((edge,edge2))
                if (edge2,edge) not in list_e:
                    edge_pos_dict[e] = return_edge_pos(node_pos[edge],node_pos[edge2],width)
                    e += 1
    edge_pos = list(edge_pos_dict.values())
    num_edges = e

    n_pts = []
    for n in range(num_nodes):
        n_pts.extend(node_pos[n])
    e_pts = []
    for e in range(num_edges):
        e_pts.extend(edge_pos[e])
    points = n_pts + e_pts
    count = 0
    for point in points:
        p = (point[0]+shiftx,point[1]+shifty)
        points[count] = (p[0],p[1])
        count+=1
    # visualization_matrix = np.zeros((n_row,n_col))
    # for p in points:
    #     visualization_matrix[p] = 1
    # plt.imshow(visualization_matrix,interpolation='none',cmap='seismic',vmin=0,vmax=2)
    # plt.show()
    return points, node_pos

from numba import jit, int64
import random


#### METHODS TO BUILD THE WORLD OF OUR CA ####
def build_neighbor_pos_dictionary(n_row, n_col,type_graph,num_nodes,radius,length,width):
    """
    Create dictionary containing the list of all neighbors (value) for a central position (key)
    :param n_row:
    :param n_col:
    :return: dictionary where the key= central position, and the value=list of neighboring positions around that center
    """
    if type_graph == "cycle":
        list_of_all_pos_in_ca, list_node_pos = cycle(num_nodes,radius,length,width,n_row,n_col)
    if type_graph == "star":
        list_of_all_pos_in_ca, list_node_pos = star(num_nodes,radius,length,width,n_row,n_col)
    if type_graph == "complete":
        list_of_all_pos_in_ca, list_node_pos = complete(num_nodes,radius,length,width,n_row,n_col)

    dict_of_neighbors_pos_lists = {pos: build_neighbor_pos_list(pos, list_of_all_pos_in_ca) for pos in list_of_all_pos_in_ca}

    return dict_of_neighbors_pos_lists, list_node_pos

def build_neighbor_pos_list(pos, list_of_all_pos_in_ca):
    """
    Use list comprehension to create a list of all positions in the wild_type's Moore neighborhood.
    Valid positions are those that are within the confines of the domain (n_row, n_col)
    and not the same as the wild_type's current position.

    :param pos: wild_type's position; tuple
    :param n_row: maximum width of domain; integer
    :param n_col: maximum height of domain; integer
    :return: list of all valid positions around the wild_type
    """
    # Unpack the tuple containing the wild_type's position
    r, c = pos

    l = [(r+i, c+j)
         for i in [-1, 0, 1]
         for j in [-1, 0, 1]
         if (r+i,c+j) in list_of_all_pos_in_ca
         if not (j == 0 and i == 0)]

    return l

#### METHODS TO SPEED UP EXECUTION ####
binomial = np.random.binomial
shuffle = np.random.shuffle
random_choice = random.choice

#@jit(int64(), nopython=True)
def divide_q(prob):
    DIVISION_PROB = prob #1 / 24.0
    verdict = binomial(1, DIVISION_PROB)

    return  verdict


@jit(int64(), nopython=True)
def die_q():
    DEATH_PROB = 0.01 #1 / 100.0
    verdict = binomial(1, DEATH_PROB)

    return verdict

#### CREATE WILD_TYPE CLASSES ####
class Mutant(object):

    def __init__(self, pos, dictionary_of_neighbor_pos_lists):
        
        self.pos = pos
        self.divisions_remaining = 10
        self.neighbor_pos_list = dictionary_of_neighbor_pos_lists[self.pos]
        self.PLOT_ID = 2

    def locate_empty_neighbor_position(self, agent_dictionary):
        """
        Search for empty positions in Moore neighborhood. If there is more thant one free position,
        randomly select one and return it

        :param agent_dictionary: dictionary of agents, key=position, value = wild_type; dict
        :return: Randomly selected empty position, or None if no empty positions
        """

        empty_neighbor_pos_list = [pos for pos in self.neighbor_pos_list if pos not in agent_dictionary]
        if empty_neighbor_pos_list:
            empty_pos = random_choice(empty_neighbor_pos_list)
            return empty_pos

        else:
            return None

    def act(self, agent_dictionary, dictionary_of_neighbor_pos_lists):
        """
        Wild_Type carries out its actions, which are division and death. Wild_Type will divide if it is lucky and
        there is an empty position in its neighborhood. Wild_Type dies either spontaneously or if it exceeds its
        maximum number of divisions.

        :param agent_dictionary: dictionary of agents, key=position, value = wild_type; dict
        :return: None
        """

        #### WILD_TYPE TRIES TO DIVIDE ####
        divide = divide_q(1/12)
        if divide == 1:
            empty_pos = self.locate_empty_neighbor_position(agent_dictionary)
            if empty_pos is not None:

                #### CREATE NEW DAUGHTER WILD_TYPE AND IT TO THE WILD_TYPE DICTIONARY ####
                daughter_mutant = Mutant(empty_pos, dictionary_of_neighbor_pos_lists)
                agent_dictionary[empty_pos] = daughter_mutant

                self.divisions_remaining -= 1

        #### DETERMINE IF WILD_TYPE WILL DIE ####
        spontaneous_death = die_q()
        if self.divisions_remaining <= 0 or spontaneous_death == 1:
            del agent_dictionary[self.pos]

class Wild_Type(object):

    def __init__(self, pos, dictionary_of_neighbor_pos_lists):
        
        self.pos = pos
        self.divisions_remaining = 10
        self.neighbor_pos_list = dictionary_of_neighbor_pos_lists[self.pos]
        self.PLOT_ID = 1

    def locate_empty_neighbor_position(self, agent_dictionary):
        """
        Search for empty positions in Moore neighborhood. If there is more thant one free position,
        randomly select one and return it

        :param agent_dictionary: dictionary of agents, key=position, value = wild_type; dict
        :return: Randomly selected empty position, or None if no empty positions
        """

        empty_neighbor_pos_list = [pos for pos in self.neighbor_pos_list if pos not in agent_dictionary]
        if empty_neighbor_pos_list:
            empty_pos = random_choice(empty_neighbor_pos_list)
            return empty_pos

        else:
            return None

    def act(self, agent_dictionary, dictionary_of_neighbor_pos_lists):
        """
        Wild_Type carries out its actions, which are division and death. Wild_Type will divide if it is lucky and
        there is an empty position in its neighborhood. Wild_Type dies either spontaneously or if it exceeds its
        maximum number of divisions.

        :param agent_dictionary: dictionary of agents, key=position, value = wild_type; dict
        :return: None
        """

        #### WILD_TYPE TRIES TO DIVIDE ####
        divide = divide_q(1/24)
        if divide == 1:
            empty_pos = self.locate_empty_neighbor_position(agent_dictionary)
            if empty_pos is not None:

                #### CREATE NEW DAUGHTER WILD_TYPE AND IT TO THE WILD_TYPE DICTIONARY ####
                daughter_wild_type = Wild_Type(empty_pos, dictionary_of_neighbor_pos_lists)
                agent_dictionary[empty_pos] = daughter_wild_type

                self.divisions_remaining -= 1

        #### DETERMINE IF WILD_TYPE WILL DIE ####
        spontaneous_death = die_q()
        if self.divisions_remaining <= 0 or spontaneous_death == 1:
            del agent_dictionary[self.pos]

if __name__ == "__main__":
    import time
    start = time.time()

    GRAPH = "complete"
    MAX_REPS = 1000
    NUM_NODES = 2
    RADIUS = 10
    LENGTH = 10*(NUM_NODES**2)
    WIDTH = 3
    N_ROW = 2*RADIUS + 2
    N_COL = 4*RADIUS + LENGTH
    DICT_NEIGHBOR_POS, NODE_POS = build_neighbor_pos_dictionary(N_ROW,N_COL,GRAPH,NUM_NODES,RADIUS,LENGTH,WIDTH)
    DICT_NEIGHBOR_POS2, NODE_POS2 = build_neighbor_pos_dictionary(N_ROW+4*RADIUS,N_COL,GRAPH,NUM_NODES,RADIUS,LENGTH*1.5,WIDTH)
    DICT_NEIGHBOR_POS3, NODE_POS3 = build_neighbor_pos_dictionary(N_ROW+8*RADIUS,N_COL,GRAPH,NUM_NODES,RADIUS,LENGTH*2,WIDTH)
    DICT_NEIGHBOR_POS4, NODE_POS4 = build_neighbor_pos_dictionary(N_ROW+12*RADIUS,N_COL,GRAPH,NUM_NODES,RADIUS,LENGTH*2.5,WIDTH)

    # cdict = {
    #    'red': ((0.0, 0.0, 0.0),
    #            (1.0, 1.0, 1.0)),
    #    'blue':((0.0, 0.0, 0.0),
    #            (1.0, 0.0, 0.0)),
    #    'green':((0.0, 0.0, 1.0),
    #             (1.0, 1.0, 1.0))
    #    }
    # cell_cmap = mcolors.LinearSegmentedColormap('my_colormap', cdict, 100)

    center_r = NODE_POS[0][0][0] + (N_ROW - 1)//2
    center_c = NODE_POS[0][0][1] + (N_COL - 1)//2
    center_pos = (center_r,center_c)
    center_r2 = NODE_POS2[0][0][0] + (N_ROW+4*RADIUS - 1)//2
    center_c2 = NODE_POS2[0][0][1] + (N_COL - 1)//2
    center_pos2 = (center_r2,center_c2)
    center_r3 = NODE_POS3[0][0][0] + (N_ROW+8*RADIUS - 1)//2
    center_c3 = NODE_POS3[0][0][1] + (N_COL - 1)//2
    center_pos3 = (center_r3,center_c3)
    center_r4 = NODE_POS4[0][0][0] + (N_ROW+12*RADIUS - 1)//2
    center_c4 = NODE_POS4[0][0][1] + (N_COL - 1)//2
    center_pos4 = (center_r4,center_c4)

    initial_cell = Mutant(center_pos,DICT_NEIGHBOR_POS)
    initial_cell2 = Mutant(center_pos2,DICT_NEIGHBOR_POS2)
    initial_cell3 = Mutant(center_pos3,DICT_NEIGHBOR_POS3)
    initial_cell4 = Mutant(center_pos4,DICT_NEIGHBOR_POS4)
    cell_dict = {center_pos:initial_cell}
    cell_dict2 = {center_pos2:initial_cell2}
    cell_dict3 = {center_pos3:initial_cell3}
    cell_dict4 = {center_pos4:initial_cell4}

    for i in range(1,NUM_NODES):
        center_r = NODE_POS[i][0][0] + (N_ROW - 1)//2
        center_c = NODE_POS[i][0][1] + (N_COL - 1)//2
        center_pos = (center_r,center_c)
        center_r2 = NODE_POS2[i][0][0] + (N_ROW+4*RADIUS - 1)//2
        center_c2 = NODE_POS2[i][0][1] + (N_COL - 1)//2
        center_pos2 = (center_r2,center_c2)
        center_r3 = NODE_POS3[i][0][0] + (N_ROW+8*RADIUS - 1)//2
        center_c3 = NODE_POS3[i][0][1] + (N_COL - 1)//2
        center_pos3 = (center_r3,center_c3)
        center_r4 = NODE_POS4[i][0][0] + (N_ROW+12*RADIUS - 1)//2
        center_c4 = NODE_POS4[i][0][1] + (N_COL - 1)//2
        center_pos4 = (center_r4,center_c4)

        next_cell = Wild_Type(center_pos,DICT_NEIGHBOR_POS)
        next_cell2 = Wild_Type(center_pos2,DICT_NEIGHBOR_POS2)
        next_cell3 = Wild_Type(center_pos3,DICT_NEIGHBOR_POS3)
        next_cell4 = Wild_Type(center_pos4,DICT_NEIGHBOR_POS4)
        cell_dict[center_pos] = next_cell
        cell_dict2[center_pos2] = next_cell2
        cell_dict3[center_pos3] = next_cell3
        cell_dict4[center_pos4] = next_cell4


    for rep in range(MAX_REPS):
        if rep % 5 == 0:
            visualization_matrix = np.zeros((N_ROW+12*RADIUS,N_COL + int(2.5*LENGTH)))
            for cell in cell_dict.values():
                visualization_matrix[cell.pos] = cell.PLOT_ID
            for cell in cell_dict2.values():
                visualization_matrix[cell.pos] = cell.PLOT_ID
            for cell in cell_dict3.values():
                visualization_matrix[cell.pos] = cell.PLOT_ID
            for cell in cell_dict4.values():
                visualization_matrix[cell.pos] = cell.PLOT_ID
            plt.imshow(visualization_matrix,interpolation='none',cmap='seismic')
            # if GRAPH == "star":
            #     img_name = "C:\\Users\\Jason\\Desktop\\Clinic Research\\CellAUtomAta\\Codes\\images_star_E._Coli\\" + str(rep).zfill(5) + '.jpg'
            # elif GRAPH == "cycle":
            #     img_name = "C:\\Users\\Jason\\Desktop\\Clinic Research\\CellAUtomAta\\Codes\\images_cycle_E._Coli\\" + str(rep).zfill(5) + '.jpg'
            # elif GRAPH == "complete":
            #     img_name = "C:\\Users\\Jason\\Desktop\\Clinic Research\\CellAUtomAta\\Codes\\images_complete_E._Coli\\" + str(rep).zfill(5) + '.jpg'
            # plt.savefig(img_name)
            plt.show()
            plt.close()
            
        cell_list = list(cell_dict.values())
        cell_list2 = list(cell_dict2.values())
        cell_list3 = list(cell_dict3.values())
        cell_list4 = list(cell_dict4.values())
        shuffle(cell_list)
        shuffle(cell_list2)
        shuffle(cell_list3)
        shuffle(cell_list4)
        for cell in cell_list:
            cell.act(cell_dict,DICT_NEIGHBOR_POS)
        for cell in cell_list2:
            cell.act(cell_dict2,DICT_NEIGHBOR_POS2)
        for cell in cell_list3:
            cell.act(cell_dict3,DICT_NEIGHBOR_POS3)
        for cell in cell_list4:
            cell.act(cell_dict4,DICT_NEIGHBOR_POS4)
    end = time.time()
    total = end-start
    print("total time", total)