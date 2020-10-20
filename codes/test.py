import numpy as np 
import matplotlib.pyplot as plt

cycle = 0
star = 1
complete = 0
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
	shift = (n_row-1)//2
	internal_angle = 2*np.pi/num_nodes
	node_pos_dict = {}
	edge_pos_dict = {}
	for node in range(num_nodes):
		x = int((length/2*np.sin(np.pi/num_nodes))*np.cos(node*internal_angle))
		y = int((length/2*np.sin(np.pi/num_nodes))*np.sin(node*internal_angle))
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
		p = (point[0]+shift,point[1]+shift)
		points[count] = (p[0],p[1])
		count+=1
	return points, node_pos


'''Star Graph'''
def star(num_nodes,radius,length,width,n_row,n_col):
	shift = (n_row-1)//2		
	internal_angle = 2*np.pi/(num_nodes -1)
	node_pos_dict = {}
	edge_pos_dict = {}
	for node in range(num_nodes):
		if node == 0:
			x = 0
			y = 0
		else:
			x = int((length/2*np.sin(np.pi/(num_nodes-1)))*np.cos((node-1)*internal_angle))
			y = int((length/2*np.sin(np.pi/(num_nodes-1)))*np.sin((node-1)*internal_angle))
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
		p = (point[0]+shift,point[1]+shift)
		points[count] = (p[0],p[1])
		count+=1
	return points, node_pos


'''Complete Graph'''
def complete(num_nodes,radius,length,width,n_row,n_col):
	shift = (n_row-1)//2
	internal_angle = 2*np.pi/num_nodes
	node_pos_dict = {}
	edge_pos_dict = {}
	for node in range(num_nodes):
		x = int((length/2*np.sin(np.pi/num_nodes))*np.cos(node*internal_angle))
		y = int((length/2*np.sin(np.pi/num_nodes))*np.sin(node*internal_angle))
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
		p = (point[0]+shift,point[1]+shift)
		points[count] = (p[0],p[1])
		count+=1
	return points, node_pos

from numba import jit, int64 # just in time compiler to speed up CA
import random


#### METHODS TO BUILD THE WORLD OF OUR CA ####
def build_neighbor_pos_dictionary(n_row, n_col,type_graph,num_nodes,radius,length,width):
    """
    Create dictionary containing the list of all neighbors (value) for a central position (key)
    :param n_row:
    :param n_col:
    :return: dictionary where the key= central position, and the value=list of neighboring positions around that center
    """
    if type_graph == cycle:
    	list_of_all_pos_in_ca, list_node_pos = cycle(num_nodes,radius,length,width,n_row,n_col)
    if type_graph == star:
    	list_of_all_pos_in_ca, list_node_pos = star(num_nodes,radius,length,width,n_row,n_col)
    if type_graph == complete:
    	list_of_all_pos_in_ca, list_node_pos = complete(num_nodes,radius,length,width,n_row,n_col)

    print(list_of_all_pos_in_ca)
    dict_of_neighbors_pos_lists = {pos: build_neighbor_pos_list(pos, n_row, n_col) for pos in list_of_all_pos_in_ca}

    return dict_of_neighbors_pos_lists, list_node_pos

def build_neighbor_pos_list(pos, n_row, n_col):
    """
    Use list comprehension to create a list of all positions in the cell's Moore neighborhood.
    Valid positions are those that are within the confines of the domain (n_row, n_col)
    and not the same as the cell's current position.

    :param pos: cell's position; tuple
    :param n_row: maximum width of domain; integer
    :param n_col: maximum height of domain; integer
    :return: list of all valid positions around the cell
    """
    # Unpack the tuple containing the cell's position
    r, c = pos

    l = [(r+i, c+j)
         for i in [-1, 0, 1]
         for j in [-1, 0, 1]
         if 0 <= r + i < n_row
         if 0 <= c + j < n_col
         if not (j == 0 and i == 0)]

    return l

#### METHODS TO SPEED UP EXECUTION ####
binomial = np.random.binomial
shuffle = np.random.shuffle
random_choice = random.choice


@jit(int64(), nopython=True)  # NOTE: declaring the return type is not required
def divide_symmetrically_q():
    SYMMETRIC_DIVISION_PROB = 0.3 #3/10.0
    verdict = binomial(1, SYMMETRIC_DIVISION_PROB)

    return verdict


@jit(int64(), nopython=True)
def divide_q():
    DIVISION_PROB = 0.0416 #1 / 24.0
    verdict = binomial(1, DIVISION_PROB)

    return  verdict


@jit(int64(), nopython=True)
def die_q():
    DEATH_PROB = 0.01 #1 / 100.0
    verdict = binomial(1, DEATH_PROB)

    return verdict

#### CREATE CELL CLASSES ####
class CancerCell(object):

    def __init__(self, pos, dictionary_of_neighbor_pos_lists):
        """
        Initialize Cancer Cell
        :param pos: position of cancer cell; tuple
        :param dictionary_of_neighbor_pos_lists: used to tell the cell the positions of its neighbors
        :return:
        """

        self.pos = pos
        self.divisions_remaining = 10
        self.neighbor_pos_list = dictionary_of_neighbor_pos_lists[self.pos]
        self.PLOT_ID = 1

    def locate_empty_neighbor_position(self, agent_dictionary):
        """
        Search for empty positions in Moore neighborhood. If there is more thant one free position,
        randomly select one and return it

        :param agent_dictionary: dictionary of agents, key=position, value = cell; dict
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
        Cell carries out its actions, which are division and death. Cell will divide if it is lucky and
        there is an empty position in its neighborhood. Cell dies either spontaneously or if it exceeds its
        maximum number of divisions.

        :param agent_dictionary: dictionary of agents, key=position, value = cell; dict
        :return: None
        """

        #### CELL TRIES TO DIVIDE ####
        divide = divide_q()
        if divide == 1:
            empty_pos = self.locate_empty_neighbor_position(agent_dictionary)
            if empty_pos is not None:

                #### CREATE NEW DAUGHTER CELL AND IT TO THE CELL DICTIONARY ####
                daughter_cell = CancerCell(empty_pos, dictionary_of_neighbor_pos_lists)
                agent_dictionary[empty_pos] = daughter_cell

                self.divisions_remaining -= 1

        #### DETERMINE IF CELL WILL DIE ####
        spontaneous_death = die_q()
        if self.divisions_remaining <= 0 or spontaneous_death == 1:
            del agent_dictionary[self.pos]

class CancerStemCell(CancerCell):

    def __init__(self, pos, dictionary_of_neighbor_pos_lists):
        """
        Inherits explicitly from CancerCell Class.

        Difference between cancer stem cells and cancer cells is that the former are immortal, have infinite replicative
        potential, and can divide either symmetrically or asymmetrically

        :param pos: cell's position; tuple
        :return: CancerStemCell object
        """
        super(CancerStemCell, self).__init__(pos, dictionary_of_neighbor_pos_lists)
        self.PLOT_ID = 2

    def act(self, agent_dictionary, dictionary_of_neighbor_pos_lists):
        """
        Cancer stem cell carries out its actions, which is to divide.

        :param agent_dictionary: dictionary of agents, key=position, value = cell; dict
        :return: None
        """

        divide = divide_q()
        if divide == 1:

            empty_pos = self.locate_empty_neighbor_position(agent_dictionary)
            if empty_pos is not None:

                symmetric_division = divide_symmetrically_q()
                if symmetric_division == 1:
                    daughter_cell = CancerStemCell(empty_pos, dictionary_of_neighbor_pos_lists)

                else:
                    daughter_cell = CancerCell(empty_pos, dictionary_of_neighbor_pos_lists)

                agent_dictionary[empty_pos] = daughter_cell

if __name__ == "__main__":
	import time
	start = time.time()

	N_ROW = 121
	N_COL = 121
	MAX_REPS = 500
	DICT_NEIGHBOR_POS, NODE_POS = build_neighbor_pos_dictionary(N_ROW,N_COL,star,6,5,5*(6**2),2)

	center_r = NODE_POS[0][0][0] + (N_ROW - 1)//2
	center_c = NODE_POS[0][0][1] + (N_COL - 1)//2
	center_pos = (center_r,center_c)

	initial_cancer_stem_cell = CancerStemCell(center_pos,DICT_NEIGHBOR_POS)
	cell_dict = {center_pos:initial_cancer_stem_cell}

	for rep in range(MAX_REPS):
		if rep % 5 == 0:
			visualization_matrix = np.zeros((N_ROW,N_COL))
			for cell in cell_dict.values():
				visualization_matrix[cell.pos] = cell.PLOT_ID
			plt.imshow(visualization_matrix,interpolation='none',cmap='seismic',vmin=0,vmax=2)
			img_name = "C:\\Users\\Jason\\Desktop\\Clinic Research\\CellAutomata\\Codes\\" + str(rep/5).zfill(5) + '.jpg'
			plt.savefig(img_name)
			
		cell_list = cell_dict.values()
		shuffle(cell_list)
		for cell in cell_list:
			cell.act(cell_dict,DICT_NEIGHBOR_POS)
	end = time.time()
	total = end-start
	print("total time", total)