import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numba import jit, int64 # just in time compiler to speed up CA
import random

def build_neighbor_pos_list(pos, x, y, z):
 
    x_pos, y_pos, z_pos = pos

    l = [(x_pos+i, y_pos+j, z_pos + k)
         for i in [-1, 0, 1]
         for j in [-1, 0, 1]
         for k in [-1, 0, 1]
         if 0 <= x_pos + i < x
         if 0 <= y_pos + j < y
         if 0 <= z_pos + k < z
         if not (i == 0 and j == 0 and k == 0)]
    return l

binomial = np.random.binomial
shuffle = np.random.shuffle
random_choice = random.choice


@jit(int64(), nopython=True) 
def divide_symmetrically_q():
    SYMMETRIC_DIVISION_PROB = 0.3 
    verdict = binomial(1, SYMMETRIC_DIVISION_PROB)
    return verdict


@jit(int64(), nopython=True)
def divide_q():
    DIVISION_PROB = 0.0416 #1 / 24.0
    verdict = binomial(1, DIVISION_PROB)
    return  verdict


@jit(int64(), nopython=True)
def die_q():
    DEATH_PROB = 0.01 
    verdict = binomial(1, DEATH_PROB)
    return verdict

#Create cancer cell classes
class CancerCell(object):

    def __init__(self, pos, x, y, z):
        self.pos = pos
        self.divisions_remaining = 10
        self.list_neighbors = build_neighbor_pos_list(pos, x, y, z)
        self.PLOT_ID = 1

    def locate_empty_neighbor_position(self, agent_dictionary):
        empty_neighbor_pos_list = [pos for pos in self.list_neighbors if pos not in agent_dictionary]
        if empty_neighbor_pos_list:
            empty_pos = random_choice(empty_neighbor_pos_list)
            return empty_pos
        else:
            return None

    def act(self, agent_dictionary, x, y, z):
        #Cell tries to divide
        divide = divide_q()
        if divide == 1:
            empty_pos = self.locate_empty_neighbor_position(agent_dictionary)
            if empty_pos is not None:

                #Creates daughter cell and adds it to the dictionary
                daughter_cell = CancerCell(empty_pos, x, y, z)
                agent_dictionary[empty_pos] = daughter_cell

                self.divisions_remaining -= 1

        #Determines if cell will die
        spontaneous_death = die_q()
        if self.divisions_remaining <= 0 or spontaneous_death == 1:
            del agent_dictionary[self.pos]

class CancerStemCell(CancerCell):

    def __init__(self, pos, x, y, z):
        #Copies Cancer Cell class but....
        super(CancerStemCell, self).__init__(pos, x, y, z)
        #... the ID must be different (doy)...
        self.PLOT_ID = 2

    def act(self, agent_dictionary, x, y, z):
        
        divide = divide_q()
        if divide == 1:

            empty_pos = self.locate_empty_neighbor_position(agent_dictionary)
            if empty_pos is not None:
                #.... and the cell has a different probability of symmetric division .....
                symmetric_division = divide_symmetrically_q()
                if symmetric_division == 1:
                    daughter_cell = CancerStemCell(empty_pos, x, y, z)

                else:
                    daughter_cell = CancerCell(empty_pos, x, y, z)

                agent_dictionary[empty_pos] = daughter_cell
    #..... and the cancer stem cell does not die.

if __name__ == "__main__":
    import time
    start = time.time()

    X = 50
    Y = 50
    Z = 50

    MAX_REPS = 500

    # Creates an origin for our "space" which in reality only includes the cancer cells and sites where cells have died, 
    # with no other empty spaces 
    center_x = int(round(X/2.0))
    center_y = int(round(Y/2.0))
    center_z = int(round(Z/2.0))
    center_pos = (center_x, center_y, center_z)

    initial_cancer_stem_cell = CancerStemCell(center_pos, X, Y, Z)
    cell_dict = {center_pos:initial_cancer_stem_cell}

    for rep in range(MAX_REPS):
        """---------------------------------------------------------------------------------------------------------
        Makes a 3d plot of the tumor at every 10th time-stamp (repetition)
        ---------------------------------------------------------------------------------------------------------"""
        if rep % 10 == 0:
            #cc = cancer cell, csc = cancer stem cell
            x_cc = []
            y_cc = []
            z_cc = []
            x_csc = []
            y_csc = []
            z_csc = []
            for cell in cell_dict.values():
                if cell.PLOT_ID == 1:
                    x_cc.append(cell.pos[0])
                    y_cc.append(cell.pos[1])
                    z_cc.append(cell.pos[2])
                elif cell.PLOT_ID == 2:
                    x_csc.append(cell.pos[0])
                    y_csc.append(cell.pos[1])
                    z_csc.append(cell.pos[2])

            fig =plt.figure(figsize=(X,Y))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_cc,y_cc,z_cc,c = 'b',alpha = 1.0,s=10.0)
            ax.scatter(x_csc,y_csc,z_csc,c = 'r',alpha = 1.0,s=10.0)
            #Forces the size of the 3d plot to be a cube of volume X*Y*Z
            ax.scatter([0,X],[0,Y],[0,Z],c = 'w',alpha = 0.0,s=10.0)

            img_name = "C:\\Users\\Jason\\Desktop\\Clinic Research\\CellAutomata\\Codes\\images\\" + str(rep).zfill(5) + '.jpg'
            plt.savefig(img_name)
            plt.close()

        #Following two lines randomize order of cell actions
        cell_list = cell_dict.values()
        shuffle(cell_list)
        for cell in cell_list:
            #The cells do their thang
            cell.act(cell_dict, X, Y, Z)
    end = time.time()
    total = end-start
    print("total time", total)
