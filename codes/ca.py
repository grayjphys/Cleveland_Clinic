import numpy as np #numerical python
import matplotlib.pyplot as plt #Package used for plotting
from numba import jit, int64 # just in time compiler to speed up CA
import random

#### METHODS TO BUILD THE WORLD OF OUR CA ####
def build_neighbor_pos_dictionary(x, y):
    """
    Create dictionary containing the list of all neighbors (value) for a central position (key)
    :param x:
    :param y:
    :return: dictionary where the key= central position, and the value=list of neighboring positions around that center
    """
    list_of_all_pos_in_ca = [(r, c) for r in np.arange(0, x) for c in np.arange(0, y)]

    dict_of_neighbors_pos_lists = {pos: build_neighbor_pos_list(pos, x, y) for pos in list_of_all_pos_in_ca}

    return dict_of_neighbors_pos_lists



def build_neighbor_pos_list(pos, x, y):
    """
    Use list comprehension to create a list of all positions in the cell's Moore neighborhood.
    Valid positions are those that are within the confines of the domain (x, y)
    and not the same as the cell's current position.

    :param pos: cell's position; tuple
    :param x: maximum width of domain; integer
    :param y: maximum height of domain; integer
    :return: list of all valid positions around the cell
    """
    # Unpack the tuple containing the cell's position
    r, c = pos

    l = [(r+i, c+j)
         for i in [-1, 0, 1]
         for j in [-1, 0, 1]
         if 0 <= r + i < x
         if 0 <= c + j < y
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

    X = 2000
    Y = 2000
    MAX_REPS  = 10000
    DICTIONARY_OF_NEIGHBOR_POS_LISTS = build_neighbor_pos_dictionary(X, Y)

    center_r = int(round(X/2.0))
    center_c = int(round(Y/2.0))
    center_pos = (center_r, center_c)

    initial_cancer_stem_cell = CancerStemCell(center_pos, DICTIONARY_OF_NEIGHBOR_POS_LISTS)
    cell_dictionary = {center_pos:initial_cancer_stem_cell}

    for rep in range(MAX_REPS):
        #### USE BELOW LINE FOR PYTHON 2 ####
        # cell_list = cell_dictionary.values()

        #### USE BELOW LINE FOR PYTHON 3 ####
        cell_list = list(cell_dictionary.values())
        shuffle(cell_list)

        if rep % 5 == 0:
            visualization_matrix = np.zeros((X,Y))
            for cell in cell_list:
                visualization_matrix[cell.pos] = cell.PLOT_ID
            plt.imshow(visualization_matrix,interpolation='none',cmap='seismic',vmin=0,vmax=2)
            img_name = "C:\\Users\\Jason\\Desktop\\Clinic Research\\CellAutomata\\Codes\\images\\" + str(rep).zfill(5) + '.jpg'
            plt.savefig(img_name)
            plt.close()

        for cell in cell_list:
            cell.act(cell_dictionary, DICTIONARY_OF_NEIGHBOR_POS_LISTS)


    end = time.time()
    total = end-start
    print("total time", total)

#X = 1000, Y = 1000, MAX_REPS = 10000, total time 1266.975870847702