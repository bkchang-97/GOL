import pygame
import numpy as np
import sys
from typing import Tuple, List
from dataclasses import dataclass, field

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class grid:
    gridSize: Tuple[int, int] # columns, rows == x,y
    data: np.ndarray
    generations: int

    def __init__(self, size, setup):
        self.gridSize = size
        self.data = setup(self.gridSize)
        self.generations = 0


#--------------------------------------------------------------------
# Initialization functions -- used by the constructor.
#--------------------------------------------------------------------

# function: randStart
# Purpose: employed by grid __init__ (constructor) to give initial value to data
# param: size
# returns: an np array of size size, whose values are uniformly selected from range(states)
def randStart(size):
    return np.random.randint(states, size = size)

# function: glideStart
# Purpose: employed by grid __init__ (constructor) to give initial value to data
# param: size
# returns: an np array of size size, whose values are all zero, except for positions
# (2,0), (0,1), (2,1), (1,2), and (2,2), whose values are 1. Intended to be used
# on a game w 2 states.
def glideStart(size):
    grid = np.random.randint(states-1, size= size)
    grid[0][2] = 1
    grid[1][0] = 1
    grid[1][2] = 1
    grid[2][1] = 1
    grid[2][2] = 1
    return grid

def shipStart(size):
    grid = np.random.randint(states - 1, size=size)
    grid[2][1] = 1
    grid[5][1] = 1
    grid[1][2] = 1
    grid[1][3] = 1
    grid[5][3] = 1
    grid[1][4] = 1
    grid[2][4] = 1
    grid[3][4] = 1
    grid[4][4] = 1
    return grid
# --------------------------------------------------------------------
# Rule function -- used by the evolve function. Only one is used
# in any game definition
# --------------------------------------------------------------------

# function: ruleGOL
# purpose: applies a set of rules given a current state and a set of tallies over neighbor states
# params: cell, an element from range(states), where states is the global variable
#           tallies, tallies[k] = number of neighbors of state k, for all k in the range of states
# returns: a new state based on the classic rules of the game of life.
#           See https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
# Note: assumes a two-state game, where 0 is "dead" and 1 is "alive"

def ruleGOL(cell, tallies):
    if cell == 1:
        if tallies[1] < 2 or tallies[1] > 3:
            return 0
        else:
            return cell
    elif cell == 0:
        if tallies[1] == 3:
            return 1
        else:
            return cell

# --------------------------------------------------------------------
# Neighbor functions -- used by the evolve function. Only one is used
# in any game definition
# --------------------------------------------------------------------
# returns a list of neighbors in a square around x,y
def neighborSquare(x, y):
    return [(x-1, y-1), (x, y-1), (x+1, y-1),
            (x-1, y), (x+1, y),
            (x-1, y+1), (x, y+1), (x+1, y+1)]

# function: tally_neighbors
# purpose: counts a given cell's the neighbors' states
# params: grid, an np array of data from a grid, containing states of all cells
#         position, the current cell position (a Tuple)
#         neighborSet, a function that when called on position x,y returns a list of x,y's neighbors
# returns: a list whose entries, tally[k] are the number of valid neighbors of x,y whose state is k.

def tally_neighbors(grid, position, neighborSet):
    valid_neighbours = []
    for neighbour in neighborSet(position[0], position[1]):
        if (neighbour[0] in range(np.size(grid.data, 1))) and (neighbour[1] in range(np.size(grid.data, 0))):
            valid_neighbours.append(neighbour)
    neighbour_state_list = []
    for p in valid_neighbours:
        neighbour_state_list.append(grid.data[(p[1], p[0])])
    state_list = [0] * states
    for state in neighbour_state_list:
        state_list[state] += 1
    return state_list


# student: putting it all together.
# function: evolve
# purpose: to increment the automata by one time step. Given an array representing the automaton at the
# start of the time step (the start grid), this function creates an array for the end of the time step
# (the end grid) by applying the rule specified in function apply_rule to every position in the array.

def evolve(gr, apply_rule, neighbors):
    evolved = np.zeros((np.size(gr.data, 0), np.size(gr.data, 1)), dtype = int)
    for y in range(np.size(gr.data, 0)):
        for x in range(np.size(gr.data, 1)):
            position = (x, y)
            state_list = tally_neighbors(gr, position, neighbors)
            evolved[(y, x)] = apply_rule(gr.data[(y, x)], state_list)
    gr.generations += 1
    gr.data = evolved

# function draw_block
# purpose: draw a rectangle of color acolor for grid location x,y. Uses globals pad and sqSize.
# function solution is:     pygame.draw.rect(screen, acolor,
#   [upper left horiz pixel location, upper left vertical pixel location, sqSize, sqSize])
# returns: nothing
def draw_block(x, y, acolor):
    pygame.draw.rect(screen, acolor, [x*(sqSize + pad + pad) + pad, y*(sqSize + pad + pad) + pad, sqSize, sqSize])


# function: draw
# purpose: translates the game representation from the grid, to an image on the screen
# param: gr, a grid. for every position in gr.data, computes a color based on the state
# in that location, and then makes a call to draw_block to place that color into the pygame
# screen. Also passes the grid location so draw_block can compute the correct screen location.
# the new color is represented in HSVA (see https://www.pygame.org/docs/ref/color.html#pygame.Color.hsva
# and has h = (360 // states) * current state, s = 100, and v = 50 (we just left A at its default value).
# you may want to experiment with these values for artistic effect. :)
# returns: nothing
def draw(gr):
    for y in range(np.size(gr.data, 0)):
        for x in range(np.size(gr.data, 1)):
            draw_block(x, y, (0 * gr.data[y][x] , 17 * gr.data[y][x] , 255 * gr.data[y][x]))



# following are the game, grid, and screen parameters for the problem

# Set the number of states to use within each cell
states = 2  # we leave this as a global variable since it doesn't change.

# words to display on the window
pygame.display.set_caption("CPSC203 Life")

# the game state is maintained in a grid object.
# grid data values will be updated upon every click of the clock.
# parameters are the (width, height) dimensions of the grid, and a
# function that initializes the start state
#g = grid((100, 150), randStart)
g = grid((75, 75), randStart)

print(g.data[0][1])
# drawing parameters that determine the look of the grid when it's shown.
# These can be set, but defaults are probably fine
sqSize = 7  # size of the squares in pixels
pad = sqSize // 5 # the number of pixels between each square

# computed from parameters above and grid g dimensions
s = (75*(sqSize + pad+pad), 75*(sqSize + pad+pad))# YOUR CODE HERE! dimensions of pixels in screen window

screen = pygame.display.set_mode(s)  # initializes the display window

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# given: necessary for gracefully ending game loop (pygame)
def handleInputEvents():
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close...
            sys.exit(0)  # quit

# -------- Main Program Loop -----------
while True:
    # --- Main event loop
    handleInputEvents()

    # --- Draw the grid
    # this function loops over the data in the grid object
    # and draws appropriately colored rectangles.
    draw(g)

    # --- Game logic should go here
    # evolve( g, rule, neighbors)
    # g -- an object of type grid, previously initialized to hold data start state
    # rule -- a function that applies the game rule, given a cell state and a neighbor tally
    # neighbors -- a function that returns a list of neighbors relative to a given x,y position.
    #evolve(g, ruleCycle, neighborDiamond)
    evolve(g, ruleGOL, neighborSquare)

    # --- Mysterious reorientation that every pygame application seems to employ
    pygame.display.flip()

    # --- Limit to 60 frames per second
    clock.tick(60)

# Close the window and quit.
pygame.quit()