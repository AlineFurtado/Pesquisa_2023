import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib import cm
from math import *

#modificada
class Person(object):

    x = 0
    y = 0
    th = 0

    xdot = 0
    ydot = 0

    _radius = 0.5#0.045  # raio do corpo da pessoa
    personal_space = 0.50  # raio da região de personal_space

    """ Public Methods """

    def __init__(self, x=0, y=0, th=0, id_node=None):
        self.x = x
        self.y = y
        self.th = th
        self.id_node = id_node

    def get_coords(self):
        return [self.x, self.y, self.th]
    
    def get_parallel_point_in_zone(self, intersection_point, zone):
        # Calculate the vector connecting intersection_point to the current person's position
        vector_to_intersection = np.array([intersection_point[0] - self.x, intersection_point[1] - self.y])
        
        # Calculate the angle between the vector and the person's orientation
        angle = np.arctan2(vector_to_intersection[1], vector_to_intersection[0]) - self.th
        
        # Calculate the distance to move parallelly based on the zone
        parallel_distance = self.personal_space if zone == 'Personal' else self.personal_space * 2
        
        # Calculate the new position using the adjusted vector and parallel_distance
        new_x = self.x + parallel_distance * np.cos(angle)
        new_y = self.y + parallel_distance * np.sin(angle)
        
        return [new_x, new_y]


    def draw(self, ax):
        # define grid.
        npts = 100
        x = np.linspace(self.x-5, self.x+5, npts)
        y = np.linspace(self.y-5, self.y+5, npts)

        X, Y = np.meshgrid(x, y)

        # Corpo
        body = Circle((self.x, self.y), radius=self._radius, fill=False)
        ax.add_patch(body)

        # Orientação
        x_aux = self.x + self._radius * np.cos(self.th)
        y_aux = self.y + self._radius * np.sin(self.th)
        heading = plt.Line2D((self.x, x_aux), (self.y, y_aux), lw=3, color='k')
        ax.add_line(heading)

        # Personal Space
        #space = Circle((self.x, self.y), radius=(self._radius+self.personal_space), fill=False, ls='--', color='r')
        #ax.add_patch(space)
