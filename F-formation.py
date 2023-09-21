# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:29:21 2023

@author: User-Aline
"""

import math
import decimal
from decimal import Decimal

import os
import tempfile
from subprocess import check_output
import networkx as nx
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon

import numpy as np
import itertools
import pickle

from sklearn.neighbors import KDTree
from scipy.signal import medfilt
from scipy.spatial import ConvexHull
#from person import Person
from matplotlib import cm
from math import *
from skimage.measure import approximate_polygon

#atualizada
class F_formation:

    def __init__(self, sd=1.2):
        self.sd = sd
        self.id_counter = 0

    def calculate_coordinates(self, x_values, y_values):
        xc = np.mean(x_values)
        yc = np.mean(y_values)
        rc = np.max(np.sqrt((x_values - xc)**2 + (y_values - yc)**2))
        return xc, yc, rc

#modificada pra incluir o id_node
    def create_group(self, *people_coords):
        people = [Person(x, y, th, self.id_counter + i) for i, (x, y, th) in enumerate(people_coords)]
        self.id_counter += len(people_coords)
        return people

    def Face_to_face(self, x1, y1, th1, x2, y2, th2):
        people = self.create_group((x1, y1, th1), (x2, y2, th2))
        xc, yc, rc = self.calculate_coordinates([x1, x2], [y1, y2])
        return people, xc, yc, rc

    def L_shaped(self, x1, y1, th1, x2, y2, th2):
        people = self.create_group((x1, y1, th1), (x2, y2, th2))
        rc = y1 - y2
        xc, yc = x1, y2
        return people, xc, yc, rc

    def Side_by_side(self, x1, y1, th1, x2, y2, th2):
        people = self.create_group((x1, y1, th1), (x2, y2, th2))
        xc = (x2 + x1) / 2
        yc = y1 + (xc - x1) * np.tan(np.deg2rad(45))
        rc = (yc - y1) / np.cos(np.deg2rad(45))
        return people, xc, yc, rc

    def v_shaped(self, x1, y1, th1, x2, y2, th2):
        people = self.create_group((x1, y1, th1), (x2, y2, th2))
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        rc = yc - y1
        return people, xc, yc, rc

    def triangular(self, x1, y1, th1, x2, y2, th2, x3, y3, th3):
        people = self.create_group((x1, y1, th1), (x2, y2, th2), (x3, y3, th3))
        xc, yc, rc = self.calculate_coordinates([x1, x2, x3], [y1, y2, y3])
        return people, xc, yc, rc

    def triang_eq(self, x1, y1, th1, x2, y2, th2, x3, y3, th3):
        people = self.create_group((x1, y1, th1), (x2, y2, th2), (x3, y3, th3))
        xc = x3
        yc = (y3 - y1) / 3 + y1
        rc = (2 * (y3 - y1)) / 3
        return people, xc, yc, rc

    def semi_circle(self, xc, yc, rc):
        num = 3
        angles = np.linspace(np.deg2rad(0), np.deg2rad(180), num)
        people_coords = [(rc * np.cos(angle) + xc, rc * np.sin(angle) + yc, np.deg2rad(180) + angle) for angle in angles]
        people = self.create_group(*people_coords)
        return people, xc, yc, rc

    def retangular(self, x1, y1, th1, x2, y2, th2, x3, y3, th3, x4, y4, th4):
        people = self.create_group((x1, y1, th1), (x2, y2, th2), (x3, y3, th3), (x4, y4, th4))
        xc = (x2 + x4) / 2
        yc = (y1 + y3) / 2
        rc = xc - x1
        return people, xc, yc, rc

    def Circular(self, xc, yc, rc):
        num = 5
        angles = np.linspace(np.deg2rad(0), np.deg2rad(300), num)
        people_coords = [(rc * np.cos(angle) + xc, rc * np.sin(angle) + yc, np.deg2rad(180) + angle) for angle in angles]
        people = self.create_group(*people_coords)
        return people, xc, yc, rc
