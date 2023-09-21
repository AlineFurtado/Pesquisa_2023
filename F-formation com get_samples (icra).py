# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 10:16:41 2023

@author: User-Aline
"""
import sys
import numpy as np
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from math import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
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

class F_formation:

    def __init__(self, sd=1.2):
        self.sd = sd

    def calculate_coordinates(self, x_values, y_values):
        xc = np.mean(x_values)
        yc = np.mean(y_values)
        rc = np.max(np.sqrt((x_values - xc)**2 + (y_values - yc)**2))
        return xc, yc, rc

    def create_group(self, *people_coords):
        people = [Person(x, y, th) for x, y, th in people_coords]
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
################################################################################
    # num é o número de approach em frente
    def approach_samples(self, p1, p2, xc, yc, rc, num=2):
        # relativa ao centro do O-sapce exceto o approach central. ]
        # Ou seja Número total de targets = num + 2 #Vou fixar em 3 targets por grupo
        # calcula o raio do P-space r e de R-space R, em seguida define a posição dos approach samples.
        # best approach sample posicionado dentro do R-space com o raio passando pelo ponto (xc,yc)
        # demais samples entre -45 e +45 ao redor dele

        # calculo do P-space (1.10=_radius+personal_space)
        r = rc + 1.10 #r = raio do P-space 

        # Então calcula o raio de approximação r<rapp<R
        #rapp = r + self.sd/2
        rapp = r + 0.6

        samples = []
        x1, y1, th1 = p1.get_coords()
        x2, y2, th2 = p2.get_coords()
        # para a maioria dos casos, excessões são calculadas ou atribuidas localmente
        thc = th1 + np.deg2rad(90)

        # calculo das coordenadas do Best approach sample (Máximo ou Mínimo Reward conforme cada caso)
        if y2 == y1:  #  side by side
            if th1==th2:
                xs1 = xc
                ys1 = rapp + yc
                samples.append([xs1, ys1])
            else: #face to face horizontal
                xs1 = xc
                ys1 = rapp + yc
                samples.append([xs1, ys1])
                minang = -35
                maxang = 35
                num = 3
                angles = np.linspace(np.deg2rad(minang), np.deg2rad(maxang), num)
                for angle in angles:
                    sx = rapp * np.cos(np.deg2rad(-90) + angle) + xc
                    sy = rapp * np.sin(np.deg2rad(-90) + angle) + yc
                    samples.append([sx, sy])
        else:  #  face to face vertical
            if x1 == x2:
                if th1==np.deg2rad(90):
                    ys1 = yc
                    xs1 = xc - rapp
                    samples.append([xs1, ys1])
                    minang = -35
                    maxang = 35
                    num = 3
                    angles = np.linspace(np.deg2rad(minang), np.deg2rad(maxang), num)
                    for angle in angles:
                        sx = rapp * np.cos(np.deg2rad(0) + angle) + xc
                        sy = rapp * np.sin(np.deg2rad(0) + angle) + yc
                        samples.append([sx, sy])
                else: #v-sheppard
                    ys1 = yc
                    xs1 = xc - rapp
                    samples.append([xs1, ys1])
            else:  # se L-shaped
                if (x2 > x1) & (y1 > y2):
                    thc = np.deg2rad(45)
                    xs1 = xc - rapp * np.sin(thc)
                    ys1 = yc - rapp * np.cos(thc)
                    samples.append([xs1, ys1])

                else:  # face to face transversal
                    thc = th1+np.deg2rad(90)
                    xs1 = xc - rapp*np.cos(th1+np.deg2rad(90))
                    ys1 = yc - rapp*np.sin(th1+np.deg2rad(90))
                    samples.append([xs1, ys1])
                    minang = -35
                    maxang = 35
                    num = 3
                    angles = np.linspace(np.deg2rad(minang), np.deg2rad(maxang), num)
                    for angle in angles:
                        sx = rapp * np.cos(thc + angle) + xc
                        sy = rapp * np.sin(thc + angle) + yc
                        samples.append([sx, sy])

        # definindo mais approach samples:
        minang = -35
        maxang = 35
        angles = np.linspace(np.deg2rad(minang), np.deg2rad(maxang), num)
        for angle in angles:  # encontra os approach-points de frente para o centro do O-space
            if (x2 > x1) & (y1 > y2):
                thc = np.deg2rad(225)
            else:
                if th1 == th2:
                    thc = th1
                else:
                    if (th2-th1) == np.deg2rad(90):
                        thc = np.deg2rad(180)
                    else:
                        if (y2 > y1) and (th1 == np.deg2rad(45)):
                            thc = np.deg2rad(315)

            sx = rapp * np.cos(thc + angle) + xc
            sy = rapp * np.sin(thc + angle) + yc
            samples.append([sx, sy])

        #print(samples)
        return samples

    # para grupos de 3 ou mais pessoas, informar o qtd = n. de pessoas
    def approach_samples_three(self, p1, p2, p3, xc, yc, rc):

        # calculo do P-space (1.10=_radius+personal_space)
        r = rc + 1.10
        # Então calcula o raio de approximação r<rapp<R
        #rapp = r + self.sd/2
        rapp = r + 0.6

        samples = []
        x1, y1, th1 = p1.get_coords()
        x2, y2, th2 = p2.get_coords()
        x3, y3, th3 = p3.get_coords()

        # Caso F-formatio semi_circle
        if th2 == np.deg2rad(270):
            num= 5
            angles= np.linspace(np.deg2rad(-45), np.deg2rad(45), num)
            #print(angles)
            for angle in angles:
                xs = rapp*np.cos(th2 + angle)+xc
                ys = rapp*np.sin(th2 + angle)+yc
                samples.append([xs, ys])
                #print(samples)
        else: #caso triangulo isosceles
            if th1==np.deg2rad(0):
                xs = rapp*np.cos(th2-np.deg2rad(25))+xc
                ys = rapp*np.sin(th2-np.deg2rad(25))+yc
                samples.append([xs, ys])
                sx = rapp*np.cos(th3+np.deg2rad(25))+xc
                sy = rapp*np.sin(th3+np.deg2rad(25))+yc
                samples.append([sx,sy])

            else:
            # caso F-formation triangulo equilatero
                xs1 = rapp * np.cos(th1) + xc
                ys1 = rapp * np.sin(th1) + yc
                samples.append([xs1, ys1])
                xs2 = rapp * np.cos(th2) + xc
                ys2 = rapp * np.sin(th2) + yc
                samples.append([xs2, ys2])
                xs3 = rapp * np.cos(th3) + xc
                ys3 = rapp * np.sin(th3) + yc
                samples.append([xs3, ys3])

        return samples

    def approach_samples_four(self, p1, p2, p3, p4, xc, yc, rc):

        # calculo do P-space (1.10=_radius+personal_space)
        r = rc + 1.10
        # Então calcula o raio de approximação r<rapp<R
        #rapp = r + self.sd/2
        rapp = r + 0.6

        samples = []
        x1, y1, th1 = p1.get_coords()
        x2, y2, th2 = p2.get_coords()
        x3, y3, th3 = p3.get_coords()
        x4, y4, th4 = p4.get_coords()

        xs1 = rapp*np.cos(np.deg2rad(45)) + xc
        ys1 = rapp*np.sin(np.deg2rad(45)) + yc
        samples.append([xs1, ys1])
        xs2 = rapp * np.cos(np.deg2rad(135)) + xc
        ys2 = rapp * np.sin(np.deg2rad(135)) + yc
        samples.append([xs2, ys2])
        xs3 = rapp * np.cos(np.deg2rad(225)) + xc
        ys3 = rapp * np.sin(np.deg2rad(225)) + yc
        samples.append([xs3, ys3])
        xs4 = rapp * np.cos(np.deg2rad(315)) + xc
        ys4 = rapp * np.sin(np.deg2rad(315)) + yc
        samples.append([xs4, ys4])

        return samples

    def approach_samples_five(self, p1, p2, p3, p4, p5, xc, yc, rc):

        # calculo do P-space (1.10=_radius+personal_space)
        r = rc + 1.10
        # Então calcula o raio de approximação r<rapp<R
        #rapp = r + self.sd/2
        rapp = r + 0.6

        samples = []
        x1, y1, th1 = p1.get_coords()
        x2, y2, th2 = p2.get_coords()
        x3, y3, th3 = p3.get_coords()
        x4, y4, th4 = p4.get_coords()
        x5, y5, th5 = p5.get_coords()

        xs1 = rapp*np.cos(th1) + xc
        ys1 = rapp*np.sin(th1) + yc
        samples.append([xs1, ys1])
        xs2 = rapp * np.cos(th2) + xc
        ys2 = rapp * np.sin(th2) + yc
        samples.append([xs2, ys2])
        xs3 = rapp * np.cos(th3) + xc
        ys3 = rapp * np.sin(th3) + yc
        samples.append([xs3, ys3])
        xs4 = rapp * np.cos(th4) + xc
        ys4 = rapp * np.sin(th4) + yc
        samples.append([xs4, ys4])
        xs5 = rapp * np.cos(th5) + xc
        ys5 = rapp * np.sin(th5) + yc
        samples.append([xs5, ys5])

        return samples
    
    def approach_samples_one(self, p,xc,yc,rc):
        
        # calculo do P-space (=personal_space)
        r = rc + 0.6
        # Então calcula o raio de approximação r<rapp<R
        #rapp = r + self.sd/2
        rapp = r + 1.10
        
        samples = []
        (x,y,th)= p.get_coords()
        num= 5
        angles = np.linspace(np.deg2rad(-90), np.deg2rad(90), num)
        for angle in angles:
            sx = rapp * np.cos(th + angle) + x
            sy = rapp* np.sin(th + angle) + y
            samples.append([sx,sy])
        
        return samples

#################################################################################
    def approach_samples_combined(self, people, xc, yc, rc):
        num_people = len(people)

        if num_people == 1:
            return self.approach_samples_one(people[0], xc, yc, rc)
        elif num_people == 2:
            p1, p2 = people
            return self.approach_samples(p1, p2, xc, yc, rc)
        elif num_people == 3:
            p1, p2, p3 = people
            return self.approach_samples_three(p1, p2, p3, xc, yc, rc)
        elif num_people == 4:
            p1, p2, p3, p4 = people
            return self.approach_samples_four(p1, p2, p3, p4, xc, yc, rc)
        elif num_people == 5:
            p1, p2, p3, p4, p5 = people
            return self.approach_samples_five(p1, p2, p3, p4, p5, xc, yc, rc)
        else:
            raise ValueError("Invalid number of people")

    def get_combined_samples(self, group_info):
        combined_samples = []
        for group in group_info:
            people, xc, yc, rc = group
            samples = self.approach_samples_combined(people, xc, yc, rc)
            combined_samples.extend(samples)
        return combined_samples

    def draw_formation(self, ax, people, xc, yc, rc, samples, color):
        p_colors = ['red', 'green', 'blue', 'purple', 'orange']

        for i, p in enumerate(people):
            p.draw(ax)

        # P_Space
        pspace = Circle((xc, yc), radius=(rc + 1.10), fill=False, ls='--', color='b')
        ax.add_patch(pspace)

        # R_Space
        R = rc + 2.20
        Rspace = Circle((xc, yc), radius=R, fill=False, ls='--', color='g')
        ax.add_patch(Rspace)

        for sample in samples:
            ax.scatter(sample[0], sample[1], color=color)

        #ax.scatter(xc, yc, color='black')

        ax.set_aspect('equal')
        ax.tick_params(labelsize=12)
        ax.grid(False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Formation')
