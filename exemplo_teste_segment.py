# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:59:06 2023

@author: User-Aline
"""

import sys
import math
from math import *
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Circle, Wedge, Polygon

sys.path.append("src/")
from src.person import Person
from src.F_formation import F_formation
from src.overall_density5 import OverallDensity



def main():
    f_formation = F_formation()
    people = []

    # Criação dos individuos na cena (caso inicial apenas 1 individuo por cluster)
    #p1 = Person(x=1, y=1, th=np.deg2rad(60), id_node=0 )
    #people.append(p1)
    #p2 = Person(x=2, y=0, th=np.deg2rad(90), id_node=1)
    #people.append(p2)
    #p3 = Person(x =3, y=1, th=np.deg2rad(120), id_node=2)
    #people.append(p3)
    #p4 = Person(x= 2, y=3, th=np.deg2rad(-90), id_node=3)
    #people.append(p4)
    #group1 = f_formation.Face_to_face(x1=13, y1=9.8, th1=np.deg2rad(90), x2=13, y2=12, th2=np.deg2rad(270))
    #p, xc, yc, rc = group1
    #for person in p:
    #    people.append(person)

    # Criação dos grupos de pessoas
    
    #group3 = f_formation.semi_circle(xc=1, yc=2, rc=1.5)
    #group2 = f_formation.Side_by_side(x1=-10, y1=-2.5, th1=np.deg2rad(90), x2= -9.47, y2=-2.5, th2=np.deg2rad(90))
    #group3 = f_formation.triang_eq(x1=-5,y1=-2.5, th1=np.deg2rad(45), x2 = -3, y2=-2.5, th2=np.deg2rad(135), x3=-4, y3=-1, th3=np.deg2rad(-90))    
    #group2 = f_formation.Side_by_side(x1=-2.5, y1=-2.3, th1=np.deg2rad(0), x2= -2.5, y2=-1.75, th2=np.deg2rad(0))
    #group4 = f_formation.retangular(x1=-2, y1= 10, th1=np.deg2rad(135), x2= -5, y2 = 10, th2=np.deg2rad(45), x3=-2, y3=12, th3=np.deg2rad(-135), x4=-5, y4=12, th4=np.deg2rad(-45))
    
    #caso tres pessoas formation aleatoria
    #group3 = f_formation.triangular(x1=2, y1=5, th1=np.deg2rad(90), x2=3, y2=6.6, th2=np.deg2rad(135), x3=3, y3=8.8, th3=np.deg2rad(225))
    
    #group2 = f_formation.L_shaped(x1=-1, y1= 5, th1=np.deg2rad(-90), x2=0, y2=4.5, th2=np.deg2rad(180))
    #group2 = f_formation.L_shaped(x1=-1, y1= 5, th1=np.deg2rad(0), x2=0, y2=4.5, th2=np.deg2rad(90))
    #group2 = f_formation.L_shaped(x1=-1, y1= 5, th1=np.deg2rad(0), x2=0, y2=6.5, th2=np.deg2rad(-90))
    #group2 = f_formation.Face_to_face(x1=2, y1=10, th1=np.deg2rad(45), x2=3, y2=12, th2=np.deg2rad(-135))
    #group2 = f_formation.Face_to_face(x1=12, y1=10, th1=np.deg2rad(0), x2=13, y2=10, th2=np.deg2rad(180))
    #group2 = f_formation.Face_to_face(x1=12, y1=10, th1=np.deg2rad(90), x2=12, y2=12, th2=np.deg2rad(-90))
    #group2 = f_formation.Face_to_face(x1=-2, y1=10, th1=np.deg2rad(135), x2=-3, y2=12, th2=np.deg2rad(-45))
    #group1 = Person(x=12.5,y=0,th=np.deg2rad(135),id_node=9)
    #group2 = f_formation.v_shaped(x1=-10, y1=7.5, th1=np.deg2rad(135), x2=-10, y2=8.7, th2=np.deg2rad(225))
    #group3 = f_formation.triangular(x1=2,y1=0.5, th1=np.deg2rad(0), x2=2, y2=1, th2=np.deg2rad(0), x3=3, y3=0.75, th3=np.deg2rad(180))
    group5 = f_formation.Circular(xc=1, yc=2, rc=1.5)
    #p_move = Person(x=-6, y=11, th=np.deg2rad(0), id_node=4)#teste para uma pessoa que chegou ao grupo em uma das regiões anteriores
    #p_move2 = Person(x=5, y=5, th=np.deg2rad(-45), id_node=3)

    #p2, xc, yc, rc = group2
    #for person in p2:
    #    people.append(person)
    #p3, xc, yc, rc = group3
    #for person in p3:
    #    people.append(person)
    #people.append(p_move)
    #people.append(p_move2)
    #p4, xc, yc, rc = group4
    #for person in p4:
    #    people.append(person)
    p5, xc, yc, rc = group5
    for person in p5:
        people.append(person)
    #p1 = Person(x=6, y=1, th=np.deg2rad(-90), id_node=14)
    #people.append(p1)
    #for person in people:
    #    print(person.id_node)
#######################################################################################
    

    #Encontra as regiões sociais da cena
    G = OverallDensity(person=people, zone='Social', map_resolution=400, window_size=1)
    G.make_graph()
    G.boundary_estimate()
    #
    #Teste 19-10-23
    mean_orientation=G.calculate_mean_orientation(people)
    print(f'mean_orientation_formation:{np.rad2deg(mean_orientation)}')
    #if is_point_back()
    #
    fig, ax = plt.subplots(figsize=(12,6))
    for pi in people:
        pi.draw(ax)
    G.draw_overall(drawDensity=False, drawCluster=True, drawGraph=False, drawSegment=True, people=people)
    plt.xlabel('x (meters)')
    plt.ylabel('y (meters)')
    plt.axis('equal')  # Para manter a proporção dos eixos igual
    plt.show()

if __name__ == "__main__":
    main()

