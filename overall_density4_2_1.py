import sys
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

from skimage.measure import find_contours
from skimage import measure
from sklearn.neighbors import KDTree
from scipy.signal import medfilt
from scipy.spatial import ConvexHull
from matplotlib import cm
from math import *



from person import Person


#com minhas modificaçõs
class OverallDensity(object):
    #
    temporal_weight = dict()
    dict = {'Intimate': 0.45, 'Personal': 1.2, 'Social': 3.6, 'Public': 10.0}

    """ Public Methods """

    def __init__(self, person=None, zone=None, map_resolution=100, window_size=1):
        #
        self.samples = []  # Lista para armazenar as amostras coletadas
        self.x = np.empty([0, 0])
        self.y = np.empty([0, 0])
        self.G = nx.Graph()
        self.cluster = list()
        self.density = list()
        self.max_dist = list()
        self.person = person
        self.window_size = window_size
        self.proxemics_th = self.dict[zone]
        self.map_resolution = map_resolution
        self.cluster_coordinates = {}

    def set_zone(self, value=None):
        self.dict['Personal'] = value
        self.proxemics_th = value

    def norm_range(self, value, in_max, in_min, out_max, out_min):
        return ((out_max - out_min) * (value - in_min)) / (in_max - in_min) + out_min;

    def pose2map(self, pose):
        return int(self.norm_range(pose, max(self.x), min(self.x), len(self.x) - 1, 0))

    def map2pose(self, pose):
        return float(self.norm_range(pose, len(self.x) - 1, 0, max(self.x), min(self.x)))

    def get_nearest_neighbors(self, seed, dataset):
        tree = KDTree(dataset, leaf_size=1)
        result, dists = tree.query_radius([seed], r=self.proxemics_th ** 2.0, return_distance=True, sort_results=True, count_only=False)
        result = result.item()
        dists = dists.item()
        temp_count_only = dists <= self.proxemics_th
        result = result[temp_count_only]
        dists = dists[temp_count_only]
        self.max_dist.append(max(dists))
        return result

    def local_density(self, p_i, p):
        sigma_h = 2.0
        sigma_r = 1.0
        sigma_s = 4.0 / 3.0
        alpha = np.arctan2(p.y - p_i.y, p.x - p_i.x) - p_i.th - pi / 2.0
        nalpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # Normalizando no intervalo [-pi, pi)
        sigma = sigma_r if nalpha <= 0 else sigma_h
        a = cos(p_i.th) ** 2.0 / 2.0 * sigma ** 2.0 + sin(p_i.th) ** 2.0 / 2.0 * sigma_s ** 2.0
        b = sin(2.0 * p_i.th) / 4.0 * sigma ** 2.0 - sin(2.0 * p_i.th) / 4.0 * sigma_s ** 2.0
        c = sin(p_i.th) ** 2.0 / 2.0 * sigma ** 2.0 + cos(p_i.th) ** 2.0 / 2.0 * sigma_s ** 2.0
        rad = (fabs(np.arctan2(p.y - p_i.y, p.x - p_i.x)) - fabs(p_i.th))
        AGF = np.exp(-(a * (p.x - p_i.x) ** 2.0 + 2.0 * b * (p.x - p_i.x) * (p.y - p_i.y) + c * (p.y - p_i.y) ** 2.0))
        return AGF

    def is_edge(self, z, x, y):
        #
        if z[x, y] != 0:
            if x > 0 and x < z.shape[0] - 1 and y > 0 and y < z.shape[1] - 1:
                if bool(z[x - 1, y] <= z[x, y]) ^ bool(z[x + 1, y] <= z[x, y]) or bool(z[x, y - 1] <= z[x, y]) ^ (
                        z[x, y + 1] <= z[x, y]):
                    return z[x, y]

    def set_threshold(self, z, x, y, th):
        #
        if z[x, y] >= th:
            if x > 0 and x < z.shape[0] - 1 and y > 0 and y < z.shape[1] - 1:
                if z[x - 1, y] > th and z[x + 1, y] > th and z[x, y - 1] > th and z[x, y + 1] > th:
                    return 0.0
                else:
                    return 1.0  # Value that defines the density of the point in the plot
        return 0.0

    def ccw(self, A, B, C):
        # http://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
        return fabs(C[1] - A[1]) * fabs(B[0] - A[0]) > fabs(B[1] - A[1]) * fabs(C[0] - A[0])

    def line_intersection(self, A, B, C, D):
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def edges_weigth(self, pi, pj):
        pi2pj = sqrt((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2)
        piFoA_x = pi.x + ((pi2pj / 2) * cos(pi.th))
        piFoA_y = pi.y + ((pi2pj / 2) * sin(pi.th))
        pjFoA_x = pj.x + ((pi2pj / 2) * cos(pj.th))
        pjFoA_y = pj.y + ((pi2pj / 2) * sin(pj.th))
        deltaFoA = sqrt((piFoA_x - pjFoA_x) ** 2 + (piFoA_y - pjFoA_y) ** 2)
        pi2pjFoA = sqrt((pi.x - pjFoA_x) ** 2 + (pi.y - pjFoA_y) ** 2)
        pj2piFoA = sqrt((pj.x - piFoA_x) ** 2 + (pj.y - piFoA_y) ** 2)
        exponent = 1.0
        if self.line_intersection([fabs(pi.x), fabs(pi.y)], [fabs(pj.x), fabs(pj.y)], [piFoA_x, piFoA_y],
                                  [pjFoA_x, pjFoA_y]):
            exponent = 2.0
        if (deltaFoA) >= min(pi2pjFoA, pj2piFoA):
            exponent += 0.5
        if pi2pj != 0:
            return round((1 - (deltaFoA / (pi2pj * 2.0))) ** exponent, 2)
        return 0

    def make_graph(self):
        dataset = np.array([[p.x, p.y] for p in self.person], dtype=float)
        for seed in dataset:
            neighbors = self.get_nearest_neighbors(seed, dataset)
            for node in neighbors:
                self.G.add_node(self.person[node].id_node, pos=(self.person[node].x, self.person[node].y))
                if neighbors[0] == node:
                    continue
                # self.G.add_node(self.person[node].id_node, pos=(self.person[node].x ,self.person[node].y))
                level_interaction = self.edges_weigth(self.person[neighbors[0]], self.person[node])
                key = str(int(self.person[neighbors[0]].id_node)) + '_' + str(int(self.person[node].id_node))
                self.temporal_weight.setdefault(key, []).append(level_interaction)
                if self.window_size > len(self.temporal_weight[key]):
                    for padding in range(1, self.window_size):
                        self.temporal_weight.setdefault(key, []).append(level_interaction)
                if self.window_size != 0:
                    median_filter = medfilt(self.temporal_weight[key], self.window_size)
                    level_interaction = median_filter[int(self.window_size / -2)]
                if level_interaction >= 0.5 or self.proxemics_th > self.dict['Personal']:
                   self.G.add_edge(self.person[neighbors[0]].id_node, self.person[node].id_node,
                                    weight=level_interaction)

    #encontra os grupos e estima as zonas sociais conforme a proxêmica
    def boundary_estimate(self):
    #
        margin = max(4.0, self.proxemics_th)
        self.x = np.linspace(min([p.x for p in self.person]) - margin,
                             max([p.x for p in self.person]) + margin, self.map_resolution)
        self.y = np.linspace(min([p.y for p in self.person]) - margin,
                             max([p.y for p in self.person]) + margin, self.map_resolution)
        X, Y = np.meshgrid(self.x, self.y)
        self.density = []
        self.cluster = []
        #
        for component in nx.connected_components(self.G):
            #print(f'component: {component}') 
            if len(component) > 1:
                sigma_th = max(self.max_dist)
                surrounding_th = max(self.proxemics_th, sigma_th)
            else:
                sigma_th = self.dict['Social']
                surrounding_th = min(self.proxemics_th, sigma_th)
            rows = set()
            cols = set()
            for node in component:
                pose = [[p.x, p.y] for p in self.person if p.id_node == node]
                pose_x = pose[0][0]
                pose_y = pose[0][1]
                pose2map_col = int(self.norm_range(pose_x, max(self.x), min(self.x), len(self.x) - 1, 0))
                pose2map_row = int(self.norm_range(pose_y, max(self.y), min(self.y), len(self.y) - 1, 0))
                deltaX = (X[pose2map_row, pose2map_col] - X)
                deltaY = (Y[pose2map_row, pose2map_col] - Y)
                euclidean = np.sqrt(deltaX ** 2.0 + deltaY ** 2.0)
                for row in range(Y.shape[0]):
                    for col in range(X.shape[0]):
                        if euclidean[row, col] <= surrounding_th:
                            rows.add(row)
                            cols.add(col)
            #
            local_cluster = np.zeros(X.shape, dtype=float)
            local_density = np.zeros(X.shape, dtype=float)
            for node in component:
                person = [p for p in self.person if p.id_node == node]
                for row in rows:
                    for col in cols:
                        cell = Person(X[row, col], Y[row, col])
                        local_density[row, col] += self.local_density(person[0], cell)
                        #
            if len(set(component)) > 1:
                threshold = np.ma.masked_equal(local_density, 0).mean()
            else:
                threshold = exp(-min(self.proxemics_th, self.dict['Personal']) ** 2.0 / 2.0)
                                #
            for row in rows:
                for col in cols:
                    local_cluster[row, col] = self.set_threshold(local_density, row, col, threshold)

            self.cluster.append(local_cluster)
            self.density.append(local_density)

#falta associar peso, pontos nessa região receberão peso null (o)
    def is_point_in_field_of_vision(self, person, point_x, point_y):
        # Calcule as diferenças entre as coordenadas do ponto e a posição da pessoa
        delta_x = point_x - person.x
        delta_y = point_y - person.y

        # Calcule a distância do ponto à posição da pessoa
        distance_to_point = math.sqrt(delta_x ** 2 + delta_y ** 2)

        # Calcule o ângulo entre a direção da pessoa e o ponto em radianos
        angle_to_point = math.atan2(delta_y, delta_x)

        # Normalize o ângulo para o intervalo de -pi a pi radianos
        angle_to_point = (angle_to_point + 2 * math.pi) % (2 * math.pi)

        # Ajuste para levar em consideração a orientação da pessoa
        person_th_normalized = (person.th + math.pi) % (2 * math.pi) - math.pi
        angle_to_point = (angle_to_point - person_th_normalized + 3 * math.pi) % (2 * math.pi) - math.pi

        # Calcule os limites do campo de visão em radianos
        lower_limit = -np.deg2rad(90)
        upper_limit = np.deg2rad(90)

        # Verifique se o ângulo e a distância estão dentro do campo de visão
        if lower_limit <= angle_to_point <= upper_limit:
            return True
        else:
            return False

#falta associar peso, pontos nessa região receberão peso mediun (por ex:3)
    def is_aprroach_point(self, person, point_x, point_y):
        # Calcule as diferenças entre as coordenadas do ponto e a posição da pessoa
        delta_x = point_x - person.x
        delta_y = point_y - person.y

        # Calcule a distância do ponto à posição da pessoa
        distance_to_point = math.sqrt(delta_x ** 2 + delta_y ** 2)

        # Calcule o ângulo entre a direção da pessoa e o ponto em radianos
        angle_to_point = math.atan2(delta_y, delta_x)
        #print(f'angle_to_point:{angle_to_point}')

        # Normalize o ângulo para o intervalo de -pi a pi radianos
        angle_to_point = (angle_to_point + 2 * math.pi) % (2 * math.pi)
        #print(f'ang_normalizado:{angle_to_point}')
        # Ajuste para levar em consideração a orientação da pessoa
        person_th_normalized = (person.th + math.pi) % (2 * math.pi) - math.pi
        angle_to_point = (angle_to_point - person_th_normalized + 3 * math.pi) % (2 * math.pi) - math.pi

        # Calcule os limites do campo de visão em radianos
        lower_limit = - np.deg2rad(15)
        upper_limit =  np.deg2rad(15)
        
        if (
                math.isclose(lower_limit, angle_to_point, abs_tol=1e-6) or
                math.isclose(upper_limit, angle_to_point, abs_tol=1e-6)
                ):
            return True
        elif lower_limit < angle_to_point < upper_limit:
            return True
        else:
            return False

#falta definir pesos, pontos nessa região receberão peso max (por ex: 5)
    def is_better_aprroach_point(self, person, point_x, point_y):
        # Calcule as diferenças entre as coordenadas do ponto e a posição da pessoa
        delta_x = point_x - person.x
        delta_y = point_y - person.y

        # Calcule a distância do ponto à posição da pessoa
        distance_to_point = math.sqrt(delta_x ** 2 + delta_y ** 2)

        # Calcule o ângulo entre a direção da pessoa e o ponto em radianos
        angle_to_point = math.atan2(delta_y, delta_x)
        #print(f'angle_to_point:{angle_to_point}')

        # Normalize o ângulo para o intervalo de -pi a pi radianos
        angle_to_point = (angle_to_point + 2 * math.pi) % (2 * math.pi)
        #print(f'ang_normalizado:{angle_to_point}')
        # Ajuste para levar em consideração a orientação da pessoa
        person_th_normalized = (person.th + math.pi) % (2 * math.pi) - math.pi
        angle_to_point = (angle_to_point - person_th_normalized + 3 * math.pi) % (2 * math.pi) - math.pi

        # Calcule os limites do campo de visão em radianos
        lower_limit = - np.deg2rad(5)
        upper_limit =  np.deg2rad(5)
        
        if (
                math.isclose(lower_limit, angle_to_point, abs_tol=1e-6) or
                math.isclose(upper_limit, angle_to_point, abs_tol=1e-6)
                ):
            return True
        elif lower_limit < angle_to_point < upper_limit:
            return True
        else:
            return 
    
    #recebe uma lista de pessoas e calcula a orientação média do grupo
    def calculate_mean_orientation(self, people):
        # Inicialize as somas das componentes X e Y dos vetores
        sum_x = 0
        sum_y = 0

        # Converter os ângulos (theta) em coordenadas de vetor
        for person in people:
            angle_rad = person.th
            sum_x += np.cos(angle_rad)
            sum_y += np.sin(angle_rad)

        # Calcule o ângulo médio em relação ao norte
        mean_orientation = np.arctan2(sum_y, sum_x)

        return mean_orientation

#recebe um cluster (grupo de individuos) e calcula as coordenadas do seu centro
    def calculate_cluster_properties(self, cluster):
        x_coords = np.array(cluster[0])  # Acessar a primeira lista para as coordenadas x
        y_coords = np.array(cluster[1])  # Acessar a segunda lista para as coordenadas y

        # Combine as coordenadas em um array de pontos
        points = np.column_stack((x_coords, y_coords))

        # Calcule o envelope convexo dos pontos
        hull = ConvexHull(points)

        # Calcule a área do envelope convexo
        area = hull.volume
        #print(f'area:{area}')
        
        # Calcule o centróide (cx, cy)
        cx = np.mean(x_coords)
        cy = np.mean(y_coords)

        # Calcule o raio como a distância máxima de qualquer ponto ao centróide
        squared_distances = [(x - cx) ** 2 + (y - cy) ** 2 for x, y in zip(x_coords, y_coords)]
        radius = np.sqrt(np.max(squared_distances))

        # Calcular o raio médio (usando a distância média do centróide aos pontos do cluster)
        distances = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        radius = np.mean(distances)
        #versão anterior retornava o radius, porém não estamos usando 

        return cx, cy
    
#calcula a distância entre duas pessoas
    def calculate_distance(self, person1, person2):
        dx = person1.x - person2.x
        dy = person1.y - person2.y
        distance = math.sqrt(dx**2 + dy**2)
        return distance

#verifica se uma f-formation é aproximandamente um triângulo equilátero
    def is_equilateral_triangle(self, guard_persons, tolerance=0.2):
        if len(guard_persons) == 3:
            side1 = self.calculate_distance(guard_persons[0], guard_persons[1])
            side2 = self.calculate_distance(guard_persons[1], guard_persons[2])
            side3 = self.calculate_distance(guard_persons[2], guard_persons[0])

            max_side = max(side1, side2, side3)
            min_side = min(side1, side2, side3)

            # Verifica se a diferença entre a maior e a menor distância está dentro da tolerância
            if max_side - min_side <= tolerance:
                return True

        return False
    
    #discretiza os clusters em "regiões de abordagem" 
    def get_points_in_field_of_vision(self, people):
        #cria dicionários específicos pra armazenas pontos de interesse
        points_in_fov = {}
        approach_points = {}
        better_approach_points = {}
    
        # Criar um grafo a partir do grafo original
        G = self.G.copy()
    
        # Get a list of connected components
        connected_components = list(nx.connected_components(G))
        
        #cria uma lista pra armazenas pontos de interesse em cada dicionário pra cada pessoa na cena
        for person in people:
            points_in_fov[person] = []
            approach_points[person] = []
            better_approach_points[person] = []
        
        for cluster_idx, cluster in enumerate(connected_components):
            # Obtenha o número de pessoas no cluster atual
            num_people = len(cluster)
            #print(f'n:{num_people}')
            if num_people == 1:
                # Lógica para clusters com 1 pessoa
                person_idx = list(cluster)[0]  # Obtenha o índice da pessoa no cluster
                person = people[person_idx]  # Acesse a pessoa correspondente ao índice
            
                # Verifique cada ponto no cluster
                for row in range(self.cluster[cluster_idx].shape[0]):
                    for col in range(self.cluster[cluster_idx].shape[1]):
                        if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                            point_x = self.x[col]
                            point_y = self.y[row]
                            #print(point_x)
                            # Verifique se o ponto está no campo de visão da pessoa
                            if self.is_point_in_field_of_vision(person, point_x, point_y):
                                points_in_fov[person].append((point_x, point_y))
                                
                            # Verifique se o ponto está na região de approach da pessoa
                            if self.is_aprroach_point(person, point_x, point_y):
                                points_in_fov[person].remove((point_x, point_y))
                                approach_points[person].append((point_x, point_y))
                            
                            # Verifique se o ponto está na melhor região de approach da pessoa
                            if self.is_better_aprroach_point(person, point_x, point_y):
                                approach_points[person].remove((point_x, point_y))
                                better_approach_points[person].append((point_x, point_y))
            
            #tratar as especificidades de grupos com 2 pessoas 
            elif num_people == 2:
                X, Y = np.meshgrid(self.x, self.y)
                #calcular as coordenadas do centro do cluster
                social_zone_borders = []  # Lista para armazenar as bordas estimadas da zona social
                # Inicialize listas para armazenar as coordenadas X e Y das bordas estimadas
                border_x = []
                border_y = []
                # Preencha as listas com as coordenadas X e Y dos pontos de borda
                for row in range(self.cluster[cluster_idx].shape[0]):
                    for col in range(self.cluster[cluster_idx].shape[1]):
                        if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                            border_x.append(X[row, col])
                            border_y.append(Y[row, col])
                # Adicione as coordenadas X e Y dos pontos de borda à lista social_zone_borders
                social_zone_borders.append((border_x, border_y))
                for border_x, border_y in social_zone_borders:
                    # Calcula as coordeandas do centro do cluster 
                    xc, yc = self.calculate_cluster_properties((border_x, border_y))
                #iterar no cluster e guardar os atributos de person1 e person2
                guard_persons = []
                for person_idx in cluster:
                    person= people[person_idx]
                    guard_persons.append(person)
                #obter o angulo médio para qual todas as orientações convergem
                mean_orientation=self.calculate_mean_orientation(guard_persons)
                #print(np.rad2deg(mean_orientation))
                #criar uma 'pessoa virtual' no centro do O_space 
                person_Ocenter = Person(xc, yc, mean_orientation, 0)
                #verificar se os pontos do cluster estão dentro do fov dessa pessoa virtual
                for row in range(self.cluster[cluster_idx].shape[0]):
                    for col in range(self.cluster[cluster_idx].shape[1]):
                        if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                            point_x = self.x[col]
                            point_y = self.y[row]
                            # Verifique se o ponto está no campo de visão da pessoa
                            if self.is_point_in_field_of_vision(person_Ocenter, point_x, point_y):
                                points_in_fov[person].append((point_x, point_y))
                            # Verifique se o ponto está na região de approach da pessoa
                            if self.is_aprroach_point(person_Ocenter, point_x, point_y):
                                points_in_fov[person].remove((point_x, point_y))
                                approach_points[person].append((point_x, point_y))
                            # Verifique se o ponto está na melhor região de approach da pessoa
                            if self.is_better_aprroach_point(person_Ocenter, point_x, point_y):
                                approach_points[person].remove((point_x, point_y))
                                better_approach_points[person].append((point_x, point_y))
                #verificar se a f-formation é face a face:
                for i in range(len(guard_persons) - 1):
                    angulo1 = guard_persons[i].th
                    angulo2 = guard_persons[i + 1].th
                if angulo1 != angulo2:
                    # Calcula a diferença entre os ângulos (em módulo)
                    diferenca = abs(angulo1 - angulo2)
                    # Verifica se a diferença é igual a 180 graus
                    if diferenca == np.deg2rad(180) or diferenca == np.deg2rad(0):
                        # Calcula o ângulo simétrico respeitando o sinal
                        mean_orientation_simetrico = (mean_orientation + np.deg2rad(180)) % np.deg2rad(360)
                        if mean_orientation_simetrico > np.deg2rad(180):
                            mean_orientation_simetrico -= np.deg2rad(360)
                        person_Ocenter_simetrico = Person(xc, yc, mean_orientation_simetrico, 0)
                        #verificar se os pontos do cluster estão dentro do fov dessa pessoa virtual
                        for row in range(self.cluster[cluster_idx].shape[0]):
                            for col in range(self.cluster[cluster_idx].shape[1]):
                                if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                                    point_x = self.x[col]
                                    point_y = self.y[row]
                                    # Verifique se o ponto está no campo de visão da pessoa
                                    if self.is_point_in_field_of_vision(person_Ocenter_simetrico, point_x, point_y):
                                        points_in_fov[person].append((point_x, point_y))
                                        # Verifique se o ponto está na região de approach da pessoa
                                    if self.is_aprroach_point(person_Ocenter_simetrico, point_x, point_y):
                                        points_in_fov[person].remove((point_x, point_y))
                                        approach_points[person].append((point_x, point_y))
                                        # Verifique se o ponto está na melhor região de approach da pessoa
                                    if self.is_better_aprroach_point(person_Ocenter_simetrico, point_x, point_y):
                                        approach_points[person].remove((point_x, point_y))
                                        better_approach_points[person].append((point_x, point_y))
            elif 2 < num_people < 5:
                #calculo do xc e yc e lista pra guardas as pessoas identico pra n=2
                X, Y = np.meshgrid(self.x, self.y)
                #calcular as coordenadas do centro do cluster
                social_zone_borders = []  # Lista para armazenar as bordas estimadas da zona social
                # Inicialize listas para armazenar as coordenadas X e Y das bordas estimadas
                border_x = []
                border_y = []
                # Preencha as listas com as coordenadas X e Y dos pontos de borda
                for row in range(self.cluster[cluster_idx].shape[0]):
                    for col in range(self.cluster[cluster_idx].shape[1]):
                        if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                            border_x.append(X[row, col])
                            border_y.append(Y[row, col])
                # Adicione as coordenadas X e Y dos pontos de borda à lista social_zone_borders
                social_zone_borders.append((border_x, border_y))
                for border_x, border_y in social_zone_borders:
                    # Calcula as coordeandas do centro do cluster 
                    xc, yc = self.calculate_cluster_properties((border_x, border_y))
                #iterar no cluster e guardar os atributos de person1 e person2
                guard_persons = []
                for person_idx in cluster:
                    person= people[person_idx]
                    guard_persons.append(person)
                #caso especial triangulo equilatero
                if self.is_equilateral_triangle(guard_persons, tolerance=0.2):
                    for person in guard_persons:
                        #verificar se os pontos do cluster estão dentro do fov dessa pessoa virtual
                        for row in range(self.cluster[cluster_idx].shape[0]):
                            for col in range(self.cluster[cluster_idx].shape[1]):
                                if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                                    point_x = self.x[col]
                                    point_y = self.y[row]
                                    # Verifique se o ponto está no campo de visão da pessoa
                                    if self.is_point_in_field_of_vision(person, point_x, point_y):
                                        points_in_fov[person].append((point_x, point_y))
                                    # Verifique se o ponto está na região de approach da pessoa
                                    if self.is_aprroach_point(person, point_x, point_y):
                                        points_in_fov[person].remove((point_x, point_y))
                                        approach_points[person].append((point_x, point_y))
                                    # Verifique se o ponto está na melhor região de approach da pessoa
                                    if self.is_better_aprroach_point(person, point_x, point_y):
                                        approach_points[person].remove((point_x, point_y))
                                        better_approach_points[person].append((point_x, point_y))
                else:
                    #agora itera entre cada duas pessoas no cluster e faz as verificações para os casos específicos
                    for person1, person2 in zip(guard_persons, guard_persons[1:] + [guard_persons[0]]):
                        angulo1 = person1.th
                        angulo2 = person2.th
                        par_persons = []
                        #print(f'ang1:{np.rad2deg(angulo1)}, ang2:{np.rad2deg(angulo2)}')
                        par_persons.append(person1)
                        par_persons.append(person2)
                        #obter o angulo médio para o qual o par de pessoas converge
                        mean_orientation=self.calculate_mean_orientation(par_persons)
                        
                        if num_people == 4:
                            person_Ocenter = Person(xc, yc, mean_orientation, id_node=0)
                            #print(f'mean:{mean_orientation}')
                            #verificar se os pontos do cluster estão dentro do fov dessa pessoa virtual
                            for row in range(self.cluster[cluster_idx].shape[0]):
                                for col in range(self.cluster[cluster_idx].shape[1]):
                                    if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                                        point_x = self.x[col]
                                        point_y = self.y[row]
                                        # Verifique se o ponto está no campo de visão da pessoa
                                        if self.is_point_in_field_of_vision(person_Ocenter, point_x, point_y):
                                            points_in_fov[person].append((point_x, point_y))
                                        # Verifique se o ponto está na região de approach da pessoa
                                        if self.is_aprroach_point(person_Ocenter, point_x, point_y):
                                            points_in_fov[person].remove((point_x, point_y))
                                            approach_points[person].append((point_x, point_y))
                                        # Verifique se o ponto está na melhor região de approach da pessoa
                                        if self.is_better_aprroach_point(person_Ocenter, point_x, point_y):
                                            approach_points[person].remove((point_x, point_y))
                                            better_approach_points[person].append((point_x, point_y))
                            if mean_orientation == np.deg2rad(0) or mean_orientation == np.deg2rad(180): #caso específico que ocorre com 4 pessoas
                                mean_orientation_simetrico = (mean_orientation + np.deg2rad(180)) % np.deg2rad(360)
                                if mean_orientation_simetrico > np.deg2rad(180):
                                    mean_orientation_simetrico -= np.deg2rad(360)
                                person_simetrico = Person(xc, yc, mean_orientation_simetrico, id_node=0)
                                #verificar se os pontos do cluster estão dentro do fov dessa pessoa virtual
                                for row in range(self.cluster[cluster_idx].shape[0]):
                                    for col in range(self.cluster[cluster_idx].shape[1]):
                                        if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                                            point_x = self.x[col]
                                            point_y = self.y[row]
                                            # Verifique se o ponto está no campo de visão da pessoa
                                            if self.is_point_in_field_of_vision(person_simetrico, point_x, point_y):
                                                points_in_fov[person].append((point_x, point_y))
                                            # Verifique se o ponto está na região de approach da pessoa
                                            if self.is_aprroach_point(person_simetrico, point_x, point_y):
                                                points_in_fov[person].remove((point_x, point_y))
                                                approach_points[person].append((point_x, point_y))
                                            # Verifique se o ponto está na melhor região de approach da pessoa
                                            if self.is_better_aprroach_point(person_simetrico, point_x, point_y):
                                                approach_points[person].remove((point_x, point_y))
                                                better_approach_points[person].append((point_x, point_y))
#casos com especificidades angulares:
                        #primeiro caso pares de pessoas com mesmo angulo (triangulo isosceles)
                        elif angulo1 == angulo2:
                            #print('ang1 e ang2 iguais, deve ser um triang isosceles')
                            continue
                        elif angulo1 != angulo2:
                            # Calcula a diferença entre os ângulos (em módulo)
                            diferenca = abs(angulo1 - angulo2)
                            soma = abs(angulo1 + angulo2)
                            #verifica se a diferença é 90 graus (forma um L entre o par)
                            if diferenca == np.deg2rad(90):
                                #print('formou um L')
                                #criar uma 'pessoa virtual' no centro do O_space 
                                #print(f'mean:{abs(np.rad2deg(mean_orientation))}')
                                if (soma == np.deg2rad(180) and diferenca == np.deg2rad(90)) or (soma==np.deg2rad(90) and diferenca==np.deg2rad(180)):
                                    mean_orientation = mean_orientation
                                    #print('mantem a mean_orientation')
                                elif abs(mean_orientation)< np.deg2rad(90):
                                    mean_orientation = mean_orientation + np.deg2rad(90)
                                else:
                                    mean_orientation = mean_orientation - np.deg2rad(90)
                                    #print(f'new_mean:{np.rad2deg(mean_orientation)}')
                                person_Ocenter = Person(xc, yc, mean_orientation, 0)
                                #verificar se os pontos do cluster estão dentro do fov dessa pessoa virtual
                                for row in range(self.cluster[cluster_idx].shape[0]):
                                    for col in range(self.cluster[cluster_idx].shape[1]):
                                        if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                                            point_x = self.x[col]
                                            point_y = self.y[row]
                                            # Verifique se o ponto está no campo de visão da pessoa
                                            if self.is_point_in_field_of_vision(person_Ocenter, point_x, point_y):
                                                points_in_fov[person].append((point_x, point_y))
                                            # Verifique se o ponto está na região de approach da pessoa
                                            if self.is_aprroach_point(person_Ocenter, point_x, point_y):
                                                points_in_fov[person].remove((point_x, point_y))
                                                approach_points[person].append((point_x, point_y))
                                            # Verifique se o ponto está na melhor região de approach da pessoa
                                            if self.is_better_aprroach_point(person_Ocenter, point_x, point_y):
                                                approach_points[person].remove((point_x, point_y))
                                                better_approach_points[person].append((point_x, point_y))
                            # Verifica se a diferença é igual a 180 graus (semi-circulo)
                            if diferenca == np.deg2rad(180) or diferenca == np.deg2rad(0):
                                # Calcula o ângulo simétrico respeitando o sinal
                                #print('par de angulos simetricos, deve ser um semi_circulo')
                                if angulo1 == np.deg2rad(0) and angulo2 == np.deg2rad(180) or soma == np.deg2rad(-180):
                                    mean_orientation_simetrico = np.deg2rad(-90)
                                elif angulo1 == np.deg2rad(180) and angulo2 == np.deg2rad(0) or soma == np.deg2rad(90):
                                    mean_orientation_simetrico = np.deg2rad(90)
                                else:
                                    mean_orientation_simetrico = (mean_orientation + np.deg2rad(180)) % np.deg2rad(360)
                                    if mean_orientation_simetrico > np.deg2rad(180):
                                        mean_orientation_simetrico -= np.deg2rad(360)
                                    #print(f'angulo simetrico central:{np.rad2deg(mean_orientation_simetrico)}')
                                    person_Ocenter_simetrico = Person(xc, yc, -mean_orientation_simetrico, 0)
                                    #verificar se os pontos do cluster estão dentro do fov dessa pessoa virtual
                                    for row in range(self.cluster[cluster_idx].shape[0]):
                                        for col in range(self.cluster[cluster_idx].shape[1]):
                                            if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                                                point_x = self.x[col]
                                                point_y = self.y[row]
                                                # Verifique se o ponto está no campo de visão da pessoa
                                                if self.is_point_in_field_of_vision(person_Ocenter_simetrico, point_x, point_y):
                                                    points_in_fov[person].append((point_x, point_y))
                                                # Verifique se o ponto está na região de approach da pessoa
                                                if self.is_aprroach_point(person_Ocenter_simetrico, point_x, point_y):
                                                    points_in_fov[person].remove((point_x, point_y))
                                                    approach_points[person].append((point_x, point_y))
                                                # Verifique se o ponto está na melhor região de approach da pessoa
                                                if self.is_better_aprroach_point(person_Ocenter_simetrico, point_x, point_y):
                                                    approach_points[person].remove((point_x, point_y))
                                                    better_approach_points[person].append((point_x, point_y))
                            #verifica se a soma é 45 graus 
                            elif soma == np.deg2rad(45):
                                #print('angulos somam 45 graus')
                                #usa o próprio fov de cada uma das pessoas no par
                                #verificar se os pontos do cluster estão dentro do fov dessa pessoa virtual
                                for row in range(self.cluster[cluster_idx].shape[0]):
                                    for col in range(self.cluster[cluster_idx].shape[1]):
                                        if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                                            point_x = self.x[col]
                                            point_y = self.y[row]
                                            # Verifique se o ponto está no campo de visão da pessoa
                                            if self.is_point_in_field_of_vision(person, point_x, point_y):
                                                points_in_fov[person].append((point_x, point_y))
                                            # Verifique se o ponto está na região de approach da pessoa
                                            if self.is_aprroach_point(person, point_x, point_y):
                                                points_in_fov[person].remove((point_x, point_y))
                                                approach_points[person].append((point_x, point_y))
                                            # Verifique se o ponto está na melhor região de approach da pessoa
                                            if self.is_better_aprroach_point(person, point_x, point_y):
                                                approach_points[person].remove((point_x, point_y))
                                                better_approach_points[person].append((point_x, point_y))
                            #para qualquer outro caso
                            else:
                                p#rint('caso genérico')
                                person_Ocenter = Person(xc, yc, mean_orientation)
                                #verificar se os pontos do cluster estão dentro do fov dessa pessoa virtual
                                for row in range(self.cluster[cluster_idx].shape[0]):
                                    for col in range(self.cluster[cluster_idx].shape[1]):
                                        if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                                            point_x = self.x[col]
                                            point_y = self.y[row]
                                            # Verifique se o ponto está no campo de visão da pessoa
                                            if self.is_point_in_field_of_vision(person_Ocenter, point_x, point_y):
                                                points_in_fov[person].append((point_x, point_y))
                                            # Verifique se o ponto está na região de approach da pessoa
                                            if self.is_aprroach_point(person_Ocenter, point_x, point_y):
                                                points_in_fov[person].remove((point_x, point_y))
                                                approach_points[person].append((point_x, point_y))
                                            # Verifique se o ponto está na melhor região de approach da pessoa
                                            if self.is_better_aprroach_point(person_Ocenter, point_x, point_y):
                                                approach_points[person].remove((point_x, point_y))
                                                better_approach_points[person].append((point_x, point_y))
                
            else:
            # grupo com 5 pessoas 
            # Crie uma lista temporária para armazenar os pontos que devem ser mantidos para cada pessoa
                temp_points_in_fov = {person: [] for person in people}
                temp_approach_points = {person: [] for person in people}
                temp_better_approach_points = {person: [] for person in people}
                

                for person_idx in cluster:
                    person = people[person_idx]

                    # Crie listas temporárias para armazenar os pontos para cada pessoa no cluster
                    temp_points_in_fov[person] = []
                    temp_approach_points[person] = []
                    temp_better_approach_points[person] = []

                for row in range(self.cluster[cluster_idx].shape[0]):
                    for col in range(self.cluster[cluster_idx].shape[1]):
                        if self.cluster[cluster_idx][row, col] == 1:  # Verifique se o ponto pertence ao cluster
                            point_x = self.x[col]
                            point_y = self.y[row]

                            # Verifique se o ponto está no campo de visão de cada pessoa no cluster
                            for person_idx in cluster:
                                person = people[person_idx]

                                if self.is_point_in_field_of_vision(person, point_x, point_y):
                                    points_in_fov[person].append((point_x, point_y))
                                
                                # Verifique se o ponto está na região de approach da pessoa
                                if self.is_aprroach_point(person, point_x, point_y):
                                    points_in_fov[person].remove((point_x, point_y))
                                    approach_points[person].append((point_x, point_y))
                            
                                # Verifique se o ponto está na melhor região de approach da pessoa
                                if self.is_better_aprroach_point(person, point_x, point_y):
                                    approach_points[person].remove((point_x, point_y))
                                    better_approach_points[person].append((point_x, point_y))

                # No final da iteração sobre os pontos, adicione as listas temporárias às listas finais
                for person in people:
                    points_in_fov[person].extend(temp_points_in_fov[person])
                    approach_points[person].extend(temp_points_in_fov[person])
                    better_approach_points[person].extend(temp_better_approach_points[person])
                    

        return points_in_fov, approach_points, better_approach_points


#Plotagens
    def draw_overall(self, drawDensity=False, drawCluster=False, drawGraph=False, drawSegment=False, plot=plt, people=None):

        if drawCluster:
            X, Y = np.meshgrid(self.x, self.y)
            color = np.linspace(0, 1, len(self.cluster))
            for idx, cluster in enumerate(self.cluster):
                plot.scatter(X, Y, cluster, edgecolors=plt.cm.Dark2(color[idx]))
        if (drawGraph):
            pos = nx.get_node_attributes(self.G, 'pos')
            nx.draw_networkx(self.G, pos, node_size=000, with_labels=False)
            labels = nx.get_edge_attributes(self.G, 'weight')  # (self.G, 'None')
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels, font_size=15)
            # Uncomment the line below to plot the labels
            # nx.draw_networkx_labels(self.G,pos,font_size=20,font_family='sans-serif')
        if (drawDensity):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            X, Y = np.meshgrid(self.x, self.y)
            matrix = np.zeros((X.shape[0], Y.shape[0]))
            for density in self.density:
                matrix[:, :] += density[:, :]
            Z = np.zeros((X.shape[0], Y.shape[0])) + 0.609730865463
            ax.plot_surface(X, Y, Z, color='red', linewidth=0)
            suface = ax.plot_surface(X, Y, matrix, cmap='jet', linewidth=0)
            plt.colorbar(suface)
            ax.set_xlabel('x (meters)')
            ax.set_ylabel('y (meters)')
            ax.set_zlabel('Z')
            ax.view_init(elev=3., azim=-45)
        if (drawSegment):
            #X, Y = np.meshgrid(self.x, self.y)
            # Chame a função para obter os pontos para todas as pessoas em 'people'
            segment_points_dict, approach_points_dict, better_approach_points_dict = self.get_points_in_field_of_vision(people)
            # Plote os pontos coletados na função get_points_in_field_of_vision
            for person, points in segment_points_dict.items():
                if len(points) > 0:
                    xs, ys = zip(*points)
                    plot.scatter(xs, ys, marker='o', color='red')
            for person, points in approach_points_dict.items():
                if len(points) > 0:
                    xb, yb = zip(*points)
                    plot.scatter(xb, yb, marker='o', color='yellow')
            for person, points in better_approach_points_dict.items():
                if len(points) > 0:
                    xb, yb = zip(*points)
                    plot.scatter(xb, yb, marker='o', color='green')

        plt.axis([self.x[0], self.x[len(self.x) - 1], self.y[0], self.y[len(self.y) - 1]])
        plot.axis('equal')


#herdei essa função do Alan não sei o que ela faz
    def precision_recall(self, f):
        #
        est = []
        graph = list(nx.connected_components(self.G))
        for component in nx.connected_components(self.G):
            # remove singleton elements
            if set(component).isdisjoint(nx.isolates(self.G)):
                est.append(np.array([list(component)], dtype=np.uint8))
        return est