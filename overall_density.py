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
from src.person import Person


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

# Modifiquei a função boundary_estimate para retornar a região estimada da zona social
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
        social_zone_borders = []  # Lista para armazenar as bordas estimadas da zona social
        #
        for component in nx.connected_components(self.G):
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
            #print(f'cluster:{self.cluster}')
            self.density.append(local_density)
            #social_zone_borders.append(local_cluster)  # Adiciona a região estimada à lista
            
            # Inicialize listas para armazenar as coordenadas X e Y das bordas estimadas
            border_x = []
            border_y = []

            # Preencha as listas com as coordenadas X e Y dos pontos de borda
            for row in rows:
                for col in cols:
                    border_x.append(X[row, col])
                    border_y.append(Y[row, col])
            
            # Adicione as coordenadas X e Y dos pontos de borda à lista social_zone_borders
            social_zone_borders.append((border_x, border_y))
        #print(social_zone_borders)
        return social_zone_borders  # Retorna a lista de regiões estimadas da zona social


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

        return cx, cy, radius

    def calculate_ospace_coords(self, cluster_coords):
        num_people = len(cluster_coords)
        if num_people < 2:
            return None  # Não é possível calcular o centro com menos de duas pessoas

        # Inicialize as listas para armazenar as coordenadas das pessoas
        x_coords = []
        y_coords = []

        for x, y, _ in cluster_coords:
            x_coords.append(x)
            y_coords.append(y)

        # Calcule o ponto de interseção (centro) das coordenadas das pessoas
        ospace_center_x = np.mean(x_coords)
        ospace_center_y = np.mean(y_coords)

        # Calcule a orientação central do O-space com base nas coordenadas do centro
        ospace_angles = []
        for x, y, _ in cluster_coords:
            angle = np.arctan2(y - ospace_center_y, x - ospace_center_x)
            ospace_angles.append(angle)

        ospace_center = np.mean(ospace_angles)

        return ospace_center


    def is_not_behind_anyone(self, sx, sy):
        for person in self.person:
            # Calculate the vector from the person to the point (sx, sy)
            vector_to_point = np.array([sx - person.x, sy - person.y])

            # Calculate the angle between the vector and the person's orientation
            angle = np.arctan2(vector_to_point[1], vector_to_point[0]) - person.th

            # Normalize the angle to be in the range [-pi, pi]
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            
            # Verifique se o ângulo é aproximadamente 180 graus
            if np.abs(angle - np.pi) <= np.deg2rad(5):
                return False #(está atrás de alguém)
            #Verifica se os angulos são complementares: (com uma margem de +- 5 graus)
            elif np.abs(angle - np.pi/2) <= np.deg2rad(5):
                return False

        return True

    def get_target_samples(self):
        target_samples = []

        # Obtém as bordas da zona social usando boundary_estimate
        social_zone_borders = self.boundary_estimate()

        for border_x, border_y in social_zone_borders:
            # Calcula o centro do cluster e o raio médio
            xc, yc, radius = self.calculate_cluster_properties((border_x, border_y))

            # Inicializa uma lista para armazenar as coordenadas das pessoas neste cluster
            cluster_coords = [(person.x, person.y, person.th) for person in self.person
                              if min(border_x) <= person.x <= max(border_x) and min(border_y) <= person.y <= max(border_y)]

            if len(cluster_coords) == 1:
                # Caso em que há apenas uma pessoa no cluster
                x1, y1, th1 = cluster_coords[0]
                sx = x1 + radius * np.cos(th1)
                sy = y1 + radius * np.sin(th1)
                target_samples.append([sx, sy])
                    
            elif len(cluster_coords) == 3:
                # Use as coordenadas do centro do cluster e a orientação da pessoa no meio do trio
                _, _, angle = cluster_coords[1]        
                sx = xc + radius * np.cos(angle)
                sy = yc + radius * np.sin(angle)
                target_samples.append([sx, sy])
                
            else:
                num_people = len(cluster_coords)
                for i in range(num_people):
                    x1, y1, th1 = cluster_coords[i]
                    x2, y2, th2 = cluster_coords[(i + 1) % num_people]
                    angle = np.arctan2(y2 - yc, x2 - xc) - np.arctan2(y1 - yc, x1 - xc)
                    angle = (angle + np.pi) % (2 * np.pi) - np.pi

                    if num_people == 2:
                        if x1 == x2:
                            mid_x = x1 + radius
                            mid_x_ = x1 - radius
                            mid_y = (y1 + y2) / 2
                            target_samples.extend([[mid_x, mid_y], [mid_x_, mid_y]])
                            if y1 != y2:
                                if th1 == np.deg2rad(0):
                                    target_samples.remove([mid_x_, mid_y])
                                else:
                                    target_samples.remove([mid_x, mid_y])
                        elif y1 == y2:
                            mid_x = (x1 + x2) / 2
                            mid_y = y1 + radius
                            mid_y_ = y1 - radius
                            target_samples.extend([[mid_x, mid_y], [mid_x, mid_y_]])
                            if x1 != x2:
                                if th1 == np.deg2rad(90):
                                    target_samples.remove([mid_x, mid_y_])
                                else:
                                    target_samples.remove([mid_x, mid_y])
                        else:
                            mid_x = (x1 + x2) / 2
                            mid_y = (y1 + y2) / 2
                            thc = -(th1 + th2) / 2 - np.deg2rad(90)
                            sx = mid_x + radius * np.cos(thc)
                            sy = mid_y + radius * np.sin(thc)
                            target_samples.append([sx, sy])

                    if num_people == 4:
                        if x1 == x2:
                            continue
                        else:
                            sx1 = x1 + radius * np.cos(th1)
                            sy1 = y1 + radius * np.sin(th1)
                            sx2 = x2 + radius * np.cos(th2)
                            sy2 = y2 + radius * np.sin(th2)
                            mid_x = (sx1 + sx2) / 2
                            mid_y = (sy1 + sy2) / 2
                            dx = mid_x - xc
                            dy = mid_y - yc
                            norm = np.sqrt(dx ** 2 + dy ** 2)
                            if norm != 0:
                                dx = dx * radius / norm
                                dy = dy * radius / norm
                            mx = xc + dx
                            my = yc + dy
                            target_samples.append([mx, my])
                            if (y1 - 0.5 <= my <= y1 + 0.5) or (y2 - 0.5 <= my <= y2 + 0.5):
                                target_samples.remove([mx, my])

                    if num_people == 5:
                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2
                        dx = mid_x - xc
                        dy = mid_y - yc
                        norm = np.sqrt(dx ** 2 + dy ** 2)
                        if norm != 0:
                            dx = dx * radius / norm
                            dy = dy * radius / norm
                            mx = xc + dx
                            my = yc + dy
                            target_samples.append([mx, my])
            
        return target_samples

    #modificada para adicionar tb a plotagem dos samples
    #def draw_overall(self, drawDensity=False, drawCluster=False, drawGraph=False, drawSamples=False, plot=plt):
    def draw_overall(self, drawDensity=False, drawCluster=False, drawGraph=False, drawSamples=False, plot=plt, social_zone_borders=None):

        if drawCluster:
            X, Y = np.meshgrid(self.x, self.y)
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
            
        if (drawSamples):
            target_samples = self.get_target_samples()
            # Convert to numpy array for indexing
            target_samples = np.array(target_samples)
            # Scatter plot the target samples
            plot.scatter([point[0] for point in target_samples], [point[1] for point in target_samples], color='blue', marker='x', label='Amostras')

            
        #ax.view_init(elev=3., azim=-45)
        plt.axis([self.x[0], self.x[len(self.x) - 1], self.y[0], self.y[len(self.y) - 1]])
        plot.axis('equal')

    def precision_recall(self, f):
        #
        est = []
        graph = list(nx.connected_components(self.G))
        for component in nx.connected_components(self.G):
            # remove singleton elements
            if set(component).isdisjoint(nx.isolates(self.G)):
                est.append(np.array([list(component)], dtype=np.uint8))
        return est