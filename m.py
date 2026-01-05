import pygame
import math
import random
import numpy as np
from typing import List, Tuple, Optional
import json
import os

# NEAT Implementation
class Gene:
    def __init__(self, input_node: int, output_node: int, weight: float, enabled: bool = True, innovation: int = 0):
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

class Node:
    def __init__(self, id: int, node_type: str, activation: float = 0.0):
        self.id = id
        self.type = node_type  # 'input', 'hidden', 'output'
        self.activation = activation
        self.sum_inputs = 0.0
        self.activated = False

class Genome:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.genes = []
        self.nodes = {}
        self.fitness = 0.0
        
        # Initialize input and output nodes
        for i in range(input_size):
            self.nodes[i] = Node(i, 'input')
        for i in range(output_size):
            self.nodes[input_size + i] = Node(input_size + i, 'output')
    
    def add_gene(self, gene: Gene):
        self.genes.append(gene)
    
    def mutate_weights(self, mutation_rate: float = 0.8, perturbation_rate: float = 0.9):
        for gene in self.genes:
            if random.random() < mutation_rate:
                if random.random() < perturbation_rate:
                    gene.weight += random.gauss(0, 0.1)
                else:
                    gene.weight = random.gauss(0, 1)
    
    def add_node_mutation(self, innovation_counter: int):
        if not self.genes:
            return innovation_counter
        
        gene = random.choice([g for g in self.genes if g.enabled])
        gene.enabled = False
        
        # Add new node
        new_node_id = max(self.nodes.keys()) + 1
        self.nodes[new_node_id] = Node(new_node_id, 'hidden')
        
        # Add new connections
        gene1 = Gene(gene.input_node, new_node_id, 1.0, True, innovation_counter)
        gene2 = Gene(new_node_id, gene.output_node, gene.weight, True, innovation_counter + 1)
        
        self.genes.append(gene1)
        self.genes.append(gene2)
        
        return innovation_counter + 2
    
    def add_connection_mutation(self, innovation_counter: int):
        input_nodes = [n for n in self.nodes.values() if n.type in ['input', 'hidden']]
        output_nodes = [n for n in self.nodes.values() if n.type in ['hidden', 'output']]
        
        if not input_nodes or not output_nodes:
            return innovation_counter
        
        input_node = random.choice(input_nodes)
        output_node = random.choice(output_nodes)
        
        # Check if connection already exists
        for gene in self.genes:
            if gene.input_node == input_node.id and gene.output_node == output_node.id:
                return innovation_counter
        
        new_gene = Gene(input_node.id, output_node.id, random.gauss(0, 1), True, innovation_counter)
        self.genes.append(new_gene)
        
        return innovation_counter + 1
    
    def feed_forward(self, inputs: List[float]) -> List[float]:
        # Reset all nodes
        for node in self.nodes.values():
            node.activation = 0.0
            node.sum_inputs = 0.0
            node.activated = False
        
        # Set input values
        for i, val in enumerate(inputs):
            if i < self.input_size:
                self.nodes[i].activation = val
                self.nodes[i].activated = True
        
        # Process network
        max_iterations = 10
        for _ in range(max_iterations):
            all_activated = True
            for gene in self.genes:
                if not gene.enabled:
                    continue
                
                input_node = self.nodes[gene.input_node]
                output_node = self.nodes[gene.output_node]
                
                if input_node.activated and not output_node.activated:
                    output_node.sum_inputs += input_node.activation * gene.weight
                    all_activated = False
            
            # Activate nodes
            for node in self.nodes.values():
                if not node.activated and node.type != 'input':
                    node.activation = self.sigmoid(node.sum_inputs)
                    node.activated = True
            
            if all_activated:
                break
        
        # Return output values
        outputs = []
        for i in range(self.output_size):
            output_node = self.nodes[self.input_size + i]
            outputs.append(output_node.activation)
        
        return outputs
    
    def sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
    
    def copy(self):
        new_genome = Genome(self.input_size, self.output_size)
        new_genome.genes = [Gene(g.input_node, g.output_node, g.weight, g.enabled, g.innovation) 
                           for g in self.genes]
        new_genome.nodes = {k: Node(v.id, v.type, v.activation) for k, v in self.nodes.items()}
        return new_genome

class NEATPopulation:
    def __init__(self, size: int, input_size: int, output_size: int):
        self.size = size
        self.input_size = input_size
        self.output_size = output_size
        self.population = []
        self.generation = 0
        self.innovation_counter = 0
        self.best_genome = None
        self.best_fitness = 0
        
        # Initialize population
        for _ in range(size):
            genome = Genome(input_size, output_size)
            # Add random initial connections
            for i in range(input_size):
                for j in range(output_size):
                    if random.random() < 0.5:
                        gene = Gene(i, input_size + j, random.gauss(0, 1), True, self.innovation_counter)
                        genome.add_gene(gene)
                        self.innovation_counter += 1
            self.population.append(genome)
    
    def evolve(self, fitness_func):
        # Evaluate fitness
        for genome in self.population:
            genome.fitness = fitness_func(genome)
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome.copy()
        
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Create new population
        new_population = []
        elite_count = int(self.size * 0.1)
        
        # Keep elite
        for i in range(elite_count):
            new_population.append(self.population[i].copy())
        
        # Fill rest with offspring
        while len(new_population) < self.size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def tournament_selection(self, tournament_size: int = 5):
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness)
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        child = Genome(self.input_size, self.output_size)
        
        # Combine genes
        p1_genes = {g.innovation: g for g in parent1.genes}
        p2_genes = {g.innovation: g for g in parent2.genes}
        
        all_innovations = set(p1_genes.keys()) | set(p2_genes.keys())
        
        for innovation in all_innovations:
            if innovation in p1_genes and innovation in p2_genes:
                # Matching genes - randomly choose from either parent
                gene = random.choice([p1_genes[innovation], p2_genes[innovation]])
                child.add_gene(Gene(gene.input_node, gene.output_node, gene.weight, gene.enabled, gene.innovation))
            elif innovation in p1_genes and parent1.fitness >= parent2.fitness:
                # Disjoint/excess from fitter parent
                gene = p1_genes[innovation]
                child.add_gene(Gene(gene.input_node, gene.output_node, gene.weight, gene.enabled, gene.innovation))
            elif innovation in p2_genes and parent2.fitness >= parent1.fitness:
                gene = p2_genes[innovation]
                child.add_gene(Gene(gene.input_node, gene.output_node, gene.weight, gene.enabled, gene.innovation))
        
        # Add nodes from parents
        all_nodes = set(parent1.nodes.keys()) | set(parent2.nodes.keys())
        for node_id in all_nodes:
            if node_id in parent1.nodes:
                node = parent1.nodes[node_id]
                child.nodes[node_id] = Node(node.id, node.type, node.activation)
            elif node_id in parent2.nodes:
                node = parent2.nodes[node_id]
                child.nodes[node_id] = Node(node.id, node.type, node.activation)
        
        return child
    
    def mutate(self, genome: Genome):
        # Weight mutation
        if random.random() < 0.6:
            genome.mutate_weights()
        
        # Add node mutation
        if random.random() < 0.05:
            self.innovation_counter = genome.add_node_mutation(self.innovation_counter)
        
        # Add connection mutation
        if random.random() < 0.1:
            self.innovation_counter = genome.add_connection_mutation(self.innovation_counter)

# Car and Track classes
import math
from typing import List, Tuple, Optional

import math
import colorsys
from typing import List, Tuple

class Car:
    def __init__(self, x: float, y: float, angle: float = 0):
        # Position and orientation
        self.x = x
        self.y = y
        self.angle = angle
        self.start_x = x
        self.start_y = y
        
        # Movement physics (enhanced for better speed control)
        self.speed = 0
        self.max_forward_speed = 8
        self.max_reverse_speed = 3
        self.acceleration = 0.3
        self.braking = 0.5
        self.friction = 0.92
        self.turn_rate = 0.06
        
        # State tracking
        self.alive = True
        self.collision_count = 0
        self.distance_traveled = 0
        self.time_elapsed = 0
        self.steering_history = []
        self.speed_history = []
        
        # Visual properties
        self.base_color = (255, 100, 100)
        self.current_color = self.base_color
        
        # Enhanced sensors
        self.sensor_angles = [-60, -30, -15, 0, 15, 30, 60]  # Wider coverage
        self.sensor_range = 200  # Longer range
        self.sensor_readings = [0] * len(self.sensor_angles)
        
        # Physical dimensions
        self.width = 20
        self.height = 12  # Slightly longer for stability
        
        # Genetic algorithm
        self.genome = None
        self.fitness = 0

def update(self, track_walls: List[Tuple[float, float, float, float]]):
    if not self.alive:
        return

    # Store previous position for distance calculation
    prev_x, prev_y = self.x, self.y
    self.time_elapsed += 1
    
    # Update sensors
    self.update_sensors(track_walls)
    
    # Get control inputs from neural network
    if self.genome:
        inputs = self.get_neural_inputs()
        outputs = self.genome.feed_forward(inputs)
        
        # Steering control (-1 to 1)
        steering = (outputs[0] - 0.5) * 2  
        
        # Speed control (0 to 1 maps to -reverse to forward)
        speed_control = outputs[1]
        
        # Update color based on speed control
        self.update_speed_color(speed_control)
    else:
        steering = 0
        speed_control = 0.5  # Neutral
        self.current_color = self.base_color

    # Apply physics
    self.apply_controls(steering, speed_control)
    self.update_position()
    
    # Update distance traveled
    self.distance_traveled += math.sqrt((self.x - prev_x)**2 + (self.y - prev_y)**2)
    
    # Update fitness
    self.fitness = self.calculate_fitness()
    
    # Check collisions
    if self.check_collision(track_walls):
        self.alive = False
        self.collision_count += 1
        # Final fitness update on collision
        self.fitness = self.calculate_fitness()

    def update_speed_color(self, speed_control: float):
        """Update car color based on speed control (red=reverse, blue=neutral, green=forward)"""
        # Convert speed control to hue (0.0-0.3 range)
        hue = 0.3 * speed_control
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        self.current_color = (int(r*255), int(g*255), int(b*255))

    def apply_controls(self, steering: float, speed_control: float):
        # Steering with speed-dependent sensitivity
        speed_factor = 1.2 - (abs(self.speed)/self.max_forward_speed) * 0.7
        self.angle += steering * self.turn_rate * speed_factor
        
        # Convert speed control to throttle/brake
        if speed_control < 0.4:  # Braking/Reversing
            throttle = -(0.4 - speed_control) * 2.5
        else:  # Accelerating
            throttle = (speed_control - 0.4) * 1.67
        
        # Apply throttle/braking
        if throttle > 0:
            self.speed += throttle * self.acceleration
        else:
            self.speed += throttle * self.braking
            
        # Apply speed limits and friction
        self.speed = max(-self.max_reverse_speed, min(self.max_forward_speed, self.speed))
        self.speed *= self.friction
        
        # Record history
        self.steering_history.append(steering)
        self.speed_history.append(self.speed)
        if len(self.steering_history) > 50:
            self.steering_history.pop(0)
            self.speed_history.pop(0)

    def update_position(self):
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

    def get_neural_inputs(self) -> List[float]:
        inputs = []
        
        # Sensor data
        inputs.extend(self.sensor_readings)
        
        # Current speed (normalized -1 to 1)
        speed_input = (self.speed / self.max_forward_speed if self.speed > 0 
                      else -self.speed / self.max_reverse_speed)
        inputs.append(speed_input)
        
        # Angle (normalized 0-1)
        inputs.append((self.angle % (2*math.pi)) / (2*math.pi))
        
        return inputs

    def calculate_fitness(self) -> float:
        """Reward distance traveled and proper speed control"""
        distance_score = self.distance_traveled * 2
        speed_bonus = max(0, self.speed) * 0.5  # Only reward forward speed
        time_penalty = -self.time_elapsed * 0.01
        collision_penalty = -self.collision_count * 50
        
        return max(0, distance_score + speed_bonus + time_penalty + collision_penalty)

    # [Keep all existing collision/sensor methods unchanged]
    def update_sensors(self, track_walls: List[Tuple[float, float, float, float]]):
        for i, angle_deg in enumerate(self.sensor_angles):
            angle_rad = self.angle + math.radians(angle_deg)
            min_distance = self.sensor_range
            
            for wall in track_walls:
                distance = self.raycast(angle_rad, wall)
                if distance < min_distance:
                    min_distance = distance
            
            self.sensor_readings[i] = 1 - (min_distance / self.sensor_range)

    def raycast(self, angle: float, wall: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = wall
        x3, y3 = self.x, self.y
        x4, y4 = self.x + math.cos(angle), self.y + math.sin(angle)

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return self.sensor_range

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and u >= 0:
            intersect_x = x1 + t * (x2 - x1)
            intersect_y = y1 + t * (y2 - y1)
            return math.sqrt((intersect_x - x3)**2 + (intersect_y - y3)**2)
        
        return self.sensor_range

    def check_collision(self, track_walls: List[Tuple[float, float, float, float]]) -> bool:
        corners = self.get_corners()
        for corner in corners:
            for wall in track_walls:
                if self.point_to_line_distance(corner, wall) < 3:
                    return True
        return False

    def get_corners(self) -> List[Tuple[float, float]]:
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        half_w = self.width / 2
        half_h = self.height / 2
        
        return [
            (self.x + half_w * cos_a - half_h * sin_a, self.y + half_w * sin_a + half_h * cos_a),
            (self.x - half_w * cos_a - half_h * sin_a, self.y - half_w * sin_a + half_h * cos_a),
            (self.x - half_w * cos_a + half_h * sin_a, self.y - half_w * sin_a - half_h * cos_a),
            (self.x + half_w * cos_a + half_h * sin_a, self.y + half_w * sin_a - half_h * cos_a)
        ]

    def point_to_line_distance(self, point: Tuple[float, float], line: Tuple[float, float, float, float]) -> float:
        px, py = point
        x1, y1, x2, y2 = line
        
        l2 = (x2 - x1)**2 + (y2 - y1)**2
        if l2 == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def draw(self, screen) -> None:
        """Draw car with speed-colored body and direction indicator"""
        # Create car surface
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw colored body
        pygame.draw.rect(car_surface, self.current_color, (0, 0, self.width, self.height))
        
        # Draw direction indicator line
        pygame.draw.line(car_surface, (0, 0, 0), 
                        (self.width/2, self.height/2), 
                        (self.width, self.height/2), 2)
        
        # Rotate and position
        rotated = pygame.transform.rotate(car_surface, math.degrees(-self.angle))
        rect = rotated.get_rect(center=(self.x, self.y))
        screen.blit(rotated, rect)
class Track:
    def __init__(self, track_type: str = "simple"):
        self.walls = []
        self.start_pos = (100, 300)
        self.start_angle = 0
        self.generate_track(track_type)
    
    def generate_track(self, track_type: str):
        if track_type == "simple":
            self.generate_simple_track()
        elif track_type == "curves":
            self.generate_curved_track()
        elif track_type == "complex":
            self.generate_complex_track()
        elif track_type == "hairpin":
            self.generate_hairpin_track()
        else:
            self.generate_mixed_track()
    
    def generate_simple_track(self):
        # Simple oval track
        self.walls = [
            # Outer walls
            (50, 50, 750, 50),    # Top
            (750, 50, 750, 550),  # Right
            (750, 550, 50, 550),  # Bottom
            (50, 550, 50, 50),    # Left
            
            # Inner walls
            (150, 150, 650, 150),  # Top inner
            (650, 150, 650, 450),  # Right inner
            (650, 450, 150, 450),  # Bottom inner
            (150, 450, 150, 150),  # Left inner
        ]
        self.start_pos = (100, 300)
        self.start_angle = 0
    
    def generate_curved_track(self):
        # Track with smooth curves
        self.walls = []
        
        # Generate curved walls using line segments
        center_x, center_y = 400, 300
        outer_radius = 200
        inner_radius = 100
        
        segments = 32
        for i in range(segments):
            angle1 = (i / segments) * 2 * math.pi
            angle2 = ((i + 1) / segments) * 2 * math.pi
            
            # Outer wall
            x1 = center_x + outer_radius * math.cos(angle1)
            y1 = center_y + outer_radius * math.sin(angle1)
            x2 = center_x + outer_radius * math.cos(angle2)
            y2 = center_y + outer_radius * math.sin(angle2)
            self.walls.append((x1, y1, x2, y2))
            
            # Inner wall
            x1 = center_x + inner_radius * math.cos(angle1)
            y1 = center_y + inner_radius * math.sin(angle1)
            x2 = center_x + inner_radius * math.cos(angle2)
            y2 = center_y + inner_radius * math.sin(angle2)
            self.walls.append((x1, y1, x2, y2))
        
        self.start_pos = (300, 300)
        self.start_angle = 0
    
    def generate_complex_track(self):
        # Complex track with chicanes
        self.walls = [
            # Main straight
            (50, 100, 300, 100),
            (50, 200, 300, 200),
            
            # Chicane
            (300, 100, 350, 120),
            (350, 120, 400, 100),
            (400, 100, 650, 100),
            (300, 200, 350, 180),
            (350, 180, 400, 200),
            (400, 200, 650, 200),
            
            # Turn
            (650, 100, 750, 150),
            (750, 150, 750, 450),
            (750, 450, 650, 500),
            (650, 200, 650, 500),
            
            # Back straight
            (650, 500, 50, 500),
            (50, 500, 50, 100),
        ]
        self.start_pos = (100, 150)
        self.start_angle = 0
    
    def generate_hairpin_track(self):
        # Track with hairpin turns
        self.walls = [
            # Start straight
            (50, 200, 300, 200),
            (50, 300, 300, 300),
            
            # Hairpin 1
            (300, 200, 350, 150),
            (350, 150, 400, 200),
            (400, 200, 400, 300),
            (400, 300, 350, 350),
            (350, 350, 300, 300),
            
            # Middle section
            (400, 200, 600, 200),
            (400, 300, 600, 300),
            
            # Hairpin 2
            (600, 200, 650, 150),
            (650, 150, 700, 200),
            (700, 200, 700, 300),
            (700, 300, 650, 350),
            (650, 350, 600, 300),
            
            # End section
            (700, 200, 750, 200),
            (750, 200, 750, 500),
            (750, 500, 50, 500),
            (50, 500, 50, 200),
        ]
        self.start_pos = (100, 250)
        self.start_angle = 0
    
    def generate_mixed_track(self):
        # Mixed track with various features
        self.walls = [
            # Complex shape with multiple turns
            (50, 100, 200, 100),
            (200, 100, 300, 50),
            (300, 50, 500, 100),
            (500, 100, 600, 200),
            (600, 200, 650, 350),
            (650, 350, 550, 450),
            (550, 450, 350, 500),
            (350, 500, 150, 450),
            (150, 450, 100, 350),
            (100, 350, 50, 250),
            (50, 250, 50, 100),
            
            # Inner walls
            (150, 200, 250, 180),
            (250, 180, 350, 200),
            (350, 200, 450, 250),
            (450, 250, 500, 350),
            (500, 350, 450, 400),
            (450, 400, 350, 420),
            (350, 420, 250, 400),
            (250, 400, 200, 350),
            (200, 350, 150, 300),
            (150, 300, 150, 200),
        ]
        self.start_pos = (100, 175)
        self.start_angle = 0
import threading
import pygame
import math
import pygame
import math
import random
from collections import defaultdict

# Improved Network Visualizer
class SimpleNetworkVisualizer:
    def __init__(self, screen, x, y, width, height):
        self.screen = screen
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.font = pygame.font.Font(None, 16)
        
        # Colors
        self.colors = {
            'background': (30, 30, 30),
            'border': (100, 100, 100),
            'node_input': (0, 200, 0),
            'node_hidden': (200, 200, 0),
            'node_output': (200, 0, 0),
            'connection_positive': (0, 150, 0),
            'connection_negative': (150, 0, 0),
            'text': (255, 255, 255)
        }
        
        self.current_genome = None
        self.node_positions = {}

    def update(self, genome):
        self.current_genome = genome
        self._calculate_node_positions()

    def _calculate_node_positions(self):
        self.node_positions = {}
        if not self.current_genome or not hasattr(self.current_genome, 'nodes'):
            return
            
        # Separate nodes by type
        input_nodes = []
        hidden_nodes = []
        output_nodes = []
        
        for node_id, node in self.current_genome.nodes.items():
            if hasattr(node, 'node_type'):
                if node.node_type == 'input':
                    input_nodes.append(node_id)
                elif node.node_type == 'output':
                    output_nodes.append(node_id)
                else:
                    hidden_nodes.append(node_id)
            else:
                if node_id < 8:
                    input_nodes.append(node_id)
                elif node_id < 10:
                    output_nodes.append(node_id)
                else:
                    hidden_nodes.append(node_id)
        
        # Calculate positions with proper scaling
        margin = 20
        available_width = self.width - 2 * margin
        available_height = self.height - 2 * margin
        
        num_layers = 2 if not hidden_nodes else 3
        layer_spacing = available_width / max(1, num_layers - 1)
        
        # Input nodes
        input_spacing = available_height / max(1, len(input_nodes) + 1)
        for i, node_id in enumerate(input_nodes):
            self.node_positions[node_id] = (
                self.x + margin,
                self.y + margin + (i + 1) * input_spacing
            )
        
        # Hidden nodes
        if hidden_nodes:
            hidden_spacing = available_height / max(1, len(hidden_nodes) + 1)
            for i, node_id in enumerate(hidden_nodes):
                self.node_positions[node_id] = (
                    self.x + margin + layer_spacing,
                    self.y + margin + (i + 1) * hidden_spacing
                )
        
        # Output nodes
        output_spacing = available_height / max(1, len(output_nodes) + 1)
        output_x = self.x + margin + (layer_spacing * (num_layers - 1))
        for i, node_id in enumerate(output_nodes):
            self.node_positions[node_id] = (
                output_x,
                self.y + margin + (i + 1) * output_spacing
            )

    def draw(self):
        if not self.current_genome:
            return
            
        # Draw background with border
        pygame.draw.rect(self.screen, self.colors['background'], 
                        (self.x, self.y, self.width, self.height))
        pygame.draw.rect(self.screen, self.colors['border'],
                        (self.x, self.y, self.width, self.height), 1)
        
        # Draw connections
        if hasattr(self.current_genome, 'genes'):
            for gene in self.current_genome.genes:
                if not hasattr(gene, 'enabled') or not gene.enabled:
                    continue
                
                in_node = gene.in_node
                out_node = gene.out_node
                weight = getattr(gene, 'weight', 0)
                
                if in_node in self.node_positions and out_node in self.node_positions:
                    start_pos = self.node_positions[in_node]
                    end_pos = self.node_positions[out_node]
                    
                    color = self.colors['connection_positive'] if weight > 0 else self.colors['connection_negative']
                    thickness = max(1, min(3, int(abs(weight) * 3)))
                    pygame.draw.line(self.screen, color, start_pos, end_pos, thickness)
        
        # Draw nodes
        for node_id, pos in self.node_positions.items():
            if node_id < 8:  # Inputs
                color = self.colors['node_input']
                radius = 10
            elif node_id < 10:  # Outputs
                color = self.colors['node_output']
                radius = 10
            else:  # Hidden
                color = self.colors['node_hidden']
                radius = 8
            
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), radius)
            
            text = self.font.render(str(node_id), True, self.colors['text'])
            text_rect = text.get_rect(center=(int(pos[0]), int(pos[1])))
            self.screen.blit(text, text_rect)


class Simulation:
    def __init__(self):
        pygame.init()
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("NEAT Self-Driving Car Evolution")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.colors = {
            'background': (20, 20, 20),
            'wall': (255, 255, 255),
            'car': (0, 255, 0),
            'best_car': (255, 0, 0),
            'sensor': (255, 255, 0),
            'text': (255, 255, 255),
            'ui_bg': (40, 40, 40),
            'button': (70, 70, 70),
            'button_hover': (100, 100, 100)
        }
        
        # Initialize components
        self.track = Track("simple")
        self.population = NEATPopulation(50, 8, 2)
        self.cars = []
        self.generation = 0
        self.running = True
        self.paused = False
        self.show_best_only = False
        self.show_sensors = True
        self.show_network = False
        
        # Network visualizer
        network_width = 350
        network_height = 300
        self.network_visualizer = SimpleNetworkVisualizer(
            self.screen,
            self.width - network_width - 20,
            self.height - network_height - 20,
            network_width,
            network_height
        )
        
        # Evolution tracking
        self.best_fitness = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generation_start_time = 0
        self.generation_duration = 60
        
        # UI elements
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Initialize cars
        self.reset_cars()

    def reset_cars(self):
        self.cars = []
        for genome in self.population.population:
            car = Car(self.track.start_pos[0], self.track.start_pos[1], self.track.start_angle)
            car.genome = genome
            self.cars.append(car)
        self.generation_start_time = pygame.time.get_ticks() / 1000
        self.current_best_car = None

    def update(self):
        if self.paused:
            return
        
        current_time = pygame.time.get_ticks() / 1000
        if current_time - self.generation_start_time > self.generation_duration:
            self.next_generation()
            return
        
        # Track best car
        current_best_fitness = -1
        self.current_best_car = None
        
        # Update cars
        alive_cars = 0
        for car in self.cars:
            if car.alive:
                car.update(self.track.walls)
                alive_cars += 1
                
                # Update fitness tracking
                car.current_fitness = car.calculate_fitness()
                if car.current_fitness > current_best_fitness:
                    current_best_fitness = car.current_fitness
                    self.current_best_car = car
        
        if alive_cars == 0:
            self.next_generation()

    def next_generation(self):
        # Calculate fitness for all cars
        def fitness_func(genome):
            for car in self.cars:
                if car.genome == genome:
                    return car.calculate_fitness()
            return 0
        
        # Evolve population
        self.population.evolve(fitness_func)
        
        # Update statistics
        current_fitnesses = [car.calculate_fitness() for car in self.cars if car.alive]
        if current_fitnesses:
            self.best_fitness = max(current_fitnesses)
            self.best_fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(sum(current_fitnesses) / len(current_fitnesses))
        
        # Reset for next generation
        self.generation += 1
        self.reset_cars()
        
        print(f"Generation {self.generation}: Best Fitness = {self.best_fitness:.2f}")

    def draw(self):
        self.screen.fill(self.colors['background'])
        
        # Draw track
        self.draw_track()
        
        # Draw cars
        if self.show_best_only and self.current_best_car:
            self.draw_car(self.current_best_car, self.colors['best_car'])
        else:
            for car in self.cars:
                if car.alive:
                    color = self.colors['car']
                    if car == self.current_best_car:
                        color = self.colors['best_car']
                    self.draw_car(car, color)
        
        # Draw UI
        self.draw_ui()
        
        # Draw network visualization
        if self.show_network and self.current_best_car and hasattr(self.population, 'best_genome'):
            self.network_visualizer.update(self.population.best_genome)
            self.network_visualizer.draw()
        
        pygame.display.flip()

    def draw_track(self):
        for wall in self.track.walls:
            if len(wall) >= 4:
                pygame.draw.line(self.screen, self.colors['wall'], 
                               (wall[0], wall[1]), (wall[2], wall[3]), 3)

    def draw_car(self, car, color):
        if not car.alive:
            return
        
        # Draw car body
        corners = car.get_corners()
        if corners and len(corners) >= 3:
            pygame.draw.polygon(self.screen, color, corners)
        
        # Draw front indicator
        front_x = car.x + math.cos(car.angle) * car.width / 2
        front_y = car.y + math.sin(car.angle) * car.width / 2
        pygame.draw.circle(self.screen, (255, 255, 255), (int(front_x), int(front_y)), 3)
        
        # Draw sensors
        if self.show_sensors and hasattr(car, 'sensor_readings'):
            for i, angle_offset in enumerate(car.sensor_angles):
                if i < len(car.sensor_readings):
                    sensor_angle = car.angle + math.radians(angle_offset)
                    distance = car.sensor_readings[i] * car.sensor_range
                    end_x = car.x + math.cos(sensor_angle) * distance
                    end_y = car.y + math.sin(sensor_angle) * distance
                    intensity = int(255 * car.sensor_readings[i])
                    sensor_color = (255 - intensity, intensity, 0)
                    pygame.draw.line(self.screen, sensor_color, 
                                   (int(car.x), int(car.y)), 
                                   (int(end_x), int(end_y)), 1)
                    pygame.draw.circle(self.screen, sensor_color, 
                                     (int(end_x), int(end_y)), 3)

    def draw_ui(self):
        ui_width = 350
        ui_surface = pygame.Surface((ui_width, self.height))
        ui_surface.fill(self.colors['ui_bg'])
        
        y_offset = 20
        
        # Generation info
        gen_text = self.font.render(f"Generation: {self.generation}", True, self.colors['text'])
        ui_surface.blit(gen_text, (10, y_offset))
        y_offset += 30
        
        # Time remaining
        current_time = pygame.time.get_ticks() / 1000
        time_remaining = max(0, self.generation_duration - (current_time - self.generation_start_time))
        time_color = (255, 255, 255) if time_remaining > 10 else (255, 100, 100)
        time_text = self.font.render(f"Time Left: {time_remaining:.1f}s", True, time_color)
        ui_surface.blit(time_text, (10, y_offset))
        y_offset += 30
        
        # Cars alive
        alive_count = sum(1 for car in self.cars if car.alive)
        alive_text = self.font.render(f"Cars Alive: {alive_count}/{len(self.cars)}", True, self.colors['text'])
        ui_surface.blit(alive_text, (10, y_offset))
        y_offset += 30
        
        # Best fitness (all-time)
        best_text = self.font.render(f"Best Fitness: {self.best_fitness:.2f}", True, self.colors['text'])
        ui_surface.blit(best_text, (10, y_offset))
        y_offset += 30
        
        # Current best car fitness (this generation)
        if self.current_best_car:
            current_text = self.font.render(f"Current Best: {self.current_best_car.current_fitness:.2f}", True, self.colors['text'])
            ui_surface.blit(current_text, (10, y_offset))
            y_offset += 30
        # Population stats
        if hasattr(self.population, 'best_genome') and self.population.best_genome:
            nodes_count = len(self.population.best_genome.nodes)
            genes_count = len([g for g in getattr(self.population.best_genome, 'genes', []) if g.enabled])
            stats_text = self.font.render(f"Nodes: {nodes_count}  Genes: {genes_count}", True, self.colors['text'])
            ui_surface.blit(stats_text, (10, y_offset))
            y_offset += 30
        
        # Network visualization status
        network_status = self.font.render(
            "Network View: " + ("ON" if self.show_network else "OFF"), 
            True, 
            (0, 255, 0) if self.show_network else (255, 0, 0)
        )
        ui_surface.blit(network_status, (10, y_offset))
        y_offset += 30
        
        # Control instructions
        y_offset += 10
        controls_text = self.font.render("Controls:", True, self.colors['text'])
        ui_surface.blit(controls_text, (10, y_offset))
        y_offset += 25
        
        control_instructions = [
            "SPACE: Pause/Resume",
            "B: Toggle Best Only",
            "S: Toggle Sensors",
            "N: Toggle Network",
            "R: Reset Population",
            "1-5: Change Track",
            "ESC: Quit"
        ]
        
        for instruction in control_instructions:
            inst_text = self.small_font.render(instruction, True, self.colors['text'])
            ui_surface.blit(inst_text, (20, y_offset))
            y_offset += 18
        
        # Fitness history graph
        if len(self.best_fitness_history) > 1:
            self.draw_fitness_graph(ui_surface, y_offset)
        
        # Blit UI to screen
        self.screen.blit(ui_surface, (self.width - ui_width - 10, 10))

    def draw_fitness_graph(self, surface, y_start):
        graph_rect = pygame.Rect(10, y_start, 330, 150)
        pygame.draw.rect(surface, (60, 60, 60), graph_rect)
        pygame.draw.rect(surface, self.colors['text'], graph_rect, 2)
        
        if len(self.best_fitness_history) < 2:
            return
        
        max_fitness = max(self.best_fitness_history) if self.best_fitness_history else 1
        min_fitness = min(self.best_fitness_history) if self.best_fitness_history else 0
        fitness_range = max_fitness - min_fitness if max_fitness > min_fitness else 1
        
        # Draw best fitness line
        points = []
        for i, fitness in enumerate(self.best_fitness_history):
            x = graph_rect.x + (i / max(1, len(self.best_fitness_history) - 1)) * graph_rect.width
            y = graph_rect.y + graph_rect.height - ((fitness - min_fitness) / fitness_range) * graph_rect.height
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(surface, (255, 0, 0), False, points, 2)
        
        # Draw average fitness line
        if self.avg_fitness_history:
            avg_points = []
            for i, fitness in enumerate(self.avg_fitness_history):
                x = graph_rect.x + (i / max(1, len(self.avg_fitness_history) - 1)) * graph_rect.width
                y = graph_rect.y + graph_rect.height - ((fitness - min_fitness) / fitness_range) * graph_rect.height
                avg_points.append((x, y))
            
            if len(avg_points) > 1:
                pygame.draw.lines(surface, (0, 255, 0), False, avg_points, 2)
        
        # Labels
        title_text = self.small_font.render("Fitness History", True, self.colors['text'])
        surface.blit(title_text, (graph_rect.x + 5, graph_rect.y - 20))
        
        best_label = self.small_font.render("Best", True, (255, 0, 0))
        surface.blit(best_label, (graph_rect.x + 5, graph_rect.y + 5))
        
        avg_label = self.small_font.render("Average", True, (0, 255, 0))
        surface.blit(avg_label, (graph_rect.x + 50, graph_rect.y + 5))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_b:
                    self.show_best_only = not self.show_best_only
                elif event.key == pygame.K_s:
                    self.show_sensors = not self.show_sensors
                elif event.key == pygame.K_n:
                    self.show_network = not self.show_network
                elif event.key == pygame.K_r:
                    self.reset_simulation()
                elif event.key == pygame.K_1:
                    self.change_track("simple")
                elif event.key == pygame.K_2:
                    self.change_track("curves")
                elif event.key == pygame.K_3:
                    self.change_track("complex")
                elif event.key == pygame.K_4:
                    self.change_track("hairpin")
                elif event.key == pygame.K_5:
                    self.change_track("mixed")

    def reset_simulation(self):
        self.population = NEATPopulation(50, 8, 2)
        self.generation = 0
        self.best_fitness = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.reset_cars()

    def change_track(self, track_type):
        self.track = Track(track_type)
        self.reset_cars()

    def run(self):
        print("NEAT Self-Driving Car Simulation")
        print("Controls:")
        print("SPACE: Pause/Resume")
        print("B: Toggle Best Car Only")
        print("S: Toggle Sensors")
        print("N: Toggle Network View")
        print("R: Reset Population")
        print("1-5: Change Track")
        print("ESC: Quit")
        
        self.generation_start_time = pygame.time.get_ticks() / 1000
        
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
def main():
    simulation = Simulation()
    simulation.run()

if __name__ == "__main__":
    main()