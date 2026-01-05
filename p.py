import pygame
import math
import os
import neat
import random
import numpy as np
import time
import pickle
from pygame import gfxdraw
from collections import deque
from neat import stagnation 
from statistics import mean, stdev
from neat.stagnation import DefaultStagnation
from race_monitor import RaceMonitor
race_monitor = RaceMonitor()

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
FPS = 60
TRACK_COLOR = (255, 255, 255)
BORDER_COLOR = (0, 0, 0)
INNER_BORDER_COLOR = (200, 200, 200)
CAR_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
SENSOR_COLOR = (0, 255, 0, 100)
BASE_MAX_SPEED = 3
SENSOR_ANGLES = [-60, -30, -15, 0, 15, 30, 60]
SENSOR_LENGTH = 100
GENERATIONS = 300
GENERATION_TIME_LIMIT = 60

# Track Design
TRACK_OUTER = [(100, 150), (400, 120), (700, 200), (750, 350), 
               (700, 500), (400, 550), (100, 500), (50, 350)]
TRACK_INNER = [(200, 250), (400, 230), (600, 280), (620, 350),
               (600, 420), (400, 440), (200, 420), (180, 350)]

# Checkpoint System
CHECKPOINTS = [
    {"id": 0, "line": [(750, 350), (620, 350)], "reward": 100},
    {"id": 1, "line": [(400, 120), (400, 230)], "reward": 100},
    {"id": 2, "line": [(50, 350), (180, 350)], "reward": 100},
    {"id": 3, "line": [(400, 550), (400, 440)], "reward": 500}
]

# UI Configuration
UI_FONT = "Arial"
UI_TITLE_SIZE = 24
UI_TEXT_SIZE = 18
UI_SMALL_TEXT_SIZE = 14
UI_PANEL_COLOR = (240, 240, 240)
UI_PANEL_BORDER = (200, 200, 200)
UI_TEXT_COLOR = (0, 0, 0)
UI_HIGHLIGHT_COLOR = (0, 100, 200)
UI_WARNING_COLOR = (200, 100, 0)
UI_ERROR_COLOR = (200, 0, 0)

# Global variables
show_sensors = False
show_network = False
manual_generation_advance = False
pop = None
generation_stats = []
best_genome_history = []
global_best_genome = None
global_best_fitness = -float('inf')
clock = pygame.time.Clock()
running = True  
generation_count = 0  

# Initialize screen globally
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("NEAT Racing Simulation")

# =============================================
# UI FUNCTIONS (Grouped together for easy modification)
# =============================================
def draw_stats_panel(screen, car, x, y, width=300, height=200):
    """Draw real-time statistics for a specific car"""
    if car is None or not hasattr(car, 'laps_completed'):
        return
    
    # Create semi-transparent panel
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 180))  # Black with transparency
    
    # Draw border
    pygame.draw.rect(panel, (100, 100, 255), (0, 0, width, height), 2)
    
    font = pygame.font.SysFont(UI_FONT, UI_SMALL_TEXT_SIZE)
    small_font = pygame.font.SysFont(UI_FONT, UI_SMALL_TEXT_SIZE-2)
    
    # Basic stats
    stats = [
        f"Lap: {car.laps_completed}",
        f"Checkpoint: {car.last_checkpoint + 1}/{len(CHECKPOINTS)}",
        f"Speed: {car.speed:.1f}/{car.max_speed:.1f}",
        f"Fitness: {car.fitness:.1f}",
        f"Collisions: {car.collisions}",
        f"Distance: {car.distance_traveled:.1f}"
    ]
    
    # Draw each stat line
    for i, stat in enumerate(stats):
        text = font.render(stat, True, (255, 255, 255))
        panel.blit(text, (10, 10 + i*25))
    
    # Draw mini lap times if available
    if hasattr(car, 'checkpoint_times') and car.checkpoint_times:
        lap_header = small_font.render("Last Lap:", True, (200, 200, 255))
        panel.blit(lap_header, (10, 140))
        
        for i, time in enumerate(car.checkpoint_times[-4:], 1):
            time_text = small_font.render(f"CP{i}: {time:.2f}s", True, (255, 255, 255))
            panel.blit(time_text, (20, 140 + i*18))
    
    screen.blit(panel, (x, y))
def draw_track(screen):
    pygame.draw.polygon(screen, (100, 100, 100), TRACK_OUTER)
    pygame.draw.polygon(screen, (50, 50, 50), TRACK_INNER)
    pygame.draw.polygon(screen, BORDER_COLOR, TRACK_OUTER, 3)
    pygame.draw.polygon(screen, INNER_BORDER_COLOR, TRACK_INNER, 3)
    
    for checkpoint in CHECKPOINTS:
        color = (255, 0, 0) if checkpoint["id"] == 3 else (0, 0, 255)
        pygame.draw.line(screen, color, checkpoint["line"][0], checkpoint["line"][1], 3)

def draw_timer(screen, time_left):
    font = pygame.font.SysFont(UI_FONT, UI_TEXT_SIZE)
    minutes = int(time_left // 60)
    seconds = int(time_left % 60)
    timer_text = f"Time: {minutes:02d}:{seconds:02d}"
    
    if time_left > 20:
        color = (0, 255, 0)
    elif time_left > 10:
        color = (255, 255, 0)
    else:
        color = (255, 0, 0)
    
    text_surface = font.render(timer_text, True, color)
    screen.blit(text_surface, (10, 10))

def draw_fitness_graph(screen, x, y, width, height, stats):
    if len(stats) < 2:
        return
    
    graph_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    
    # Axes
    pygame.draw.line(graph_surface, UI_TEXT_COLOR, (0, height-20), (width, height-20), 2)
    pygame.draw.line(graph_surface, UI_TEXT_COLOR, (30, 0), (30, height-20), 2)
    
    # Calculate scaling
    max_fitness = max(s['best_fitness'] for s in stats)
    min_fitness = min(s['best_fitness'] for s in stats)
    range_fitness = max(1, max_fitness - min_fitness)
    
    x_scale = (width - 50) / len(stats)
    y_scale = (height - 40) / range_fitness
    
    # Draw grid and labels
    font = pygame.font.SysFont(UI_FONT, UI_SMALL_TEXT_SIZE)
    for i in range(0, len(stats), max(1, len(stats)//5)):
        gen_x = 30 + i * x_scale
        pygame.draw.line(graph_surface, (100, 100, 100), (gen_x, height-20), (gen_x, height-15), 1)
        text = font.render(str(i+1), True, UI_TEXT_COLOR)
        graph_surface.blit(text, (gen_x - 5, height-15))
    
    for f in np.linspace(min_fitness, max_fitness, 5):
        y_pos = height - 20 - (f - min_fitness) * y_scale
        pygame.draw.line(graph_surface, (100, 100, 100), (25, y_pos), (30, y_pos), 1)
        text = font.render(f"{f:.0f}", True, UI_TEXT_COLOR)
        graph_surface.blit(text, (5, y_pos - 8))
    
    # Draw data lines
    prev_x, prev_y = None, None
    for i, stat in enumerate(stats):
        x_pos = 30 + i * x_scale
        y_pos = height - 20 - (stat['best_fitness'] - min_fitness) * y_scale
        
        # Best fitness (blue)
        pygame.draw.circle(graph_surface, (0, 0, 255), (int(x_pos), int(y_pos)), 3)
        
        # Average fitness (green)
        avg_y = height - 20 - (stat['avg_fitness'] - min_fitness) * y_scale
        pygame.draw.circle(graph_surface, (0, 255, 0), (int(x_pos), int(avg_y)), 3)
        
        if prev_x is not None:
            # Connect best fitness points
            pygame.draw.line(graph_surface, (0, 0, 255), (prev_x, prev_y), (x_pos, y_pos), 2)
            # Connect average fitness points
            pygame.draw.line(graph_surface, (0, 255, 0), 
                           (prev_x, height - 20 - (stats[i-1]['avg_fitness'] - min_fitness) * y_scale),
                           (x_pos, avg_y), 2)
        
        prev_x, prev_y = x_pos, y_pos
    
    # Legend
    legend_font = pygame.font.SysFont(UI_FONT, UI_SMALL_TEXT_SIZE)
    best_text = legend_font.render("Best Fitness", True, (0, 0, 255))
    avg_text = legend_font.render("Avg Fitness", True, (0, 255, 0))
    graph_surface.blit(best_text, (width - 150, 10))
    graph_surface.blit(avg_text, (width - 150, 30))
    
    screen.blit(graph_surface, (x, y))

def draw_generation_panel(screen, x, y, width, height, current_gen, stats):
    panel = pygame.Surface((width, height))
    panel.fill(UI_PANEL_COLOR)
    pygame.draw.rect(panel, UI_PANEL_BORDER, (0, 0, width, height), 2)
    
    font = pygame.font.SysFont(UI_FONT, UI_TEXT_SIZE)
    small_font = pygame.font.SysFont(UI_FONT, UI_SMALL_TEXT_SIZE)
    
    # Title
    title = font.render(f"Generation {current_gen + 1} Summary", True, UI_TEXT_COLOR)
    panel.blit(title, (10, 10))
    
    # Current generation stats
    if stats and len(stats) > current_gen:
        gen_stat = stats[current_gen]
        lines = [
            f"Best Fitness: {gen_stat['best_fitness']:.1f}",
            f"Avg Fitness: {gen_stat['avg_fitness']:.1f}",
            f"Max Speed: {gen_stat['max_speed']:.1f}",
            f"Alive Cars: {gen_stat['alive_cars']}/{gen_stat['total_cars']}",
            f"Laps Completed: {gen_stat['laps_completed']}",
            f"Best Checkpoint: {gen_stat['best_checkpoint'] + 1}"
        ]
        
        for i, line in enumerate(lines):
            text = small_font.render(line, True, UI_TEXT_COLOR)
            panel.blit(text, (20, 50 + i * 25))
    
    # All-time best stats
    if stats:
        all_time_best = max(stats, key=lambda x: x['best_fitness'])
        best_text = font.render("All-Time Best:", True, UI_TEXT_COLOR)
        panel.blit(best_text, (10, height - 100))
        
        lines = [
            f"Gen {all_time_best['generation'] + 1}: {all_time_best['best_fitness']:.1f}",
            f"Speed: {all_time_best['max_speed']:.1f}",
            f"Laps: {all_time_best['laps_completed']}"
        ]
        
        for i, line in enumerate(lines):
            text = small_font.render(line, True, UI_TEXT_COLOR)
            panel.blit(text, (20, height - 70 + i * 20))
    
    screen.blit(panel, (x, y))

def draw_neural_network(screen, genome, config, x, y, width, height):
    """Draw a visualization of the neural network"""
    if not show_network:
        return
    
    # Network visualization parameters
    node_radius = 20
    node_coords = {}
    layers = {}
    font = pygame.font.SysFont(UI_FONT, UI_SMALL_TEXT_SIZE)
    
    # Organize nodes by layer
    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys
    all_nodes = set(input_keys) | set(output_keys) | set(genome.nodes.keys())
    
    for node in all_nodes:
        if node in input_keys:
            layers.setdefault(0, []).append(node)
        elif node in output_keys:
            layers.setdefault(999, []).append(node)
        else:
            layers.setdefault(1, []).append(node)
    
    # Sort layers and remap indices
    sorted_layers = sorted(layers.keys())
    layer_mapping = {old: new for new, old in enumerate(sorted_layers)}
    new_layers = {}
    for old_layer, nodes in layers.items():
        new_layers[layer_mapping[old_layer]] = nodes
    layers = new_layers
    
    # Calculate layout
    max_layer = max(layers.keys()) if layers else 0
    min_layer_spacing = 150
    min_node_spacing = 50
    
    total_width = max(width, min_layer_spacing * max(1, max_layer))
    total_height = max(height, min_node_spacing * max(len(nodes) for nodes in layers.values()))
    
    # Position nodes
    for layer_idx in sorted(layers.keys()):
        layer = layers[layer_idx]
        n = len(layer)
        layer_x = x + (layer_idx * total_width / max(1, max_layer)) if max_layer > 0 else x + total_width/2
        
        for i, node in enumerate(layer):
            py = y + (i + 1) * total_height / (n + 1)
            node_coords[node] = (int(layer_x), int(py))
    
    # Draw connections
    for conn_key, conn in genome.connections.items():
        if not conn.enabled:
            continue
        
        in_node, out_node = conn_key
        weight = conn.weight
        
        if abs(weight) > 0.01:
            if weight > 0:
                r = min(255, 50 + int(weight * 100))
                g = min(255, 200 + int(weight * 55))
                b = 50
            else:
                r = min(255, 200 + int(abs(weight) * 55))
                g = min(255, 50 + int(abs(weight) * 100))
                b = 50
                
            base_color = (r, g, b)
            thickness = max(1, min(8, int(abs(weight) * 4)))
            
            if in_node in node_coords and out_node in node_coords:
                start_pos = node_coords[in_node]
                end_pos = node_coords[out_node]
                
                pygame.draw.aaline(screen, base_color, start_pos, end_pos, True)
                pygame.draw.line(screen, base_color, start_pos, end_pos, thickness)
    
    # Draw nodes
    for node, (nx, ny) in node_coords.items():
        if node in input_keys:
            base_color = (100, 100, 255)  # Blue for inputs
            label = f"I{input_keys.index(node)}"
            sensor_names = ["Left-60°", "Left-30°", "Left-15°", "Front", 
                          "Right-15°", "Right-30°", "Right-60°", "Speed", "Angle"]
            if input_keys.index(node) < len(sensor_names):
                label = sensor_names[input_keys.index(node)]
        elif node in output_keys:
            base_color = (255, 100, 100)  # Red for outputs
            label = ["Steering", "Throttle"][output_keys.index(node)]
        else:
            base_color = (100, 255, 100)  # Green for hidden
            label = f"H{node}"
        
        pygame.draw.circle(screen, base_color, (nx, ny), node_radius)
        pygame.draw.circle(screen, (255, 255, 255), (nx, ny), node_radius, 2)
        
        # Draw label
        label_lines = label.split('\n')
        for i, line in enumerate(label_lines):
            text = font.render(line, True, (0, 0, 0))
            text_rect = text.get_rect(center=(nx, ny - (len(label_lines)//2 - i) * 10))
            screen.blit(text, text_rect)
        
        # Draw fitness meter for output nodes
        if node in output_keys and hasattr(genome, 'fitness'):
            fitness_norm = min(1.0, genome.fitness / 5000.0)
            meter_radius = node_radius + 8
            meter_width = 4
            
            pygame.draw.circle(screen, (200, 200, 200), (nx, ny), meter_radius, meter_width)
            
            meter_color = (
                int(255 * (1 - fitness_norm)),
                int(255 * fitness_norm),
                0
            )
            
            start_angle = -90
            end_angle = start_angle + 360 * fitness_norm
            
            rect = pygame.Rect(
                nx - meter_radius,
                ny - meter_radius,
                meter_radius * 2,
                meter_radius * 2
            )
            pygame.draw.arc(
                screen, meter_color, rect,
                math.radians(start_angle), math.radians(end_angle),
                meter_width
            )

def draw_realtime_network(screen, car, genome, config):
    """Draw real-time neural network visualization showing current activations"""
    if not show_network or not hasattr(car, 'network_state'):
        return
    
    # Network visualization area
    net_x, net_y = screen.get_width() - 380, 120
    net_width, net_height = 360, 280
    
    # Create semi-transparent background
    network_surface = pygame.Surface((net_width, net_height), pygame.SRCALPHA)
    network_surface.fill((0, 0, 0, 200))
    pygame.draw.rect(network_surface, (100, 100, 100), (0, 0, net_width, net_height), 2)
    
    # Add car identification header
    header_font = pygame.font.SysFont(UI_FONT, UI_TEXT_SIZE)
    car_id_text = header_font.render(f"Car {id(car) % 1000:03d}", True, car.color)
    network_surface.blit(car_id_text, (10, 10))
    
    # Add performance stats
    stats_font = pygame.font.SysFont(UI_FONT, UI_SMALL_TEXT_SIZE)
    stats_text = [
        f"Fitness: {car.fitness:.1f}",
        f"Lap: {car.laps_completed}",
        f"Checkpoint: {car.last_checkpoint + 1}/{len(CHECKPOINTS)}",
        f"Speed: {car.speed:.1f}/{car.max_speed:.1f}"
    ]
    
    for i, text in enumerate(stats_text):
        text_surface = stats_font.render(text, True, (255, 255, 255))
        network_surface.blit(text_surface, (net_width - 150, 10 + i * 20))
    
    # Get network structure
    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys
    hidden_keys = [k for k in genome.nodes.keys() if k not in input_keys and k not in output_keys]
    
    # Node parameters
    node_radius = 15
    node_positions = {}
    
    # Position input nodes (left side)
    input_labels = ["L-60°", "L-30°", "L-15°", "Front", "R-15°", "R-30°", "R-60°", "Speed", "Angle"]
    for i, key in enumerate(input_keys):
        x = 40
        y = 60 + i * (net_height - 120) / len(input_keys)
        node_positions[key] = (x, y)
    
    # Position output nodes (right side)
    output_labels = ["Steering", "Throttle"]
    for i, key in enumerate(output_keys):
        x = net_width - 40
        y = 100 + i * 80
        node_positions[key] = (x, y)
    
    # Position hidden nodes (middle)
    if hidden_keys:
        for i, key in enumerate(hidden_keys):
            x = net_width // 2
            y = 60 + i * (net_height - 120) / len(hidden_keys)
            node_positions[key] = (x, y)
    
    # Draw connections with real-time signal flow
    current_time = pygame.time.get_ticks()
    for conn_key, conn in genome.connections.items():
        if not conn.enabled:
            continue
        
        in_node, out_node = conn_key
        if in_node in node_positions and out_node in node_positions:
            start_pos = node_positions[in_node]
            end_pos = node_positions[out_node]
            
            # Weight-based visualization
            weight = conn.weight
            abs_weight = abs(weight)
            
            # Get signal strength from input node
            signal_strength = 0
            if hasattr(car.network_state, 'inputs') and in_node < len(car.network_state.inputs):
                signal_strength = abs(car.network_state.inputs[in_node])
            elif hasattr(car.network_state, 'hidden_activations') and in_node in car.network_state.hidden_activations:
                signal_strength = abs(car.network_state.hidden_activations[in_node])
            
            # Color based on weight and signal strength
            base_intensity = min(255, int(50 + abs_weight * 100))
            signal_boost = min(150, int(signal_strength * 150))
            
            if weight > 0:
                color = (0, min(255, base_intensity + signal_boost), 0)
            else:
                color = (min(255, base_intensity + signal_boost), 0, 0)
            
            # Thickness based on weight strength and signal
            thickness = max(1, min(6, int(abs_weight * 2 + signal_strength * 2)))
            
            # Draw connection
            pygame.draw.line(network_surface, color, start_pos, end_pos, thickness)
            
            # Draw signal flow animation
            if signal_strength > 0.1:
                progress = (current_time * 0.005) % 1.0
                dot_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
                dot_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
                
                dot_color = (255, 255, 255, min(255, int(signal_strength * 255)))
                pygame.draw.circle(network_surface, dot_color[:3], (int(dot_x), int(dot_y)), 3)
    
    # Draw nodes with real-time activation
    font = pygame.font.SysFont(UI_FONT, 9)
    value_font = pygame.font.SysFont(UI_FONT, 8)
    
    for key, pos in node_positions.items():
        # Get activation value
        activation = 0
        if key in input_keys:
            idx = input_keys.index(key)
            if hasattr(car.network_state, 'inputs') and idx < len(car.network_state.inputs):
                activation = car.network_state.inputs[idx]
            color = (100, 150, 255)  # Blue for inputs
            label = input_labels[idx]
        elif key in output_keys:
            idx = output_keys.index(key)
            if hasattr(car.network_state, 'outputs') and idx < len(car.network_state.outputs):
                activation = car.network_state.outputs[idx]
            color = (255, 100, 100)  # Red for outputs
            label = output_labels[idx]
        else:
            if hasattr(car.network_state, 'hidden_activations') and key in car.network_state.hidden_activations:
                activation = car.network_state.hidden_activations[key]
            color = (100, 255, 100)  # Green for hidden
            label = f"H{key}"
        
        # Color intensity based on activation
        intensity = min(255, int(abs(activation) * 255))
        activated_color = (
            min(255, color[0] + intensity),
            min(255, color[1] + intensity),
            min(255, color[2] + intensity)
        )
        
        # Draw node with activation glow
        if abs(activation) > 0.1:
            glow_radius = node_radius + int(abs(activation) * 10)
            glow_color = (*activated_color, 100)
            for r in range(glow_radius, node_radius, -2):
                alpha = int(50 * (glow_radius - r) / (glow_radius - node_radius))
                glow_surface = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*activated_color, alpha), (r, r), r)
                network_surface.blit(glow_surface, (int(pos[0] - r), int(pos[1] - r)))
        
        # Draw main node
        pygame.draw.circle(network_surface, activated_color, (int(pos[0]), int(pos[1])), node_radius)
        pygame.draw.circle(network_surface, (255, 255, 255), (int(pos[0]), int(pos[1])), node_radius, 2)
        
        # Draw label
        label_lines = label.split('\n')
        for i, line in enumerate(label_lines):
            text = font.render(line, True, (255, 255, 255))
            text_rect = text.get_rect(center=(int(pos[0]), int(pos[1]) + node_radius + 12 - (len(label_lines)//2 - i) * 10))
            network_surface.blit(text, text_rect)
        
        # Draw activation value
        value_text = value_font.render(f"{activation:.2f}", True, (255, 255, 255))
        value_rect = value_text.get_rect(center=(int(pos[0]), int(pos[1]) + node_radius + 24))
        network_surface.blit(value_text, value_rect)
    
    # Blit to main screen
    screen.blit(network_surface, (net_x, net_y))
    
    # Draw connection line from car to network panel
    if hasattr(car, 'x') and hasattr(car, 'y'):
        car_pos = (int(car.x), int(car.y))
        panel_pos = (net_x + 30, net_y + 30)
        pygame.draw.line(screen, car.color, car_pos, panel_pos, 2)
def draw_controls_panel(screen):
    """Draw the controls help panel"""
    small_font = pygame.font.SysFont(UI_FONT, UI_SMALL_TEXT_SIZE)
    
    controls_text = [
        "Controls:",
        "S - Toggle Sensors",
        "N - Toggle Network",
        "G - Toggle Manual Advance",
        "Space - Next Generation (if manual)"
    ]
    
    for i, text in enumerate(controls_text):
        color = (200, 200, 200) if i == 0 else (150, 150, 150)
        text_surface = small_font.render(text, True, color)
        screen.blit(text_surface, (screen.get_width() - 250, 10 + i * 20))

def draw_generation_info(screen, generation, alive_cars, best_fitness):
    """Draw basic generation information"""
    font = pygame.font.SysFont(UI_FONT, UI_TEXT_SIZE)
    
    generation_text = font.render(f"Generation: {generation}", True, (255, 255, 255))
    cars_text = font.render(f"Cars Alive: {alive_cars}", True, (255, 255, 255))
    best_fitness_text = font.render(f"Best Fitness: {best_fitness:.2f}", True, (255, 255, 255))
    
    screen.blit(generation_text, (10, 10))
    screen.blit(cars_text, (10, 35))
    screen.blit(best_fitness_text, (10, 60))

# =============================================
# CAR AND SIMULATION CLASSES
# =============================================

class NetworkState:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.hidden_activations = {}
        self.last_update_time = 0

class Car:
    def __init__(self, x, y, angle=0, color=(255, 0, 0)):
        self.checkpoint_times = [] 
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.width = 16
        self.height = 8
        self.color = color
        self.sensors = []
        self.collisions = 0
        self.last_checkpoint = -1
        self.laps_completed = 0
        self.distance_traveled = 0
        self.last_position = (x, y)
        self.idle_time = 0
        self.alive = True
        self.fitness = 0
        self.steering_input = 0
        self.throttle_input = 0
        self.checkpoint_times = []
        self.total_time = 0
        self.max_speed = BASE_MAX_SPEED
        self.frames_since_last_checkpoint = 0
        self.checkpoint_progress = 0
        self.boundary_collisions = 0
        self.frames_on_track = 0
        self.total_frames = 0
        self.frames_without_boundary_collision = 0
        self.avg_distance_from_center = 0
        self.distances_from_center = []
        self.network_state = NetworkState()

    def get_corners(self):
        corners = []
        half_width = self.width / 2
        half_height = self.height / 2
        
        if not all(isinstance(val, (int, float)) and not math.isnan(val) and math.isfinite(val) 
                for val in [self.x, self.y, self.angle, half_width, half_height]):
            return []
        
        try:
            front_x = self.x + math.cos(math.radians(self.angle)) * half_width
            front_y = self.y + math.sin(math.radians(self.angle)) * half_width
            front_left_x = front_x + math.cos(math.radians(self.angle + 90)) * half_height
            front_left_y = front_y + math.sin(math.radians(self.angle + 90)) * half_height
            front_right_x = front_x + math.cos(math.radians(self.angle - 90)) * half_height
            front_right_y = front_y + math.sin(math.radians(self.angle - 90)) * half_height
            
            rear_x = self.x - math.cos(math.radians(self.angle)) * half_width
            rear_y = self.y - math.sin(math.radians(self.angle)) * half_width
            rear_left_x = rear_x + math.cos(math.radians(self.angle + 90)) * half_height
            rear_left_y = rear_y + math.sin(math.radians(self.angle + 90)) * half_height
            rear_right_x = rear_x + math.cos(math.radians(self.angle - 90)) * half_height
            rear_right_y = rear_y + math.sin(math.radians(self.angle - 90)) * half_height
            
            return [
                (front_left_x, front_left_y),
                (front_right_x, front_right_y),
                (rear_right_x, rear_right_y),
                (rear_left_x, rear_left_y)
            ]
        except (ValueError, TypeError, ZeroDivisionError):
            return [] 

    def update_sensors(self):
        self.sensors = []
        front_x = self.x + math.cos(math.radians(self.angle)) * self.width / 2
        front_y = self.y + math.sin(math.radians(self.angle)) * self.width / 2
        
        for angle in SENSOR_ANGLES:
            sensor_angle = self.angle + angle
            end_x = front_x + math.cos(math.radians(sensor_angle)) * SENSOR_LENGTH
            end_y = front_y + math.sin(math.radians(sensor_angle)) * SENSOR_LENGTH
            
            min_distance = SENSOR_LENGTH
            hit_point = (end_x, end_y)
            
            # Check outer track boundaries
            for i in range(len(TRACK_OUTER)):
                p1 = TRACK_OUTER[i]
                p2 = TRACK_OUTER[(i + 1) % len(TRACK_OUTER)]
                intersection = self.line_intersection(
                    (front_x, front_y), (end_x, end_y), p1, p2
                )
                if intersection is not None and intersection["distance"] < min_distance:
                    min_distance = intersection["distance"]
                    hit_point = (intersection["x"], intersection["y"])
            
            # Check inner track boundaries
            for i in range(len(TRACK_INNER)):
                p1 = TRACK_INNER[i]
                p2 = TRACK_INNER[(i + 1) % len(TRACK_INNER)]
                intersection = self.line_intersection(
                    (front_x, front_y), (end_x, end_y), p1, p2
                )
                if intersection is not None and intersection["distance"] < min_distance:
                    min_distance = intersection["distance"]
                    hit_point = (intersection["x"], intersection["y"])
            
            self.sensors.append({
                "start": (front_x, front_y),
                "end": hit_point,
                "distance": min_distance
            })

    def line_intersection(self, line1_start, line1_end, line2_start, line2_end):
        x1, y1 = line1_start
        x2, y2 = line1_end
        x3, y3 = line2_start
        x4, y4 = line2_end
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None
        
        try:
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        except (ZeroDivisionError, TypeError):
            return None
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
            return {"x": x, "y": y, "distance": distance}
        
        return None
    def draw_highlight(self, screen, highlight_color=(255, 255, 0), thickness=3):
        """Draw a highlight around the car"""
        corners = self.get_corners()
        if len(corners) == 4:
            pygame.draw.polygon(screen, highlight_color, corners, thickness)
    def is_on_track(self, point):
        return (self.point_in_polygon(point, TRACK_OUTER) and 
                not self.point_in_polygon(point, TRACK_INNER))

    def point_in_polygon(self, point, polygon):
        x, y = point
        inside = False
        n = len(polygon)
        p1x, p1y = polygon[0]
        
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def passed_checkpoint(self, checkpoint):
        line1_start = self.last_position
        line1_end = (self.x, self.y)
        line2_start = checkpoint["line"][0]
        line2_end = checkpoint["line"][1]
        
        intersection = self.line_intersection(
            line1_start, line1_end, line2_start, line2_end
        )
        
        return intersection is not None

    def check_checkpoints(self):
        """
        Check if the car has passed the next checkpoint in sequence
        Updates lap count, checkpoint status, and records timing information
        """
        next_checkpoint_id = (self.last_checkpoint + 1) % len(CHECKPOINTS)
        checkpoint = CHECKPOINTS[next_checkpoint_id]
        
        if self.passed_checkpoint(checkpoint):
            # Update checkpoint status
            self.last_checkpoint = next_checkpoint_id
            self.frames_since_last_checkpoint = 0
            
            # Record checkpoint time (in seconds)
            current_time = pygame.time.get_ticks() / 1000  
            self.checkpoint_times.append(current_time)
            
            # Calculate time since last checkpoint (for reward calculation)
            time_since_last = 0
            if len(self.checkpoint_times) > 1:
                time_since_last = current_time - self.checkpoint_times[-2]
            
            # Apply checkpoint reward
            self.fitness += checkpoint["reward"]
            
            # Additional time-based bonus (faster = better)
            if len(self.checkpoint_times) > 1:
                time_bonus = max(0, 20 - time_since_last)
                self.fitness += time_bonus
            
            # Check for completed lap (when passing checkpoint 0)
            if next_checkpoint_id == 0:
                self.laps_completed += 1
                self.fitness += 2000  # Large bonus for completing a lap
                
                # Calculate and store lap time if we have full lap data
                if len(self.checkpoint_times) >= len(CHECKPOINTS):
                    lap_start_time = self.checkpoint_times[-len(CHECKPOINTS)]
                    lap_time = current_time - lap_start_time
                    if hasattr(self, 'lap_times'):
                        self.lap_times.append(lap_time)
                    else:
                        self.lap_times = [lap_time]
                    
                    print(f"Car completed lap {self.laps_completed} in {lap_time:.2f}s")
            
            # Reset the progress timer
            self.checkpoint_progress = 0
    def calculate_fitness(self):
        fitness = self.distance_traveled * 0.8
        
        if self.last_checkpoint >= 0:
            fitness += 200 * (2.0 ** self.last_checkpoint)
        
        fitness += 10000 * (self.laps_completed ** 2)
        
        speed_bonus = min(self.speed, 1.0) * 5
        fitness += speed_bonus
        
        steering_penalty = abs(self.steering_input) * 2
        fitness -= steering_penalty
        
        fitness -= self.total_time * 0.2
        fitness -= 100 * self.collisions
        
        if self.frames_since_last_checkpoint > FPS * 10:
            fitness -= 20 * (self.frames_since_last_checkpoint / FPS)
        
        fitness -= 500 * self.boundary_collisions
        
        if self.total_frames > 0:
            track_adherence_ratio = self.frames_on_track / self.total_frames
            fitness += 100 * track_adherence_ratio
        
        center_bonus = max(0, 50 - self.avg_distance_from_center * 10)
        fitness += center_bonus
        
        consecutive_bonus = min(self.frames_without_boundary_collision * 0.1, 50)
        fitness += consecutive_bonus
        
        self.fitness = max(0, fitness)
        return self.fitness

    def update_boundary_status(self, is_on_track):
        self.total_frames += 1
        
        if is_on_track:
            self.frames_on_track += 1
            self.frames_without_boundary_collision += 1
        else:
            self.boundary_collisions += 1
            self.frames_without_boundary_collision = 0

    def update_distance_from_center(self, distance):
        self.distances_from_center.append(distance)
        self.avg_distance_from_center = sum(self.distances_from_center) / len(self.distances_from_center)

    def update(self, steering, throttle):
        if not self.alive:
            return
        
        try:
            steering = float(steering)
            throttle = float(throttle)
        except (TypeError, ValueError):
            return
            
        self.total_time += 1
        self.frames_since_last_checkpoint += 1
        
        max_turn = 2.5
        self.angle += steering * max_turn
        
        acceleration = throttle * 0.1
        deceleration = self.speed * 0.05
        self.speed += acceleration - deceleration
        self.speed = max(0, min(self.max_speed, self.speed))

        new_x = self.x + math.cos(math.radians(self.angle)) * self.speed
        new_y = self.y + math.sin(math.radians(self.angle)) * self.speed

        distance_moved = math.sqrt((new_x - self.x)**2 + (new_y - self.y)**2)
        self.distance_traveled += distance_moved

        center_on_track = self.is_on_track((new_x, new_y))
        self.update_boundary_status(center_on_track)
        
        if center_on_track:
            self.last_position = (self.x, self.y)
            self.x = new_x
            self.y = new_y
            self.check_checkpoints()
            self.idle_time = 0
        else:
            self.collisions += 1
            self.speed *= 0.5
            self.fitness -= 25

            if self.collisions > 20:
                self.alive = False

        if self.speed <= 0.05:
            self.idle_time += 1
            if self.idle_time > FPS * 8:
                self.alive = False
        
        if self.frames_since_last_checkpoint > FPS * 25:
            self.alive = False

        self.update_sensors()
        self.calculate_fitness()

    def draw(self, screen):
        if not self.alive:
            return

        corners = self.get_corners()
        pygame.draw.polygon(screen, self.color, corners)

        if show_sensors:
            for sensor in self.sensors:
                color = (255, 0, 0) if sensor["distance"] < 20 else SENSOR_COLOR
                pygame.draw.line(screen, color, sensor["start"], sensor["end"], 1)

    def get_sensor_data(self):
        return [sensor["distance"] / SENSOR_LENGTH for sensor in self.sensors]

# =============================================
# NEAT CONFIGURATION AND EVALUATION
# =============================================

class FixedStagnation(DefaultStagnation):
    def remove_stagnant(self, generation):
        """Override to first remove empty species before stagnation checks"""
        self.species = {sid: species for sid, species in self.species.items() 
                       if species.members}
        return super().remove_stagnant(generation)

    def update(self, species_set, generation):
        super().update(species_set, generation)

class CustomReporter(neat.reporting.BaseReporter):
    def __init__(self):
        self.generation = 0
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

    def start_generation(self, generation):
        self.generation = generation
        self.generation_start_time = time.time()
        print(f'\n****** Running generation {generation} ******')

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.generation_start_time is not None:
            elapsed = time.time() - self.generation_start_time
            self.generation_times.append(elapsed)
            print(f'Population of {ng} members in {ns} species:')
            
            fitnesses = [c.fitness for c in population.values()]
            fit_mean = mean(fitnesses) if fitnesses else 0
            fit_std = stdev(fitnesses) if len(fitnesses) > 1 else 0
            best_fitness = max(fitnesses) if fitnesses else 0
            worst_fitness = min(fitnesses) if fitnesses else 0
            
            print(f'  Best fitness: {best_fitness:.6f}')
            print(f'  Worst fitness: {worst_fitness:.6f}')
            print(f'  Mean fitness: {fit_mean:.6f}')
            print(f'  Stdev fitness: {fit_std:.6f}')
            print(f'Generation time: {elapsed:.3f} sec')
            
            return False

def post_evaluate(self, config, population, species, best_genome):
    if best_genome is None:
        print("Warning: No best genome to report")
        return
             
    try:
        if best_genome.key in population:
            try:
                best_species_id = species.get_species_id(best_genome.key)
                print(f"Best genome {best_genome.key} in species {best_species_id} with fitness {best_genome.fitness}")
            except KeyError:
                print(f"Best genome {best_genome.key} exists in population but not in species mapping - fitness: {best_genome.fitness}")
        else:
            print(f"Best genome {best_genome.key} not in current population - fitness: {best_genome.fitness}")
    except Exception as e:
        print(f"Could not determine species for best genome: {str(e)}")
        if hasattr(best_genome, 'fitness'):
            print(f"Best genome fitness: {best_genome.fitness}")

def complete_extinction(self):
    self.num_extinctions += 1
    print(f'All species extinct (extinction #{self.num_extinctions})')

def found_solution(self, config, generation, best):
    print(f'Found solution in generation {generation}')
    return False

def create_neat_config():
    config_text = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 9999999999
pop_size              = 100
reset_on_extinction   = True

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh relu sigmoid
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.3
conn_delete_prob        = 0.3
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full
node_add_prob           = 0.2
node_delete_prob        = 0.2
num_hidden              = 0
num_inputs              = 9
num_outputs             = 2
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
"""
    
    with open('neat_config.txt', 'w') as f:
        f.write(config_text)
    
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'neat_config.txt'
    )
    
    config.stagnation_config.max_stagnation = 30
    config.stagnation_config.species_elitism = 2
    
    return config

class CustomPopulation(neat.Population):
    def __init__(self, config, initial_state=None):
        super().__init__(config, initial_state)
        self.best_genome = None
        self.best_fitness = float('-inf')
        
    def run(self, fitness_function, n=None):
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1
            self.reporters.start_generation(self.generation)

            fitness_function(list(self.population.items()), self.config)

            current_best = None
            current_best_fitness = float('-inf')
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome")
                if g.fitness > current_best_fitness:
                    current_best = g
                    current_best_fitness = g.fitness

            if current_best and (self.best_genome is None or current_best_fitness > self.best_fitness):
                self.best_genome = current_best
                self.best_fitness = current_best_fitness

            reporting_genome = current_best if current_best else None
            
            if reporting_genome:
                self.reporters.post_evaluate(self.config, self.population, self.species, reporting_genome)

            if not self.config.no_fitness_termination:
                fv = self.fitness_criterion(g.fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    if reporting_genome:
                        self.reporters.found_solution(self.config, self.generation, reporting_genome)

            if not self.species.species:
                self.reporters.complete_extinction()
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                  self.config.genome_config,
                                                                  self.config.pop_size)
                else:
                    raise neat.CompleteExtinctionException()

            self.species.speciate(self.config, self.population, self.generation)
            self.reporters.end_generation(self.config, self.population, self.species)
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                         self.config.pop_size, self.generation)

            if not self.species.species:
                self.reporters.complete_extinction()
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                  self.config.genome_config,
                                                                  self.config.pop_size)
                else:
                    raise neat.CompleteExtinctionException()

            self.generation += 1

        return self.best_genome

def eval_genomes(genomes, config):
    global show_sensors, show_network, manual_generation_advance, generation_stats, best_genome_history, global_best_genome, global_best_fitness, screen, race_monitor
    
    # Initialize race monitor
    from race_monitor import RaceMonitor
    race_monitor = RaceMonitor()
    
    screen = pygame.display.get_surface()
    
    cars = []
    nets = []
    ge = []
    
    current_max_speed = min(BASE_MAX_SPEED + pop.generation * 0.05, 6)
    
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        color = CAR_COLORS[genome_id % len(CAR_COLORS)]
        car = Car(400, 495, 0, color)
        car.max_speed = current_max_speed
        car.update_sensors()
        cars.append(car)
        nets.append(net)
        ge.append(genome)
        race_monitor.track_car(car)
    
    running = True
    generation_finished = False
    start_time = pygame.time.get_ticks()
    generation_time_limit = GENERATION_TIME_LIMIT * 1000
    clock = pygame.time.Clock()
    highlighted_car = None
    
    while running and not generation_finished:
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - start_time
        time_left = max(0, (generation_time_limit - elapsed_time) / 1000)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    show_sensors = not show_sensors
                elif event.key == pygame.K_n:
                    show_network = not show_network
                    # When toggling network view, update highlighted car
                    if show_network and best_car:
                        highlighted_car = best_car
                    else:
                        highlighted_car = None
                elif event.key == pygame.K_g:
                    manual_generation_advance = not manual_generation_advance
                    print(f"Manual generation advance: {'ON' if manual_generation_advance else 'OFF'}")
                elif event.key == pygame.K_SPACE and manual_generation_advance:
                    generation_finished = True
                    print("Manually advancing to next generation")
                elif event.key == pygame.K_h:  # Cycle through cars to highlight
                    if show_network and cars:
                        if highlighted_car:
                            current_index = cars.index(highlighted_car)
                            next_index = (current_index + 1) % len(cars)
                            highlighted_car = cars[next_index]
                        else:
                            highlighted_car = cars[0]
        
        if not manual_generation_advance:
            active_cars = [car for car in cars if car.alive]
            if len(active_cars) == 0 or elapsed_time >= generation_time_limit:
                generation_finished = True
       
        if not pygame.display.get_init():
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        i = 0
        best_car = None
        best_fitness = -1
        
        while i < len(cars):
            car = cars[i]
            if not car.alive:
                cars.pop(i)
                nets.pop(i)
                ge.pop(i)
                if highlighted_car == car:
                    highlighted_car = None
                continue
            
            sensor_data = car.get_sensor_data()
            speed_input = car.speed / car.max_speed
            angle_input = math.sin(math.radians(car.angle)) * 0.5
            
            inputs = sensor_data + [speed_input, angle_input]
            output = nets[i].activate(inputs)
            
            car.network_state.inputs = inputs.copy()
            car.network_state.outputs = output.copy()
            car.network_state.last_update_time = current_time
            
            if hasattr(nets[i], 'values'):
                car.network_state.hidden_activations = nets[i].values.copy()
            
            steering = output[0]
            throttle = output[1]
            
            car.update(steering, throttle)
            ge[i].fitness = car.fitness
            race_monitor.update(car)
            
            if car.fitness > best_fitness:
                best_fitness = car.fitness
                best_car = car
            
            i += 1
                
        # Draw everything
        screen.fill((50, 50, 50))
        draw_track(screen)
        
        # Draw all cars first
        for car in cars:
            car.draw(screen)
        
        # Highlight the selected car (either best car or manually selected)
        if show_network:
            car_to_highlight = highlighted_car if highlighted_car else best_car
            if car_to_highlight and car_to_highlight.alive:
                car_to_highlight.draw_highlight(screen)
                # Draw connection lines from car to network visualization
                if show_network:
                    start_pos = (car_to_highlight.x, car_to_highlight.y)
                    end_pos = (SCREEN_WIDTH - 380 + 30, 120)  # Network panel position
                    pygame.draw.line(screen, (255, 255, 0, 100), start_pos, end_pos, 1)
        
        active_cars_count = len([car for car in cars if car.alive])
        draw_generation_info(screen, pop.generation, active_cars_count, best_fitness)
        draw_controls_panel(screen)
        
        # Show real-time stats for the highlighted car
        if show_network and highlighted_car and highlighted_car.alive:
            draw_stats_panel(screen, highlighted_car, 10, SCREEN_HEIGHT - 210)
        elif best_car and best_car.alive:
            draw_stats_panel(screen, best_car, 10, SCREEN_HEIGHT - 210)
        
        # Show neural network visualization
        if show_network:
            car_to_show = highlighted_car if highlighted_car else best_car
            if car_to_show and car_to_show.alive:
                genome_to_show = ge[cars.index(car_to_show)] if car_to_show in cars else None
                if genome_to_show:
                    draw_realtime_network(screen, car_to_show, genome_to_show, config)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    # Generation summary
    if best_car:
        print("\n=== GENERATION SUMMARY ===")
        print(f"Best car ID: {id(best_car)}")
        print(f"Fitness: {best_fitness:.2f}")
        print(race_monitor.get_stats(best_car))
        print("=========================\n")
    
    if ge:
        generation_best_fitness = max(genome.fitness for genome in ge)
        generation_avg_fitness = sum(genome.fitness for genome in ge) / len(ge)
        
        generation_stats.append({
            'generation': pop.generation,
            'best_fitness': generation_best_fitness,
            'avg_fitness': generation_avg_fitness,
            'cars_alive': len([car for car in cars if car.alive]),
            'best_laps': best_car.laps_completed if best_car else 0
        })
        
        if generation_best_fitness > global_best_fitness:
            global_best_fitness = generation_best_fitness
            global_best_genome = max(ge, key=lambda g: g.fitness)
        
        print(f"Generation {pop.generation} - Best: {generation_best_fitness:.2f}, Avg: {generation_avg_fitness:.2f}")
def run_neat_evolution():
    global pop, generation_stats, best_genome_history, global_best_genome, global_best_fitness
    
    generation_stats = []
    best_genome_history = []
    global_best_genome = None
    global_best_fitness = -float('inf')
    
    try:
        config = create_neat_config()
        pop = neat.Population(config) 
        
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        
        def post_evaluate(config, pop, species, best_genome):
            global global_best_genome, global_best_fitness
            
            if best_genome is None:
                print("Warning: No best genome to report")
                return
                    
            if best_genome.fitness > global_best_fitness:
                global_best_fitness = best_genome.fitness
                global_best_genome = best_genome
        
        pop.add_reporter(neat.Checkpointer(10, filename_prefix='neat-checkpoint-'))
        pop.reporters.reporters.append(post_evaluate)

        generation = 0
        while generation < 1000:
            try:
                pop.run(eval_genomes, 1)
                generation += 1
                
                generation_stats.append({
                    'generation': generation,
                    'best_fitness': max([g.fitness for g in pop.population.values()]),
                    'species_count': len(pop.species.species)
                })
                
            except neat.CompleteExtinctionException:
                print("All species extinct - restarting population")
                pop.population = pop.reproduction.create_new(
                    pop.config.genome_type,
                    pop.config.genome_config,
                    pop.config.pop_size
                )
                continue
                
            except Exception as e:
                print(f"Generation {generation} error: {str(e)}")
                break

        winner = max(pop.population.values(), key=lambda g: g.fitness)
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(winner, f)
        return winner

    except KeyboardInterrupt:
        print(f"\nEvolution stopped after {getattr(pop, 'generation', 0)} generations")
        if global_best_genome:
            with open('best_genome.pkl', 'wb') as f:
                pickle.dump(global_best_genome, f)
            print(f"Best genome (fitness: {global_best_fitness:.2f}) saved")
        return global_best_genome

def test_saved_genome(genome_file='best_genome.pkl'):
    try:
        with open(genome_file, 'rb') as f:
            genome = pickle.load(f)
        
        config = create_neat_config()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        print(f"Testing genome with fitness: {genome.fitness}")
        test_genomes = [(1, genome)]
        eval_genomes(test_genomes, config)
        
        return genome, net
        
    except FileNotFoundError:
        print(f"Genome file '{genome_file}' not found")
        return None, None

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("NEAT Car AI Evolution")
    
    
    
    config = create_neat_config()
    pop = neat.Population(config)
    
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    try:
        winner = pop.run(eval_genomes, GENERATIONS)
        
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(winner, f)
            
        # Test the winner with monitoring (MODIFIED THIS)
        print("\n=== TESTING BEST GENOME ===")
        test_genomes = [(1, winner)]
        eval_genomes(test_genomes, config)
        
        # Display final stats (ADD THIS)
        best_car = max(race_monitor.car_stats.keys(), 
                      key=lambda c: race_monitor.car_stats[c]['lap_count'])
        print("\n=== FINAL RACE STATISTICS ===")
        print(race_monitor.get_stats(best_car))
        
    except KeyboardInterrupt:
        print("\nEvolution stopped by user")
        # Display interim stats if interrupted (ADD THIS)
        if 'race_monitor' in globals() and race_monitor.car_stats:
            best_car = max(race_monitor.car_stats.keys(), 
                          key=lambda c: race_monitor.car_stats[c]['lap_count'])
            print("\n=== CURRENT RACE STATISTICS ===")
            print(race_monitor.get_stats(best_car))
    finally:
        pygame.quit()