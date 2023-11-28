import random
import networkx as nx
import sys
import os
import pandas as pd
import string
import secrets
import numpy as np
import time
import matplotlib.pyplot as plt

def generate_random_string(length: int) -> str:
    """
    Generate a random string of alphanumeric characters.

    Parameters:
    - length (int): Length of the generated string.

    Returns:
    - str: Random string of the specified length.
    """
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

def get_next_event(time_steps_birth: np.ndarray, time_steps_death: np.ndarray) -> tuple:
    """
    Get the next event (birth or death) and associated information.

    Parameters:
    - time_steps_birth (np.ndarray): Array of birth event times.
    - time_steps_death (np.ndarray): Array of death event times.

    Returns:
    - tuple: Tuple containing event type ('birth' or 'death'),
             next event time, updated birth event times, and updated death event times.
    """
    if time_steps_birth[0] <= time_steps_death[0]:
        next_event_time = time_steps_birth[0]
        time_steps_birth = np.delete(time_steps_birth, 0)
        return 'birth', next_event_time, time_steps_birth, time_steps_death
        
    else:
        next_event_time = time_steps_death[0]
        time_steps_death = np.delete(time_steps_death, 0)
        return 'death', next_event_time, time_steps_birth, time_steps_death

def birth_event(G: nx.Graph, m: int) -> nx.Graph:
    """
    Perform a birth event in the network.

    Parameters:
    - G (nx.Graph): NetworkX graph representing the network.
    - m (int): Number of existing nodes to connect to the new node.

    Returns:
    - nx.Graph: Updated network after the birth event.
    """
    # Calculate probabilities for attaching to existing nodes
    probabilities = [G.degree(node) for node in G.nodes()]
    total_degree = sum(probabilities)
    probabilities = [prob / total_degree for prob in probabilities]

    # Select m existing nodes to connect to the new node based on preferential attachment
    selected_nodes = random.choices(list(G.nodes()), weights=probabilities, k=m)

    # Connect the new node to the selected nodes
    node_label = generate_random_string(15)
    G.add_node(node_label)
    for node in selected_nodes:
        G.add_edge(node_label, node)
        
    return G

def death_event(G: nx.Graph) -> nx.Graph:
    """
    Perform a death event in the network.

    Parameters:
    - G (nx.Graph): NetworkX graph representing the network.

    Returns:
    - nx.Graph: Updated network after the death event.
    """
    node_label = random.choice(list(G.nodes()))
    G.remove_node(node_label)
    
    return G

def get_network(time_period: float, m: int, lambda_birth: float, lambda_death: float) -> nx.Graph:
    """
    Generate a dynamic network using birth and death events.

    Parameters:
    - time_period (float): Total time period for network evolution.
    - m (int): Number of existing nodes to connect to a new node.
    - lambda_birth (float): Birth event rate.
    - lambda_death (float): Death event rate.

    Returns:
    - nx.Graph: Final network after the specified time period.
    """
    time_steps_birth_steps = np.random.exponential(scale=1/lambda_birth, size=int(time_period * lambda_birth * 10))
    time_steps_death_steps = np.random.exponential(scale=1/lambda_death, size=int(time_period * lambda_death * 10))
    time_steps_birth = np.cumsum(time_steps_birth_steps)
    time_steps_death = np.cumsum(time_steps_death_steps)
        
    G = nx.Graph()
    node_labels = [generate_random_string(15) for i in range(m)]
    
    # Start with a fully connected initial graph with m nodes
    for i in range(m):
        for j in range(i + 1, m):
            G.add_edge(node_labels[i], node_labels[j])

    # Preferential attachment and random deletion
    time_current = 0
    while time_current < time_period:
        if len(G.edges()) == 0:
            print('Network is empty')
            return G
        
        event, next_event_time, time_steps_birth, time_steps_death = get_next_event(time_steps_birth, time_steps_death)
        
        if event == 'birth':
            G = birth_event(G, m)
        
        if event == 'death':
            G = death_event(G)

        time_current = next_event_time
        
        percentage = np.round(time_current / time_period * 100, 2)
        print(f'{percentage} %', end='\r')

    return G