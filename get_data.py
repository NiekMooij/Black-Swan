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

from black_swan import get_network, generate_random_string

if __name__ == "__main__":

    # time_period = 1e4
    m = 50
    lambda_birth = 1
    lambda_death_range = np.arange(0.2, lambda_birth, 0.8/3)
    data = []
    for lambda_death in lambda_death_range:
        time_period = 10**4 * (1 - lambda_death/lambda_birth)**(-1)
        G = get_network(time_period, m, lambda_birth, lambda_death)
        
        new_row = {
            'id': generate_random_string(10),
            'type': 'black_swan',
            'size': len(G),
            'vertices': G.nodes(), 
            'edges': G.edges(),
            'time_period': time_period,
            'm': m,
            'lambda_birth': lambda_birth,
            'lambda_death': lambda_death,
            }

        data.append(new_row)

        df = pd.DataFrame(data)

        save_name = os.path.join(sys.path[0], 'data/black_swan.csv')
        df.to_csv(save_name, index=False)

        save_name = os.path.join(sys.path[0], 'data/black_swan.pkl')    
        df.to_pickle(save_name)