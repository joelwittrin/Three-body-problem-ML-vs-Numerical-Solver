import numpy as np
from scipy.integrate import solve_ivp
from threebodysolver import system_odes

'''
Data generator for the three body problem, yielding random initial coordinates and velocities for each body. Note that certain
values are filtered out as these result in either unphysical scenarios or irrelevant cases for the ML created in this project.
These criterias can simply be removed or changed for other projects if needed. The full trajectories are stored over the given
time span.
'''

def generate_training_data(n = 100):

    # Generates training data and returns the state vector s used in the file threebodysolver

    # Mass for each body
    m1, m2, m3 = 1, 1, 1

    # Set min_space = 0 and max_velocity = np.inf to remove these criterias
    min_space = 0.1
    max_velocity = 2.0

    x = []
    y = []

    # Our state vector s = (x1,y1,z1, x2,y2,z2 ... dx3/dt, dy3/dt, dz3/dt)

    # Time specifications (in seconds). Make sure these correlate between this file and threebodysolver
    t0 = 0
    t1 = 10
    t_eval_pts = np.linspace(t0, t1, 100)

    while len(x) < n:
        p1 = np.random.uniform(0.0, 1.2, 3)
        p2 = np.random.uniform(0.0, 1.2, 3)
        p3 = np.random.uniform(0.0, 1.2, 3)

        # To filter out positions and velocities we require a minimum distance between the bodies and a maximum velocity
        if (np.linalg.norm(p1 - p2) < min_space or
            np.linalg.norm(p1 - p3) < min_space or
            np.linalg.norm(p2 - p3) < min_space):
            continue

        v1 = np.random.uniform(0.0, 1.2, 3)
        v2 = np.random.uniform(0.0, 1.2, 3)
        v3 = np.random.uniform(0.0, 1.2, 3)

        if (np.linalg.norm(v1) > max_velocity or
            np.linalg.norm(v2) > max_velocity or
            np.linalg.norm(v3) > max_velocity):
            continue

        initial_conditions = np.concatenate((p1,p2,p3,v1,v2,v3))

        print(f"Solving {len(x)+1}/{n}")
        sol = solve_ivp(
            system_odes,
            (t0, t1),
            t_eval = t_eval_pts,
            y0 = initial_conditions,
            method = 'DOP853',
            args = (m1, m2, m3)
        )

        # Filter out unstable solutions. Stores the initial state and the final state with all corresponding rows to the final column.
        if sol.status == 0:
            x.append(initial_conditions)
            y.append(sol.y)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

def debug():
    x, y = generate_training_data(n=101)
    print(x.shape)
    print(y.shape)
    return None

if __name__ == "__main__":
    debug()
