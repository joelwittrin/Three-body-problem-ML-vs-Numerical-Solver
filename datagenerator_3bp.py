import numpy as np
from scipy.integrate import solve_ivp
from threebodysolver import system_odes

'''
Data generator for the three body problem, yielding random initial coordinates and velocities for each body. Note that certain
values are filtered out as these result in either unphysical scenarios or irrelevant cases for the ML created in this project.
These criterias can simply be removed or changed for other projects if needed. The full trajectories are stored over the given
time span.

The total energy E of the system decides whether it can stay together: if E < 0 the system is bound as a whole, while if E > 0
it must break up and at least one body escapes. To give the ML examples of both, the share of systems with E > 0 is controlled.
'''

def total_energy(p1, p2, p3, v1, v2, v3, m1, m2, m3, G = 1):

    # Returns the total energy, i.e. kinetic energy + potential energy. G = 1 to match the units used in threebodysolver.

    E_k = 0.5 * (m1 * np.dot(v1, v1) + m2 * np.dot(v2, v2) + m3 * np.dot(v3, v3))
    E_p = -G * (m1 * m2 / np.linalg.norm(p1 - p2) +
                m1 * m3 / np.linalg.norm(p1 - p3) +
                m2 * m3 / np.linalg.norm(p2 - p3))

    return E_k + E_p


def generate_training_data(n = 100):

    # Generates training data and returns the state vector s used in the file threebodysolver

    # Mass for each body
    m1, m2, m3 = 1, 1, 1

    # Initial positions are drawn in [0, box_size] and each velocity component in [-max_velocity_comp, max_velocity_comp]
    box_size = 3.0
    max_velocity_comp = 1.2

    # Set min_space = 0 and max_velocity = np.inf to remove these criterias
    min_space = 0.3
    max_velocity = 2.0

    # Share of the systems that should have positive energy (E > 0), i.e. that must break up. Set to None to remove this criteria.
    positive_energy_share = 0.5
    n_positive = 0
    n_negative = 0

    x = []
    y = []

    # Our state vector s = (x1,y1,z1, x2,y2,z2 ... dx3/dt, dy3/dt, dz3/dt)

    # Time specifications (in seconds). Make sure these correlate between this file and threebodysolver
    t0 = 0
    # A dense time grid is used so that close encounters between the time points are not missed by the classifier
    t1 = 30
    t_eval_pts = np.linspace(t0, t1, 1000)

    while len(x) < n:
        p1 = np.random.uniform(0.0, box_size, 3)
        p2 = np.random.uniform(0.0, box_size, 3)
        p3 = np.random.uniform(0.0, box_size, 3)

        # To filter out positions and velocities we require a minimum distance between the bodies and a maximum velocity
        if (np.linalg.norm(p1 - p2) < min_space or
            np.linalg.norm(p1 - p3) < min_space or
            np.linalg.norm(p2 - p3) < min_space):
            continue

        # Velocities may point in any direction
        v1 = np.random.uniform(-max_velocity_comp, max_velocity_comp, 3)
        v2 = np.random.uniform(-max_velocity_comp, max_velocity_comp, 3)
        v3 = np.random.uniform(-max_velocity_comp, max_velocity_comp, 3)

        # Move to the centre of mass frame, so that the system starts at the origin and does not drift away as a whole
        com_pos = (m1 * p1 + m2 * p2 + m3 * p3) / (m1 + m2 + m3)
        com_vel = (m1 * v1 + m2 * v2 + m3 * v3) / (m1 + m2 + m3)
        p1, p2, p3 = p1 - com_pos, p2 - com_pos, p3 - com_pos
        v1, v2, v3 = v1 - com_vel, v2 - com_vel, v3 - com_vel

        if (np.linalg.norm(v1) > max_velocity or
            np.linalg.norm(v2) > max_velocity or
            np.linalg.norm(v3) > max_velocity):
            continue

        # Skip the system if we already have enough systems with the same sign of the energy
        E = total_energy(p1, p2, p3, v1, v2, v3, m1, m2, m3)
        if positive_energy_share is not None:
            if E > 0 and n_positive >= positive_energy_share * n:
                continue
            if E <= 0 and n_negative >= (1 - positive_energy_share) * n:
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

            if E > 0:
                n_positive += 1
            else:
                n_negative += 1

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

def debug():
    x, y = generate_training_data(n=101)
    print(x.shape)
    print(y.shape)
    return None

if __name__ == "__main__":
    debug()