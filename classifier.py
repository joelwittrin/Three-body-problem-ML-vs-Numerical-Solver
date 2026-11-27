import numpy as np
from datagenerator_3bp import generate_training_data

'''
Classifier for the three trajectory types: collision, escape and bounded. The code checks each step, with each function returning a list
with T/F corresponding to each initial state (x,y,z). The parameters for each function has been chosen as to generate a variance in the
results.
'''

# Make sure that the masses match with the other files.
m1, m2, m3 = 1, 1, 1
# G = 1 to match the units used in threebodysolver
G = 1

def check_collision(y, collision_radius = 0.1):

    # Checks if any collision occurs between the bodies corresponding to each trajectory. Returns a list with T/F for each iteration.
    # Takes entire state vector y as input.
    # Given that each body is a point mass, no physical radius is present and the collision radius is purely arbitrary.
    # A radius of 0.1 is chosen to fit the initial positions in datagenerator_3bp, giving a reasonable share of collisions.

    lst = []

    # We implement a nestled for loop as we check each trajectory (i-loop) for each point in time (j-loop)s.
    # The array y has the shape y = [coordinates (9pts) + velocities(9pts), i iterations, j points in time], i.e. i 18xj matrices.
    for i in range(len(y)):

        collision = False

        for j in range(y.shape[2]):
            p1 = y[i, :3, j]
            p2 = y[i, 3:6, j]
            p3 = y[i, 6:9, j]

            d1d2 = np.linalg.norm(p1-p2)
            d2d3 = np.linalg.norm(p2-p3)
            d3d1 = np.linalg.norm(p3-p1)

            if d1d2 < collision_radius or d2d3 < collision_radius or d3d1 < collision_radius:
                collision = True
                print(f'Trajectory {i}, step {j}: Collision detected, distances: d1d2 as {d1d2}, d2d3 as {d2d3}, d3d1 as {d3d1}')
                break

        lst.append(collision)

    return lst


def check_escape(y, r_escape = 15):

    # Checks if a body leaves the system. Does not take into account if the two other bodies stay bounded.
    # r_escape can be adjusted as needed.
    # Takes entire state vector y as input.

    lst = []

    for i in range(len(y)):

        escape = False

        for j in range(y.shape[2]):
            p1 = y[i, :3, j]
            p2 = y[i, 3:6, j]
            p3 = y[i, 6:9, j]

            v1 = y[i, 9:12, j]
            v2 = y[i, 12:15, j]
            v3 = y[i, 15:18, j]

            d1d2 = np.linalg.norm(p1-p2)
            d2d3 = np.linalg.norm(p2-p3)
            d3d1 = np.linalg.norm(p3-p1)

            # The indexes correspond to each point, i.e. 1, 2 & 3
            sums = [d1d2 + d3d1, d1d2 + d2d3, d2d3 + d3d1]
            furthest_dist = np.argmax(sums)

            points = [p1, p2, p3]
            velocities = [v1, v2, v3]
            masses = [m1, m2, m3]

            # Use the centre of mass frame, so that a drift of the whole system is not counted as an escape
            com_pos = sum(masses[k] * points[k] for k in range(3)) / sum(masses)
            com_vel = sum(masses[k] * velocities[k] for k in range(3)) / sum(masses)

            m, v, pos = masses[furthest_dist], velocities[furthest_dist] - com_vel, points[furthest_dist]

            # Calculate the energy for the body that is furthest away, i.e. most likely to escape
            E_k = 0.5 * m * np.dot(v, v)
            E_p = sum(-G * m * masses[j] / np.linalg.norm(pos - points[j]) for j in range(3) if j != furthest_dist)

            if np.linalg.norm(pos - com_pos) > r_escape and E_k + E_p > 0:
                escape = True
                break

        lst.append(escape)

    return lst

            
def check_bounded(collision_lst, escape_lst):

    lst = []

    for i in range(len(collision_lst)):

        bounded = True

        if collision_lst[i] == True or escape_lst[i] == True:
            bounded = False

        lst.append(bounded)

    return lst


if __name__ == "__main__":

    x , y = generate_training_data()

    collision_lst = check_collision(y)
    escape_lst = check_escape(y)
    bounded_lst = check_bounded(collision_lst, escape_lst)

    print('\n================================================================\n')
    print("Collision: \n", collision_lst)
    print('\n================================================================\n')
    print("Escape: \n", escape_lst)
    print('\n================================================================\n')
    print("Bounded: \n", bounded_lst)