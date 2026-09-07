import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
This solver utilizes the DOP853 method from the scipy library in order to solve the three-body problem in three dimensions. This code serves as a
baseline to compare with other results, such as that of ML.
'''

# Mass for each body
m1, m2, m3 = 1, 1, 1

# Initial positions of the bodies
init_pos_1 = [1.0, 1.0, 0.0]
init_pos_2 = [0.0, 1.0, 1.0]
init_pos_3 = [1.0, 0.0, 1.0]

# Initial velocity of the bodies
init_vel_1 = [0.0, 0.0, -1.0]
init_vel_2 = [0.0, 0.0, 1.0]
init_vel_3 = [0.0, 0.0, -0.6]

# The array of initial conditions for position and velocity, on the form (x1,y1,z1, x2,y2,z2 ... dx3/dt, dy3/dt, dz3/dt)
initial_conditions = np.concatenate((
    init_pos_1,
    init_pos_2,
    init_pos_3,
    init_vel_1,
    init_vel_2,
    init_vel_3
))

def system_odes(time, s, m1, m2, m3):

    # Establishes and returns the ODE for the three body problem

    # Extraxt the values from the state vector s. Note that f1, f2, f3 = dp1_dt, dp2_dt, dp3_dt
    p1, p2, p3 = s[:3], s[3:6], s[6:9]
    f1, f2, f3 = s[9:12], s[12:15], s[15:18]

    df1_dt = m3*(p3 - p1) / (np.linalg.norm(p3 - p1)**3) + m2*(p2 - p1) / (np.linalg.norm(p2 - p1)**3)
    df2_dt = m3*(p3 - p2) / (np.linalg.norm(p3 - p2)**3) + m1*(p1 - p2) / (np.linalg.norm(p1 - p2)**3)
    df3_dt = m1*(p1 - p3) / (np.linalg.norm(p1 - p3)**3) + m2*(p2 - p3) / (np.linalg.norm(p2 - p3)**3)

    return np.concatenate((f1, f2, f3, df1_dt, df2_dt, df3_dt))

# Time specifications (in seconds)
t0 = 0
t1 = 10
t_eval_pts = np.linspace(t0, t1, 200)

# Solving the system
def solve():
    sol = solve_ivp(
        system_odes,
        (t0, t1),
        t_eval = t_eval_pts,
        y0 = initial_conditions,
        method = 'DOP853',
        args = (m1, m2, m3)
    )

    return sol

# For a fun visual representation
def plot(sol):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    colors = ['red', 'blue', 'green']
    for i, color in enumerate(colors):
        ax.plot(sol.y[3*i], sol.y[3*i+1], sol.y[3*i+2], color, label=f'Body {i+1}', linewidth=1)

    ax.set_title("Trajectory of three bodies")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.grid()
    plt.legend()
    plt.show()

# Implement a main safe so no unnecessary calculations are done in other files
if __name__ == "__main__":
    sol = solve()
    plot(sol)
