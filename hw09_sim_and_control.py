## The template python file for hw09 can be a bit confusing.
## Please feel free to only use whatever makes sense to you and
## delete the rest if you don't find it helpful.

# %% [markdown]
# # Homework 9

# %%
import dynamics as dyn
from visualization import VizScene
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from matplotlib import pyplot as pl

import numpy as np


# %% [markdown]
# # Problem #1

# %%
# set up model

# defining kinematic parameters for three-link planar robot
dh = [[0, 0, 0.2, 0],
      [0, 0, 0.2, 0],
      [0, 0, 0.2, 0]]

joint_type = ['r', 'r', 'r']

link_masses = np.array([1, 0.5, 0.3])
link_lengths = np.array([0.2,0.2,0.2])
# print(link_lengths)
gravity = np.array([0, -9.81, 0])

# defining three different centers of mass, one for each link
r_coms = [np.array([-0.1, 0, 0]),
          np.array([-0.1, 0, 0]),
          np.array([-0.1, 0, 0])]

# all terms except Izz are zero because they don't matter in the
# equations, Ixx, and Iyy are technically non-zero, we just don't
# rotate about those axes so it doesn't matter.
link_inertias = [np.diag([0, 0, 0.1]),
                 np.diag([0, 0, 0.1]),
                 np.diag([0, 0, 0.1])]

# the viscous friction coefficients for B*q_dot are:
B = np.diag([0.8, 0.3, 0.2])  # but you will need to either include
                              # these in your dynamic arm constructor,
                              # or just use them as a global variable
                              # in your EOM function for numerical
                              # integration.


# my assumption is that you are using your RNE dynamics
arm = dyn.SerialArmDyn(dh,
                       jt=joint_type,
                       mass=link_masses,
                       r_com=r_coms,
                       link_inertia=link_inertias)


# %% [markdown]
# part a) - calculate joint torques here using your RNE, or E-L equations
# %%
data = loadmat('desired_accel.mat')

t = data['t'].squeeze() # remove singleton dimensions
q = data['q'] # probably want to figure out shape of this, relates to axis of np.gradient

# print(q)
# TODO you will need to take the derivative of q to get qd and qdd,
# you can use np.gradient with q and the time vector "t" but you will need
# to specify the axis (0 or 1) that you are taking the gradient along,
# e.g. np.gradient(q, t, axis=)
qd = np.gradient(q,t,axis=0)
# print(qd)
qdd = []
for i in range(len(q)):
    # pre = 3/(link_masses*link_lengths*link_lengths)
    qddi = (3/(link_masses*link_lengths*link_lengths))*(-1*link_masses*gravity*np.cos(q[i])-B*qd[i]+t[i])
    qdd.append(qddi)

# qdd = (3/(link_masses*link_lengths*link_lengths))*(-1*link_masses*9.8*np.cos(q)-B*qd+t)
# print(qdd)

# # TODO - calc torques for every time step given q, qd, qdd

taus = []
wrenchess = []
for i in range(len(q)):
    tau, wrenches = arm.rne(q=q[i],qd=qd[i],qdd=qdd[i])
    taus.append(tau)
    wrenchess.append(wrenches)
taus = np.stack(taus)
wrenchess = np.stack(wrenchess)
# print(taus)
torque = taus

# If you have a vector of all torques called "torque",
# you can use the following code to plot:
pl.figure()
for i in range(arm.n):
    pl.subplot(arm.n, 1, i+1)
    pl.plot(t, torque[:,i])
    pl.ylabel('joint '+str(i+1))
pl.xlabel('time (s)')
pl.tight_layout()
pl.show()



# %% [markdown]
# part b) - perform numerical integration as specified in  problem statement


# %%
# def controller(q, qd, q_des):
#     # we will eventually define controllers like this that takes these inputs.

#     # setting torque to zero for right now.
#     torque = 0.0

#     return torque 


# Define the dynamics function (x_dot = f(x,u)) for integration
def one_dof_eom(t,x):
    qd = x[0:arm.n]
    q = x[arm.n:]

    # making an empty vector for x_dot that is the right size
    x_dot = np.zeros((arm.n*2,))

    # for this to work, you need to implement these next three functions
    M = arm.get_M(q)
    C_q_dot = arm.get_C(q,qd) # not the matrix C(q, qdot), it is C(q,qdot)@ q_dot
    G = arm.get_G(q, gravity)

    #adding viscous friction 
    B = np.array([0.1])

    q_des = q  # eventually we will define a desired q, and use it for control, but for now
               # we'll just set it equal to our current q. Either way, this will not
               # be used in the torque calculation at this point. 
               
    # calculating torque from our "controller" function u
    # torque = get_u(q, qd, q_des)

    # solving for q_ddot by moving all other terms to right hand side and inverting M(q)
    # print(qd)
    qdd = np.linalg.inv(M) @ (0.0 - C_q_dot - G - B*qd)
    # from original equation M(q)@qdd + C(q,qdot)@qdot + G(q) = torque - B@qd

    # filling the x_dot vector 
    x_dot[0:arm.n] = qdd
    x_dot[arm.n:] = qd

    return x_dot

# TODO - perform numerical integration here using "solve_ivp".
# When finished, you can use the plotting code below to help you.
num_steps = 1000
tf = 15.0
t_sim = np.linspace(0, tf, num=num_steps)

# this is the numerical integration performed in one function call. 
# it looks complicated but we'll talk through it in lecture. Here are brief descriptions to help:
#    fun=lambda t,x - this is just to tell the integration the state variables and time variable
#    one_dof_eom(t, x, control, q_des) - this is the f(x,u) function, it requires t, and x as inputs, but can take other inputs as shown.
#    t_span - is the length of time for the integration
#    t_eval - if defined, this specifies specific time steps where we want a solution from the integration (rather than letting the step size be determined by the algorithm)
#    y0 - is the initial condition for the integration (order is [q_dot, q] in this case, based on definition in eom)

xs = []
for i in range(len(q)):
    xi = []
    xi.extend(q[i])
    xi.extend(qd[i])
    xdi = one_dof_eom(0,xi)
    xs.append(xdi)

def system(t):
    return xs

sol = solve_ivp(fun=system, t_span=[0,tf], t_eval=t_sim, y0=np.array([0, 0]))


## NOTE: In all of the plotting code below, I assume x = [qd, q]. If you used
## x = [q, qd], you will need to change the indexing in the plotting code below.

# making an empty figure
fig = pl.figure()

# plotting the time vector "t" versus the solution vector for
# the three joint positions, entry 3-6 in sol.y
pl.plot(sol.t, sol.y[arm.n:].T)
pl.ylabel('joint positions (rad)')
pl.xlabel('time (s)')
pl.title('three-link robot falling in gravity field')
pl.show()


# now show the actual robot being simulated in pyqtgraph, this will only
# work if you have found the integration solution
# # %%
# # visualizing the robot acting under gravity
# viz = VizScene()
# viz.add_arm(arm)

# for i in range(len(sol.t)-1):
#     viz.update(qs=[sol.y[arm.n:,i]])
#     viz.hold(t[i+1]-t[i])
# viz.close_viz()


# %% [markdown]
# # Problem #2

# Define the dynamics function (x_dot = f(x,u)) for integration
def eom(t, x, u, qdd_des, qd_des, q_des):
    x_dot = np.zeros(2*arm.n)

    # TODO - calculate torque from a controller function (as in demo for one DoF)
    #      - then calculate qdd from that applied torque and other torques (C and G)

    return x_dot


# you can define any q_des, qd_des, and qdd_des you want, but feel free to use this
# code below if it makes sense to you. I'm just defining q as a function of time
# and then taking the symbolic derivative.
import sympy as sp

t = sp.symbols('t')
q_des_sp = sp.Matrix([sp.cos(2*sp.pi*t),
                      sp.cos(2*sp.pi*2*t),
                      sp.cos(2*sp.pi*3*t)])
qd_des_sp = q_des_sp.diff(t)
qdd_des_sp = qd_des_sp.diff(t)

# turn them into numpy functions so that they are faster and return
# the right data type. Now we can call these functions at any time "t"
# in the "eom" function.
q_des = sp.lambdify(t, q_des_sp, modules='numpy')
qd_des = sp.lambdify(t, qd_des_sp, modules='numpy')
qdd_des = sp.lambdify(t, qdd_des_sp, modules='numpy')

# %%
# TODO define three different control functions and numerically integrate for each one.
# If "sol" is output from simulation, plotting can look something like this:

num_sec =
time = np.linspace(0, num_sec, num=100*num_sec)
sol =

pl.figure()
title = "PD + G control"
for i in range(arm.n):
    pl.subplot(arm.n, 1, i+1)
    pl.plot(sol.t, sol.y[arm.n+i,:].T, label='actual')
    pl.plot(sol.t, q_des(time)[i,:].T, '--', label='commanded')
    pl.legend()
    pl.ylabel('joint '+str(i+1))
pl.xlabel('time (s)')
pl.suptitle(title)
pl.tight_layout()
pl.subplots_adjust(top=0.88)
pl.show()

# %% [markdown]
# # Problem #3

# If we make a "new" arm object, and call it a different variable name, we can use it
# for control, while we simulate with the original model (or vice versa). Here's a way
# to add some noise to our parameters:
percent_err = 0.10

# masses of each link with some error.
link_masses = [np.random.uniform(low = link_masses[0]*(1-percent_err), high = link_masses[0]*(1+percent_err)),
               np.random.uniform(low = link_masses[1]*(1-percent_err), high = link_masses[1]*(1+percent_err)),
               np.random.uniform(low = link_masses[2]*(1-percent_err), high = link_masses[2]*(1+percent_err))]

# defining three different centers of mass, one for each link
r_coms = [np.array([np.random.uniform(low = -0.1*(1+percent_err), high = -0.1*(1-percent_err)), 0, 0]),
          np.array([np.random.uniform(low = -0.1*(1+percent_err), high = -0.1*(1-percent_err)), 0, 0]),
          np.array([np.random.uniform(low = -0.1*(1+percent_err), high = -0.1*(1-percent_err)), 0, 0])]

# %%
