# %% [markdown]
# # Homework 5
# * Copy the contents of the file "hw05_jacobian.py" into the SerialArm class definition in "kinematics.py".
# * Now complete the blank sections that will allow you to calculate the Geometric Jacobian for any robot based on a DH parameter description.
# * After completion, check that your answers match the the answers for the Jacobian test below.

# %%
import kinematics as kin
import transforms as tr
from visualization import VizScene
import numpy as np
np.set_printoptions(precision=4, suppress=True)


# %% [markdown]
# # Problem 1:

# Check your implementation of "jacob" using the DH parameters shown below. For the given vectors of q's, you should get the following:
# $$ q = \left[\begin{matrix}0 \\ 0 \\ 0 \\0 \end{matrix}\right]$$
# $$ J = \left[\begin{matrix}0 & 0 & 0 & 0\\0.5 & 0 & 0.1 & -1.0\\0 & 0.3 & 0 & 0\\0 & 0 & 0 & 0\\0 & -1.0 & 0 & 0\\1.0 & 0 & 1.0 & 0\end{matrix}\right]$$

# While for $$ q = \left[\begin{matrix}\pi/4 \\ \pi/4 \\ \pi/4 \\ 0.10 \end{matrix}\right]$$
# $$ J = \left[\begin{matrix}-0.3121 & -0.1707 & -0.1 & 0.8536\\0.3121 & -0.1707 & 0.1 & -0.1464\\0 & 0.2414 & -6.939 \cdot 10^{-18} & 0.5\\0 & 0.7071 & -0.5 & 0\\0 & -0.7071 & -0.5 & 0\\1.0 & 0 & 0.7071 & 0\end{matrix}\right] $$


# %%
dh = [[0, 0, 0.2, np.pi/2.0],
      [0, 0, 0.2, -np.pi/2.0],
      [0, 0, 0.1, np.pi/2.0],
      [0, 0, 0.0, 0.0]]

# An example of defining joint types which we may not have done yet.
# The 4th joint, and 4th row of the DH parameters correspond to a prismatic joint.
jt_types = ['r', 'r', 'r', 'p']

# notice how "jt_types" can be passed as an argument into "SerialArm"
arm = kin.SerialArm(dh, jt=jt_types)

# defining two different sets of joint angles
q_set1 = [0, 0, 0, 0]
q_set2 = [np.pi/4, np.pi/4, np.pi/4, 0.10]

# calculating two different jacobians for the two different joint configurations.
J1 = arm.jacob(q_set1)
J2 = arm.jacob(q_set2)

print("from first set of q's, J is:")
print(J1)

print("now look at the configuration of the arm for q_set1 to understand J")

# making a visualization
viz = VizScene()

# adding a SerialArm to the visualization, and telling it to draw the joint frames.
viz.add_arm(arm, draw_frames=True)

# setting the joint angles to draw
viz.update(qs=[q_set1])

viz.hold()


print("from second set of q's, J is:")
print(J2)

# updating position of the arm in visualization
viz = VizScene()
viz.add_arm(arm, draw_frames=True)
viz.update(qs=[q_set2])

print("now look at the configuration of the arm for q_set2 to understand J")

viz.hold()


# %%
# 1c - All of the zeros in the first two columns are
# a result of there only being revolute motion and that
# the rotations were 90 degrees

#%%
# Problem 2
# J_tip = np.array([[0,0,0],
#                   [0,0,0],
#                   [0,0,D1],
#                   [0,Sq,0],
#                   [0,-Cq,0],
#                   [1,0,1]])

# %%
dh = [[0, 0.3, 0, np.pi/2.0],
      [0, 0, 0.3, 0]]

# An example of defining joint types which we may not have done yet.
# The 4th joint, and 4th row of the DH parameters correspond to a prismatic joint.
jt_types = ['r', 'r']

# notice how "jt_types" can be passed as an argument into "SerialArm"
arm = kin.SerialArm(dh, jt=jt_types)

# defining two different sets of joint angles
q_set1 = [0, 0]
q_set2 = [np.pi/3, np.pi/5]

# calculating two different jacobians for the two different joint configurations.
J1 = arm.jacob(q_set1)
J2 = arm.jacob(q_set2)

print("from first set of q's, J is:")
print(J1)

print("now look at the configuration of the arm for q_set1 to understand J")

# making a visualization
viz = VizScene()

# adding a SerialArm to the visualization, and telling it to draw the joint frames.
viz.add_arm(arm, draw_frames=True)

# setting the joint angles to draw
viz.update(qs=[q_set1])

viz.hold()


print("from second set of q's, J is:")
print(J2)

# updating position of the arm in visualization
viz = VizScene()
viz.add_arm(arm, draw_frames=True)
viz.update(qs=[q_set2])

print("now look at the configuration of the arm for q_set2 to understand J")

viz.hold()
# %%
# Problem 3
dh = np.array([[0,0,1,0],
               [0,0,1,0]])

# jacobian = np.array([[0,0,0],
#                      [0,0,0],
#                      [0,0,0],
#                      [Sq1,Sq2,-Sq3],
#                      [-Cq1,-Cq3,-Cq3],
#                      [0,0,0]])

jt_types = ['r', 'r']

arm = kin.SerialArm(dh, jt=jt_types)

q_set1 = [0, 0]

J1 = arm.jacob(q_set1)

print(J1)

#%%

import kinematics as kin
import transforms as tr
from visualization import VizScene
import numpy as np

dh = np.array([[0,0,0,-1*np.pi/2],
               [0,0.154,0,np.pi/2],
               [0,0.25,0,0],
               [-1*np.pi/2,0,0,-1*np.pi/2],
               [-1*np.pi/2,0,0,np.pi/2],
               [np.pi/2,0.263,0,0]])

jt_types = ['r', 'r', 'p', 'r', 'r', 'r']

arm = kin.SerialArm(dh, jt=jt_types, tip=tr.se3(tr.roty(-1*np.pi/2)))

q_set1 = [0, 0, 0, 0, 0, 0]
q_set2 = [0, 0, 0.1, 0, 0, 0]

J1 = arm.jacob(q_set1)
print("J1")
print(J1)
J2 = arm.jacob(q_set2)
print("J2")
print(J2)

viz = VizScene()
viz.add_arm(arm, draw_frames=True)
viz.update(qs=[q_set1])
viz.hold()

viz = VizScene()
viz.add_arm(arm, draw_frames=True)
viz.update(qs=[q_set2])
viz.hold()
# %%
