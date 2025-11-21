import numpy as np
import kinematics as kin
import transforms as tr
from visualization import VizScene

# these DH parameters are based on solutions from HW 3, if you
# pick a different set that still describe the robots accurately,
# that's great.
a_len = 0.5
d_len = 0.35

dh_part_a = [[0, d_len, 0., np.pi/2.0],
            [0, 0, a_len, 0], 
            [0, 0, a_len, 0]]

dh_part_b = [[0, d_len, 0., -np.pi/2.0],
            [0, 0, a_len, 0], 
            [np.pi/2.0, 0, 0, np.pi/2.0], 
            [np.pi/2.0, d_len*2, 0, -np.pi/2.0],
            [0, 0, 0, np.pi/2],
            [0, d_len*2, 0, 0]]

# jt_types_a = ['r', 'r', 'r']

# arm_a = kin.SerialArm(dh_part_a, jt=jt_types_a)

# q_a = [0,np.pi/2,0]
# jacobian_a = arm_a.jacob(q_a)
# print(jacobian_a)
# print("Condition #: ", str(np.linalg.cond(jacobian_a)))
# print("Rank: ", str(np.linalg.matrix_rank(jacobian_a)))

# viz = VizScene()
# viz.add_arm(arm_a, draw_frames=True)

# viz.update(qs=q_a)
# viz.hold()

# #Question 2
# #A force could be applied with infinite strength directly into the last arm link,
# #and into the axis of rotation for both the second and third joint


# jt_types_b = ['r', 'r', 'r', 'r', 'r', 'r']

# arm_b = kin.SerialArm(dh_part_b, jt=jt_types_b)

# q_b = [0,-np.pi/2,0,0,0,0]
# jacobian_b = arm_b.jacob(q_b)
# print(jacobian_b)
# print("Condition #: ", str(np.linalg.cond(jacobian_b)))
# print("Rank: ", str(np.linalg.matrix_rank(jacobian_b)))

# viz = VizScene()
# viz.add_arm(arm_b, draw_frames=True)

# viz.update(qs=q_b)
# viz.hold()


#Question 2
#A force could be applied with infinite strength directly into the last arm link

#%%
# #Question 3
# dh_3 = [[0, 1, 0, np.pi/2],
#         [0, 0, 1, -np.pi/2], 
#         [0, 0, 1, 0]]

# jt_types_3 = ['r', 'r', 'r']

# arm_3 = kin.SerialArm(dh_3, jt=jt_types_3)

# q_3 = [0,0,0]
# jacobian_3 = arm_3.jacob(q_3)

# # viz = VizScene()
# # viz.add_arm(arm_3, draw_frames=True)

# # viz.update(qs=q_3)
# # viz.hold()

# R_big = np.zeros((6,6))
# R_big[0:3, 0:3] = tr.rotx(np.pi/2)
# R_big[3:6, 3:6] = tr.rotx(np.pi/2)
# J_nn = R_big @ jacobian_3
# print(J_nn)
#%%
dh_4 = [[0,0,0,np.pi/2],
        [0,0,.4318,0],
        [0,.15,.02,-np.pi/2],
        [0,.4318,0,np.pi/2],
        [0,0,0,-np.pi/2],
        [0,0.4,0,0]]

jt_types_4 = ['r','r','r','r','r','r']

arm_4 = kin.SerialArm(dh_4, jt=jt_types_4)

jacob_4 = arm_4.jacob([0,0,0,0,0,0])

R_4 = [[0,0,1],
     [0,1,0],
     [-1,0,0]]
j_o_4 = arm_4.Z_shift(R_4, [np.pi/2,0,0]) @ jacob_4
print(j_o_4)