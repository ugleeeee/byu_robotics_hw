# %% [markdown]
# # Midterm 2025
# * Copy this file to your homework workspace to have access to your other kinematic and visualization functions

# %%
# To test your setup, after defining the robot arm as described below, (but nothing else)
# you can run this file directly to make sure it is plotting the arm, obstacle, and goal 
# as expected. 

import kinematics as kin  #this is your kinematics file that you've been developing all along
from visualization import VizScene #this is the visualization file you've been using for homework
import time
import numpy as np


# Define your kinematics and an "arm" variable here using DH parameters so they
# are global variables that are available in your function below:

dh = np.array([[np.pi/2, 4, 0, np.pi/2],
               [np.pi/6, 0, 0, np.pi/2],
               [0, 4, 0, np.pi/2],
               [np.pi/6, 0, 2, np.pi/2]])
arm = kin.SerialArm(dh)


# let's also plot robot to make sure it matches what we think it should
# (this will look mostly like the pictures on part 1 if your DH parameters
# are correct)
# viz_check = VizScene()
# viz_check.add_arm(arm, joint_colors=[np.array([0.95, 0.13, 0.13, 1])]*arm.n)
# viz_check.update(qs = [[0, 0, 0, 0]])
# viz_check.hold()


def compute_robot_path(q_init, goal, obst_location, obst_radius):
      # this can be similar to your IK solution for HW 6, but will require modifications
      # to make sure you avoid the obstacle as well.
      K1 = np.eye(3) * .01

      ik_info = ik_info = arm.ik_with_obs(target = goal, q0 = q_init, method='J_T', K = K1, max_iter=300,obs_pos=obst_location, obs_rad=obst_radius)

      q_s = ik_info[4]
      # print(q_s)

      return q_s

if __name__ == "__main__":

      # if your function works, this code should show the goal, the obstacle, and your robot moving towards the goal.
      # Please remember that to test your function, I will change the values below to see if the algorithm still works.
      q_0 = [0, 0, 0, 0]
      goal = [2,4,3]
      obst_position = [2,2,2]
      obst_rad = 1.0 

      q_ik_slns = compute_robot_path(q_0, goal, obst_position, obst_rad)

      # depending on how you store q_ik_slns inside your function, you may need to change this for loop
      # definition. However if you store q as I've done above, this should work directly.
      viz = VizScene()
      viz.add_arm(arm, joint_colors=[np.array([0.95, 0.13, 0.13, 1])]*arm.n)
      viz.add_marker(goal, radius=0.1)
      viz.add_obstacle(obst_position, rad=obst_rad)
      for q in q_ik_slns:
            viz.update(qs=[q])

            # if your step in q is very small, you can shrink this time, or remove it completely to speed up your animation
            time.sleep(0.02)
      viz.hold()
