"""
dynamics Module - Contains code for:
- Dynamic SerialArm class
- RNE Algorithm
- Euler - Lagrange formulation

John Morrell, Jan 28 2022
Tarnarmour@gmail.com

modified by:
Marc Killpack, October 25, 2022
               Nov. 20, 2023
"""

import numpy as np
from kinematics import SerialArm


class SerialArmDyn(SerialArm):
    """
    SerialArmDyn class represents serial arms with dynamic properties and is used to calculate forces, torques, accelerations,
    joint forces, etc. using the Newton-Euler and Euler-Lagrange formulations. It inherits from the previously defined kinematic
    robot arm class "SerialArm".
    """

    def __init__(self,
                 dh,
                 jt=None,
                 base=np.eye(4),
                 tip=np.eye(4),
                 joint_limits=None,
                 mass=None,
                 r_com=None,
                 link_inertia=None,
                 motor_inertia=None,
                 joint_damping=None):

        SerialArm.__init__(self, dh, jt, base, tip, joint_limits)
        self.mass = mass
        self.r_com = r_com
        self.link_inertia = link_inertia
        self.motor_inertia = motor_inertia
        if joint_damping is None:
            self.B = np.zeros((self.n, self.n))
        else:
            self.B = np.diag(joint_damping)

    def rne(self, q, qd, qdd,
            Wext=np.zeros((6,)),
            g=np.zeros((3,)),
            omega_base=np.zeros((3,)),
            alpha_base=np.zeros((3,)),
            v_base=np.zeros((3,)),
            acc_base=np.zeros((3,))):

        """
        tau, W = RNE(q, qd, qdd):
        returns the torque in each joint (and the full wrench at each joint) given the joint configuration, velocity, and accelerations
        Args:
            q:
            qd:
            qdd:

        Returns:

        We start with the velocity and acceleration of the base frame, v0 and a0, and the joint positions, joint velocities,
        and joint accelerations (q, qd, qdd).

        For each joint, we find the new angular velocity, w_i = w_(i-1) + z * qdot_(i-1)
        v_i = v_(i-1) + w_i x r_(i-1, com_i)


        if motor inertia is None, we don't consider it. Solve for now without motor inertia. The solution will provide code for motor inertia as well.
        """

        omegas = []
        alphas = []
        v_ends = []
        v_coms = []
        acc_ends = []
        acc_coms = []


        # First we'll define some additional terms that we'll use in each iteration of the algorithm
        Rs = []  # List of Ri-1_i, rotation from i-1 to i in the i-1 frame
        R0s = []  # List of R0_i, rotation from 0 to i in the 0 frame
        rp2cs = []  # List of pi-1_i-1_i, vector from i-1 to i frame in frame i-1
        forces = []  # List of fi_i, force applied to link i at frame i-1, expressed w.r.t frame i
        moments = []  # List of Mi_i, moment applied to link i expressed w.r.t frame i
        rp2coms = []  # List of r_i_i-1,com, the vector from the origin of frame i-1 to the COM of link i in the i frame
        zaxes = []  # List of z axis of frame i-1, expressed in frame i

        # Lets generate all of the needed transforms now to simplify code later and save unnecessary calls to self.fk
        for i in range(self.n):
            # print(lenq)
            # print(self.n)
            T = self.fk(q, [i, i+1])  # Find the transform from link i to link i+1
            # print(T)
            R = T[0:3, 0:3]
            p = T[0:3, 3]

            Rs.append(R)                              # these are the R's for each link from frame i-1 to i
            rp2cs.append(R.T @ p)                     # these are the vectors from i-1 to i, but in frame i
            rp2coms.append(R.T @ p + self.r_com[i])   # these are the vectors from i-1 to COM_i, but in frame i
            zaxes.append(R.T[0:3, 2])                 # these are the z-axes for joint 1 -> joint n

            R0 = self.fk(q, i+1)[0:3, 0:3]            # the R describing frame i in the base frame
            R0s.append(R0)


        ## Solve for needed angular velocities, angular accelerations, and linear accelerations
        ## If helpful, you can define a function to call here so that you can debug the output more easily.

        # setting up the variables for the first time through the kinematics loop.
        w_prev = omega_base
        alph_prev = alpha_base
        a_prev = acc_base

        for i in range(0, self.n):
            # Find kinematics of the current link
            if self.jt[i] == 'r':
                w_cur = Rs[i].T @ w_prev + zaxes[i]*qd[i]
                alph_cur = Rs[i].T @ alph_prev + zaxes[i]*qdd[i] + np.cross(w_cur, zaxes[i]) * qd[i]
                a_com = Rs[i].T @ a_prev + np.cross(alph_cur, rp2coms[i]) + np.cross(w_cur, np.cross(w_cur, rp2coms[i]))
                a_end = Rs[i].T @ a_prev + np.cross(alph_cur, rp2cs[i]) + np.cross(w_cur, np.cross(w_cur, rp2cs[i]))
            else:
                print("you need to implement kinematic equations for joint type:\t", self.jt[i])

            # update "prev" values for next iteration
            w_prev = w_cur
            alph_prev = alph_cur
            a_prev = a_end

            # Append values to our lists
            omegas.append(w_cur)
            alphas.append(alph_cur)
            acc_coms.append(a_com)
            acc_ends.append(a_end)

        ## Now solve Kinetic equations by starting with forces at last link and going backwards
        ## If helpful, you can define a function to call here so that you can debug the output more easily.
        Wrenches = np.zeros((6, self.n,))  # these are the total forces and torques at each joint,
                                           # this includes both motor forces, and forces supported
                                           # by the structure
        tau = np.zeros((self.n,)) # these are the motor torques only, expressed in the appropriate frame

        # setting up variables for the first time through the for loop
        f_prev = Wext[0:3]                      # this is the external force in the "tip" tool frame
        M_prev = Wext[3:]                       # this is the external torque in the "tip" tool frame
        R_ip1_in_frame_i = self.tip[0:3, 0:3]   # this is the rotation describing the tip, in the nth frame

        for i in range(self.n - 1, -1, -1):  # Index from n-1 to 0
            R_0_in_frame_i = R0s[i].T  # this is giving a rotation describing frame 0, expressed in frame i
            g_cur = R_0_in_frame_i @ g # Convert the gravity to frame "i"

            # Sum of forces and mass * acceleration to find forces
            # m*a = f_cur - f_prev + m*g --> f_cur = m * (a - g) + f_prev
            f_cur = R_ip1_in_frame_i @ f_prev + self.mass[i] * (acc_coms[i] - g_cur)

            # Using the sum of moments and d/dt(angular momentum) to find moment at joint
            # Be very careful with the r x f terms here; easy to mess up
            M_cur = self.link_inertia[i] @ alphas[i] \
                + np.cross(omegas[i], self.link_inertia[i] @ omegas[i]) \
                + R_ip1_in_frame_i @ M_prev \
                + np.cross(self.r_com[i], - (R_ip1_in_frame_i @ f_prev)) \
                + np.cross(rp2coms[i], f_cur)


            # store "prev" values for next iterations
            R_ip1_in_frame_i = Rs[i] # need R from i+1 to i, just setting this up for next iteration
            f_prev = f_cur
            M_prev = M_cur

            # store the total force and torque at joint i in the Wrench output.
            Wrenches[0:3,i] = f_cur
            Wrenches[3:, i] = M_cur

        # now calculate the joint force or torque in the frame co-located with the joint
        for i in range(self.n):
            if self.jt[i] == 'r':
                # this is the same as doing R_(i-1)^i @ tau_i^i and taking only the third element.
                tau[i] = zaxes[i] @ Wrenches[3:, i]
            else:
                print("you need to implement generalized force calculation for joint type:\t", self.jt[i])

        return tau, Wrenches


    def get_M(self, q):
        M = np.zeros((self.n, self.n))

        # calculating the mass matrix by iterating through RNE "n" times, and changing the location of the "1" entry in qdd
        for i in range(self.n):
            qdd = np.zeros((self.n, ))
            qdd[i] = 1
            tau, _ = self.rne(q, np.zeros((self.n, )), qdd)
            M[:,i] = tau

        return M

    def get_C(self, q, qd):
        Cq_dot, _ = self.rne(q, qd, np.zeros((self.n,)))

        return Cq_dot

    def get_G(self, q, g):
        zeros = np.zeros((self.n,))
        G, _ = self.rne(q, zeros, zeros, g = g)

        return G


if __name__ == '__main__':

    ## this just gives an example of how to define a robot, this is a planar 3R robot.
    dh = [[0, 0, 0.4, 0],
        [0, 0, 0.4, 0],
        [0, 0, 0.4, 0]]

    joint_type = ['r', 'r', 'r']

    link_masses = [1, 1, 1]

    # defining three different centers of mass, one for each link
    r_coms = [np.array([-0.2, 0, 0]), np.array([-0.2, 0, 0]), np.array([-0.2, 0, 0])]

    link_inertias = []
    for i in range(len(joint_type)):
        iner = 0.01
        # this inertia tensor is only defined as having Izz non-zero
        link_inertias.append(np.array([[0, 0, 0], [0, 0, 0], [0, 0, iner]]))


    arm = SerialArmDyn(dh,
                       jt=joint_type,
                       mass=link_masses,
                       r_com=r_coms,
                       link_inertia=link_inertias)

    # once implemented, you can call arm.RNE and it should work.
    q = [np.pi/4.0]*3
    qd = [np.pi/6.0, -np.pi/4.0, np.pi/3.0]
    qdd = [-np.pi/6.0, np.pi/3.0, np.pi/6.0]
    arm.rne(q, qd, qdd, g=np.array([0, -9.81, 0]))