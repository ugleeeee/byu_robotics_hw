"""
Kinematics Module - Contains code for:
- Forward Kinematics, from a set of DH parameters to a serial linkage arm with callable forward kinematics
- Inverse Kinematics
- Jacobian

John Morrell, Jan 26 2022
Tarnarmour@gmail.com

modified by:
Marc Killpack, Sept 21, 2022 and Sept 21, 2023
"""

import numpy as np
import transforms as tr
from numpy.typing import NDArray
from typing import Callable, Iterable


# this is a convenience function that makes it easy to define a function that calculates
# "A_i(q_i)", given the DH parameters for link and joint "i" only.
def dh2A(dh: list[float], jt: str) -> Callable[[float], NDArray]:
    """
    Creates a function, A(q), that will generate a homogeneous transform T for a single
    joint/link given a set of DH parameters. A_i(q_i) represents the transform from link
    i-1 to link i, e.g. A1(q1) gives T_1_in_0. This follows the "standard" DH convention.

    :param list[float] dh: list of 4 dh parameters (single row from DH table) for the
        transform from link i-1 to link i, in the order [theta d a alpha] - THIS IS NOT
        THE CONVENTION IN THE BOOK!!! But it is the order of operations.
    :param str jt: joint type: 'r' for revolute joint, 'p' for prismatic joint
    :return A: a function of the corresponding joint angle, A(q), that generates a 4x4
        numpy array representing the homogeneous transform from one link to the next
    """
    # if joint is revolute implement correct equations here:
    if jt == 'r':
        # although A(q) is only a function of "q", the dh parameters are available to these next functions
        # because they are passed into the function above.

        def A(q: float) -> NDArray:
            # See eq. (2.52), pg. 64
            # TODO - complete code that defines the "A" or "T" homogenous matrix for a given set of DH parameters.
            # Do this in terms of the variables "dh" and "q" (so that one of the entries in your dh list or array
            # will need to be added to q).

            theta = dh[0] + q
            d = dh[1]
            a = dh[2]
            alpha = dh[3]

            cth = np.cos(theta)
            sth = np.sin(theta)
            cal = np.cos(alpha)
            sal = np.sin(alpha)

            return np.array(
                [[cth, -sth * cal, sth *sal, a * cth],
                    [sth, cth * cal, -cth * sal, a * sth],
                    [0, sal, cal, d],
                    [0, 0, 0, 1]])

    # if joint is prismatic implement correct equations here:
    else:
        def A(q: float) -> NDArray:
            # See eq. (2.52), pg. 64
            # TODO - complete code that defines the "A" or "T" homogenous matrix for a given set of DH parameters.
            # Do this in terms of the variables "dh" and "q" (so that one of the entries in your dh list or array
            # will need to be added to q).

            theta = dh[0]
            d = dh[1] + q
            a = dh[2]
            alpha = dh[3]

            cth = np.cos(theta)
            sth = np.sin(theta)
            cal = np.cos(alpha)
            sal = np.sin(alpha)

            return np.array(
                [[cth, -sth * cal, sth * sal, a * cth],
                    [sth, cth * cal, -cth * sal, a * sth],
                    [0, sal, cal, d],
                    [0, 0, 0, 1]])

    return A


class SerialArm:
    """
    SerialArm - A class designed to represent a serial link robot arm

    SerialArms have frames 0 to n defined, with frame 0 located at the first joint and
    aligned with the robot body frame, and frame n located at the end of link n.
    """

    def __init__(self, dh: list[list[float]], jt: list[str]|None=None,
                 base: NDArray=np.eye(4), tip: NDArray=np.eye(4),
                 joint_limits: NDArray|None=None):
        """
        arm = SerialArm(dh, jt, base=I, tip=I, joint_limits=None)

        :param list[list[float]] dh: n length list where each entry is another list of
            4 dh parameters: [theta d a alpha]
        :param list[str] | None jt: n length list of strings for joint types,
            'r' for revolute joint and 'p' for prismatic joint.
            If None, all joints are set to revolute.
        :param NDArray base: 4x4 numpy array representing SE3 transform from world or
            inertial frame to frame 0 (T_0_in_base)
        :param NDArray tip: 4x4 numpy array representing SE3 transform from frame n to
            tool frame or tip of robot (T_tip_in_n)
        :param NDArray | None joint_limits: 2*n array, min joint limit in 1st row then
            max joint limit in 2nd row (values in radians/meters).
            None for not implemented (these are only used in visualization).
        """
        self.dh = dh
        self.dh2a = dh2A
        self.n = len(dh)

        # we will use this list to store the A matrices for each set/row of DH parameters.
        self.transforms: list[Callable[[float], NDArray]] = []

        # assigning a joint type
        if jt is None:
            self.jt = ['r'] * self.n
        else:
            assert len(jt) == self.n, "Joint type list does not have the same size as dh param list!"
            self.jt = jt

        # using the code we wrote above to generate the function A(q) for each set of DH parameters
        for i in range(self.n):
            # TODO use the function definition above (dh2A), and the dh parameters and
            # joint type to make a function and then append that function to the
            # "transforms" list (use the versions from self because they have error checks).
            A = self.dh2a(self.dh[i], self.jt[i])
            self.transforms.append(A)

        # assigning the base, and tip transforms that will be added to the default DH transformations.
        self.base = base.copy()
        self.tip = tip.copy()
        self.qlim = joint_limits

        # calculating rough numbers to understand the workspace for drawing the robot
        self.reach = 0
        for dh in self.dh:
            self.reach += np.linalg.norm(dh[1:3])


    # You don't need to touch this function, but it is helpful to be able to "print"
    # a description about the robot that you make.
    def __str__(self):
        """
        This function just provides a nice interface for printing information about the arm.
        If we call "print(arm)" on an SerialArm object "arm", then this function gets called.
        See example in "main" below.
        """
        chars_per_col = 9
        dh_string = 'Serial Arm: DH Parameters\n'
        labels = ['θ', 'd', 'a', 'α', 'jt']
        cols = len(labels)
        dh_string += f'┌{"┬".join(["—"*chars_per_col for i in range(cols)])}┐\n'
        dh_string += f"|{''.join([f'{l.center(chars_per_col)}|' for l in labels])}\n"
        line = f"{'—'*chars_per_col}|"
        dh_string += f'|{line*cols}\n'
        for dh, jt in zip(self.dh, self.jt):
            row = [f'{val:.3f}'.rstrip('0') if isinstance(val,float) else f'{val}' for val in [*dh,jt]]
            dh_string += f"|{''.join([f'{str(s).center(chars_per_col)}|' for s in row])}\n"
        dh_string += f'└{"┴".join(["—"*chars_per_col for i in range(cols)])}┘\n'
        return dh_string


    def fk(self, q: Iterable[float], index: int|Iterable[int]|None=None,
           base: bool=False, tip: bool=False) -> NDArray:
        """
        T_n_in_0 = arm.fk(q, index=None, base=False, tip=False)

        Returns the transform from a specified frame to another given a set of
        joint angles q, the index of the starting and ending frames, and whether
        or not to include the base and tip transforms created in the constructor.

        :param Iterable[float] q: list or iterable of floats which represent the joint angles.
        :param int | Iterable[int] | None index: integer, list of two integers, or None.
            If an integer, it represents end_frame and start_frame is 0.
            If an iterable of two integers, they represent (start_frame, end_frame).
            If None, then start_frame is 0 and end_frame is n.
        :param bool base: specify whether to use the base transform (T_0_in_base) in the calculation.
            If start_frame is not 0, the frames do not line up and the base transform will not be used.
        :param bool tip: specify whether to use the tip transform (T_tip_in_n) in the calculation.
            If end_frame is not n, the frames do not line up and the tip transform will not be used.
        :return T: the 4 x 4 homogeneous transform between the specified frames.
        """
        ###############################################################################################
        # the following lines of code are data type and error checking. You don't need to understand
        # all of it, but it is helpful to keep.

        if not hasattr(q, '__getitem__'):
            q = [q]

        assert len(q) == self.n, "q must be the same size as the number of links!"

        if isinstance(index, int):
            start_frame = 0
            end_frame = index
        elif hasattr(index, '__getitem__'):
            start_frame = index[0]
            end_frame = index[1]
        elif index == None:
            start_frame = 0
            end_frame = self.n
        else:
            raise TypeError("Invalid index type!")

        assert 0 <= start_frame <= end_frame <= self.n, "Invalid index values!"
        ###############################################################################################
        ###############################################################################################

        # TODO - Write code to calculate the total homogeneous transform "T" based on variables stored
        # in "base", "tip", "start_frame", and "end_frame". Look at the function definition if you are
        # unsure about the role of each of these variables. This is mostly easily done with some if/else
        # statements and a "for" loop to add the effect of each subsequent A_i(q_i). But you can
        # organize the code any way you like.
        # print(self.transforms)

        if(index==None):
            if(base==True):
                T_out = self.base
            else:
                T_out = np.eye(4)

            for i in range(len(self.transforms)):
                func = self.transforms[i]
                T_out = T_out @ func(q[i])

            if(tip==True):
                T_out = T_out @ self.tip
        
        
        elif(type(index)==int): # For creating T from 0 to int
            if(base==True):
                T_out = self.base
            else:
                T_out = np.eye(4)
            for i in range(index):
                func = self.transforms[i]
                T_out = T_out @ func(q[i])
            
        elif(len(index)==2):
            start = index[0]
            end = index[1]
            T_out = np.eye(4)
            for i in range(end - start):
                func = self.transforms[start + i]
                print(func(q[i]))
                T_out = T_out @ func(q[i])
            
        return T_out
    

    def jacob(self, q, index=None, base=False, tip=False):
        """
        J = arm.jacob(q)
        Description: 
        Returns the geometric jacobian for the end effector frame of the arm in a given configuration

        Parameters:
        q - list or numpy array of joint positions
        index - integer, which joint frame at which to calculate the Jacobian

        Returns:
        J - numpy matrix 6xN, geometric jacobian of the robot arm
        """


        if index is None:
            index = self.n
        elif index > self.n:
            print("WARNING: Index greater than number of joints!")
            print(f"Index: {index}")

        J = np.zeros((6, self.n))
        Te = self.fk(q, index, base=base, tip=tip)
        pe = Te[0:3, 3]

        for i in range(index):
            # check if joint is revolute
            if self.jt[i] == 'r':
                T = self.fk(q, i, base=base, tip=tip)
                z_axis = T[0:3, 2]
                p = T[0:3, 3]
                J[0:3, i] = np.cross(z_axis, pe - p, axis=0)
                J[3:6, i] = z_axis
                
            # if not assume joint is prismatic
            else:
                T = self.fk(q, i, base=base, tip=tip)
                z_axis = T[0:3, 2]
                J[0:3, i] = z_axis
                J[3:6, i] = np.zeros_like(z_axis)

        return J
    
    # insert this function into your SerialArm class and complete it.
    # Please keep the function definition, and what it returns the same.
    def ik_position(self, target: NDArray, q0: list[float]|NDArray|None=None,
                    method: str='J_T', force: bool=True, tol: float=1e-4,
                    K: NDArray=None, kd: float=0.001, max_iter: int=100,
                    debug: bool=False, debug_step: bool=False
                    ) -> tuple[NDArray, NDArray, int, bool]:
        """
        qf, error_f, iters, converged = arm.ik_position(target, q0, 'J_T', K=np.eye(3))

        Computes the inverse kinematics solution (position only) for a given target
        position using a specified method by finding a set of joint angles that
        place the end effector at the target position without regard to orientation.

        :param NDArray target: 3x1 numpy array that defines the target location.
        :param list[float] | NDArray | None q0: list or array of initial joint positions,
            defaults to q0=0 (which is often a singularity - other starting positions
            are recommended).
        :param str method: select which IK algorithm to use. Options include:
            - 'pinv': damped pseudo-inverse solution, qdot = J_dag * e * dt, where
            J_dag = J.T * (J * J.T + kd**2)^-1
            - 'J_T': jacobian transpose method, qdot = J.T * K * e
        :param bool force: specify whether to attempt to solve even if a naive reach
            check shows the target is outside the reach of the arm.
        :param float tol: tolerance in the norm of the error in pose used as
            termination criteria for while loop.
        :param NDArray K: 3x3 numpy array. For both pinv and J_T, K is the positive
            definite gain matrix.
        :param float kd: used in the pinv method to make sure the matrix is invertible.
        :param int max_iter: maximum attempts before giving up.
        :param bool debug: specify whether to plot the intermediate steps of the algorithm.
        :param bool debug_step: specify whether to pause between each iteration when debugging.

        :return qf: 6x1 numpy array of final joint values. If IK fails to converge
            within the max iterations, the last set of joint angles is still returned.
        :return error_f: 3x1 numpy array of the final positional error.
        :return iters: int, number of iterations taken.
        :return converged: bool, specifies whether the IK solution converged within
            the max iterations.
        """
        ###############################################################################################
        # the following lines of code are data type and error checking. You don't need to understand
        # all of it, but it is helpful to keep.
        if isinstance(q0, np.ndarray):
            q = q0
        elif q0 == None:
            q = np.array([0.0]*self.n)
        elif isinstance(q0, list):
            q = np.array(q0)
        else:
            raise TypeError("Invlid type for initial joint positions 'q0'")

        # Try basic check for if the target is in the workspace.
        # Maximum length of the arm is sum(sqrt(d_i^2 + a_i^2)), distance to target is norm(A_t)
        target_distance = np.linalg.norm(target)
        target_in_reach = target_distance <= self.reach
        if not force:
            assert target_in_reach, "Target outside of reachable workspace!"
        if not target_in_reach:
            print("Target out of workspace, but finding closest solution anyway")

        assert isinstance(K, np.ndarray), "Gain matrix 'K' must be provided as a numpy array"
        ###############################################################################################
        ###############################################################################################

        # you may want to define some functions here to help with operations that you will
        # perform repeatedly in the while loop below. Alternatively, you can also just define
        # them as class functions and use them as self.<function_name>.

        # for example:
        # def get_error(q):
        #     cur_position =
        #     e =
        #     return e

        iters = 0
        while np.linalg.norm(error) > tol and iters < max_iter:

        # In this while loop you will update q for each iteration, and update, then
        # your error to see if the problem has converged. You may want to print the error
        # or the "count" at each iteration to help you see the progress as you debug.
        # You may even want to plot an arm initially for each iteration to make sure
        # it's moving in the right direction towards the target.



        # when "while" loop is done, return the relevant info.
        return q, error, iters, iters < max_iter




if __name__ == "__main__":
    from visualization import VizScene
    np.set_printoptions(precision=4, suppress=True)

    # Defining a table of DH parameters where each row corresponds to another joint.
    # The order of the DH parameters is [theta, d, a, alpha] - which is the order of operations.
    # The symbolic joint variables "q" do not have to be explicitly defined here.
    # This is a two link, planar robot arm with two revolute joints.
    dh = [[0, 0, 0.3, 0],
          [0, 0, 0.3, 0]]

    # make robot arm (assuming all joints are revolute)
    arm = SerialArm(dh)

    # defining joint configuration
    q = [np.pi/4.0, np.pi/4.0]  # 45 degrees and 45 degrees

    # show an example of calculating the entire forward kinematics
    Tn_in_0 = arm.fk(q)
    print("Tn_in_0:\n", Tn_in_0, "\n")

    # show an example of calculating the kinematics between frames 0 and 1
    T1_in_0 = arm.fk(q, index=[0,1])
    print("T1_in 0:\n", T1_in_0, "\n")

    # showing how to use "print" with the arm object
    print(arm)

    # now visualizing the coordinate frames that we've calculated
    viz = VizScene()

    viz.add_frame(arm.base, label='base')
    viz.add_frame(Tn_in_0, label="Tn_in_0")
    viz.add_frame(T1_in_0, label="T1_in_0")

    viz.hold()
