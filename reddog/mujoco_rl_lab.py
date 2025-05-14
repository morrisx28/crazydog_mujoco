import time
import os

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import argparse
from scipy.spatial.transform import Rotation as R

NUM_MOTOR = 12

# def get_gravity_orientation(quaternion):
#     qw = quaternion[0]
#     qx = quaternion[1]
#     qy = quaternion[2]
#     qz = quaternion[3]

#     gravity_orientation = np.zeros(3)

#     gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
#     gravity_orientation[1] = -2 * (qz * qy + qw * qx)
#     gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

#     return gravity_orientation

def quat_rotate_inverse(q, v):
    """
    Rotate a vector by the inverse of a quaternion.
    Direct translation from the PyTorch version to NumPy.
    
    Args:
        q: The quaternion in (w, x, y, z) format. Shape is (..., 4).
        v: The vector in (x, y, z) format. Shape is (..., 3).
        
    Returns:
        The rotated vector in (x, y, z) format. Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    
    # Equivalent to (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    term1 = 2.0 * np.square(q_w) - 1.0
    term1_expanded = np.expand_dims(term1, axis=-1)
    a = v * term1_expanded
    
    # Equivalent to torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    q_w_expanded = np.expand_dims(q_w, axis=-1)
    b = np.cross(q_vec, v) * q_w_expanded * 2.0
    
    # Equivalent to the torch.bmm or torch.einsum operations
    # This calculates the dot product between q_vec and v
    dot_product = np.sum(q_vec * v, axis=-1)
    dot_product_expanded = np.expand_dims(dot_product, axis=-1)
    c = q_vec * dot_product_expanded * 2.0
    
    return a - b + c

def get_gravity_orientation(quaternion):
    """
    Get the gravity vector in the robot's base frame.
    Uses the exact algorithm from your PyTorch code.
    
    Args:
        quaternion: Quaternion in (w, x, y, z) format.
        
    Returns:
        3D gravity vector in the robot's base frame.
    """
    # Ensure quaternion is a numpy array
    quaternion = np.array(quaternion)
    
    # Standard gravity vector in world frame (pointing down)
    gravity_world = np.array([0, 0, -1])
    
    # Handle both single quaternion and batched quaternions
    if quaternion.shape == (4,):
        quaternion = quaternion.reshape(1, 4)
        gravity_world = gravity_world.reshape(1, 3)
        result = quat_rotate_inverse(quaternion, gravity_world)[0]
    else:
        gravity_world = np.broadcast_to(gravity_world, quaternion.shape[:-1] + (3,))
        result = quat_rotate_inverse(quaternion, gravity_world)
    
    return result


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # policy_path = "/home/csl/robot_model/crazydog_urdf/model/biped_wheel_2dof_policy.pt"
        policy_path = config["policy_path"]
        policy = torch.jit.load(policy_path)
        xml_path = config["xml_path"]
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)
        initial_pos = np.array(config["initial_pos"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

        animation = config["animation"]
        max_step = config["max_step"]
        loop_start = config["loop_start"]

        lin_vel_activation = config["lin_vel_activation"]

    target_dof_pos = default_angles.copy()
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    base_body_id = 1

    # Create a directory to save plots
    plot_dir = "simulation_plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Record data
    lin_vel_data_list = []
    ang_vel_data_list = []
    gravity_b_list = []
    joint_pos_list = [] # Added
    joint_vel_list = []
    action_list = []
    joint_torque_list = [] # Added

    counter = 0
    current_step = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        mujoco.mj_resetDataKeyframe(m, d, 0)
        viewer.sync()
        print('key frame reset')
        # time.sleep(2)
        step_start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
    
            tau = pd_control(target_dof_pos, d.sensordata[:NUM_MOTOR], kps, np.zeros(12), d.sensordata[NUM_MOTOR:NUM_MOTOR + NUM_MOTOR], kds)

            d.ctrl[:] = tau

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            # com_base = calculate_com_in_base_frame(m, d, base_body_id)
            # print("Center of Mass in Base Coordinates:", com_base)

            counter += 1
            print("Counter:", counter)
            if counter % control_decimation == 0 and counter > 0:

                # create observation
                qpos = d.sensordata[:12]
                qvel = d.sensordata[12:24]
                imu_quat = d.sensordata[36:40] # (w, x, y, z)
                lin_vel_I = d.sensordata[49:52] # World frame
                ang_vel_I = d.sensordata[52:55] # World frame
                
                # Transform velocities to base link frame
                lin_vel_B = quat_rotate_inverse(imu_quat, lin_vel_I)
                ang_vel_B = quat_rotate_inverse(imu_quat, ang_vel_I)

                gravity_b = get_gravity_orientation(imu_quat)
                # print(gravity_b)
                cmd_vel = np.array(config["cmd_init"], dtype=np.float32)

                if lin_vel_activation == True:
                    obs[:3] = lin_vel_B * lin_vel_scale  # Linear velocity in base frame
                    obs[3:6] = ang_vel_B * ang_vel_scale  # Angular velocity in base frame
                    obs[6:9] = gravity_b  # <framequat name="imu_quat" objtype="site" objname="imu" /> with gravity [0,0,-1]
                    obs[9:12] = cmd_vel * cmd_scale  
                    obs[12:24] = (qpos - default_angles) * dof_pos_scale  # jointpos
                    obs[24:36] = qvel * dof_vel_scale  # jointvel
                    obs[36:48] = action
                else:
                    obs[:3] = ang_vel_B * ang_vel_scale  # Angular velocity in base frame
                    obs[3:6] = gravity_b  # <framequat name="imu_quat" objtype="site" objname="imu" /> with gravity [0,0,-1]
                    obs[6:9] = cmd_vel * cmd_scale  
                    obs[9:21] = (qpos - default_angles) * dof_pos_scale  # jointpos
                    obs[21:33] = qvel * dof_vel_scale  # jointvel
                    obs[33:45] = action
                    
                if animation == True:
                    obs[48] = current_step / (max_step-1)
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # print("action :", action)

                # transform action to target_dof_pos
                if counter < 0:
                    target_dof_pos = initial_pos
                else:
                    target_dof_pos = action * action_scale + default_angles
                    ## Record Data ##
                    lin_vel_data_list.append(lin_vel_B * lin_vel_scale) # Log base frame linear velocity
                    ang_vel_data_list.append(ang_vel_B * ang_vel_scale) # Log base frame angular velocity
                    gravity_b_list.append(gravity_b)
                    joint_pos_list.append(qpos.copy()) # Added
                    joint_torque_list.append(tau.copy()) # Added
                    joint_vel_list.append(qvel * dof_vel_scale)
                    action_list.append(target_dof_pos)
                    ###
                    current_step += 1
                    if current_step >= max_step:
                        current_step = loop_start
                # target_dof_pos = action * action_scale + default_angles
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            step_start = time.time()
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # Plot the collected data after the simulation ends
    fig1 = plt.figure(figsize=(10, 12)) # Adjusted figure size for 3 rows

    # Linear Velocity
    plt.subplot(3, 1, 1)
    labels = ['Lin Vel X', 'Lin Vel Y', 'Lin Vel Z']
    for i in range(3):
        plt.plot([step[i] for step in lin_vel_data_list], label=labels[i])
    plt.title("Linear Velocity (Base Frame)", fontsize=10, pad=10)
    plt.xlabel("Time Step")
    plt.ylabel("Velocity (scaled)")
    plt.legend()
    plt.grid(True)

    # Angular Velocity
    plt.subplot(3, 1, 2)
    labels = ['Ang Vel X', 'Ang Vel Y', 'Ang Vel Z']
    for i in range(3):
        plt.plot([step[i] for step in ang_vel_data_list], label=labels[i])
    plt.title("Angular Velocity (Base Frame)", fontsize=10, pad=10)
    plt.xlabel("Time Step")
    plt.ylabel("Velocity (scaled)")
    plt.legend()
    plt.grid(True)

    # Projected Gravity
    plt.subplot(3, 1, 3)
    labels = ['Gravity X', 'Gravity Y', 'Gravity Z']
    for i in range(3):
        plt.plot([step[i] for step in gravity_b_list], label=labels[i])
    plt.title("Projected Gravity (Base Frame)", fontsize=10, pad=10)
    plt.xlabel("Time Step")
    plt.ylabel("Vector Component")
    plt.legend()
    plt.grid(True)

    # REMOVED Joint Positions, Joint Velocities, and Joint Torques subplots from fig1

    plt.tight_layout(pad=2.0) # Add padding between subplots
    fig1_path = os.path.join(plot_dir, "simulation_summary.png")
    plt.savefig(fig1_path)
    print(f"Saved summary plot to {fig1_path}")
    plt.close(fig1) # Close the figure to free memory
    # plt.show() # Commented out show

    # Define joint indices based on sensor order in go2.xml
    hip_indices = [0, 1, 2, 3]   # FL, FR, RL, RR
    thigh_indices = [4, 5, 6, 7] # FL, FR, RL, RR
    calf_indices = [8, 9, 10, 11] # FL, FR, RL, RR
    joint_names = ["FL", "FR", "RL", "RR"] # Corrected order to match indices

    # Figure for Hip Joints
    fig_hip = plt.figure(figsize=(15, 8))
    plt.suptitle("Hip Joint Actions (--) vs. Positions (-)", fontsize=12)
    for i, joint_idx in enumerate(hip_indices):
        plt.subplot(2, 2, i + 1)
        # Assuming action_list also follows the sensor order
        plt.plot([step[joint_idx] for step in action_list], '--', label=f"{joint_names[i]} Hip Action")
        plt.plot([step[joint_idx] for step in joint_pos_list], '-', label=f"{joint_names[i]} Hip Pos")
        plt.title(f"{joint_names[i]} Hip Joint")
        plt.xlabel("Time Step")
        plt.ylabel("Angle (rad)")
        plt.legend()
        plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    fig_hip_path = os.path.join(plot_dir, "hip_joints_action_vs_pos.png")
    plt.savefig(fig_hip_path)
    print(f"Saved hip joints plot to {fig_hip_path}")
    plt.close(fig_hip)
    # plt.show() # Commented out show

    # Figure for Thigh Joints
    fig_thigh = plt.figure(figsize=(15, 8))
    plt.suptitle("Thigh Joint Actions (--) vs. Positions (-)", fontsize=12)
    for i, joint_idx in enumerate(thigh_indices):
        plt.subplot(2, 2, i + 1)
        # Assuming action_list also follows the sensor order
        plt.plot([step[joint_idx] for step in action_list], '--', label=f"{joint_names[i]} Thigh Action")
        plt.plot([step[joint_idx] for step in joint_pos_list], '-', label=f"{joint_names[i]} Thigh Pos")
        plt.title(f"{joint_names[i]} Thigh Joint")
        plt.xlabel("Time Step")
        plt.ylabel("Angle (rad)")
        plt.legend()
        plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_thigh_path = os.path.join(plot_dir, "thigh_joints_action_vs_pos.png")
    plt.savefig(fig_thigh_path)
    print(f"Saved thigh joints plot to {fig_thigh_path}")
    plt.close(fig_thigh)
    # plt.show() # Commented out show

    # Figure for Calf Joints
    fig_calf = plt.figure(figsize=(15, 8))
    plt.suptitle("Calf Joint Actions (--) vs. Positions (-)", fontsize=12)
    for i, joint_idx in enumerate(calf_indices):
        plt.subplot(2, 2, i + 1)
        # Assuming action_list also follows the sensor order
        plt.plot([step[joint_idx] for step in action_list], '--', label=f"{joint_names[i]} Calf Action")
        plt.plot([step[joint_idx] for step in joint_pos_list], '-', label=f"{joint_names[i]} Calf Pos")
        plt.title(f"{joint_names[i]} Calf Joint")
        plt.xlabel("Time Step")
        plt.ylabel("Angle (rad)")
        plt.legend()
        plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_calf_path = os.path.join(plot_dir, "calf_joints_action_vs_pos.png")
    plt.savefig(fig_calf_path)
    print(f"Saved calf joints plot to {fig_calf_path}")
    plt.close(fig_calf)
    # plt.show() # Commented out show