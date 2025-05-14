import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import argparse
NUM_MOTOR = 12

def calculate_com_in_base_frame(model, data, base_body_id):
    total_mass = 0.0
    com_sum = np.zeros(3)

    # Get base position and rotation
    base_pos = data.xipos[base_body_id]  # Position of the base in world coordinates
    base_rot = data.ximat[base_body_id].reshape(3, 3)  # Rotation matrix of the base

    for i in range(model.nbody):
        # Get body mass and world COM position
        mass = model.body_mass[i]
        world_com = data.xipos[i]

        # Transform COM to base coordinates
        local_com = world_com - base_pos  # Translate to base origin
        local_com = base_rot.T @ local_com  # Rotate into base frame

        # Accumulate mass-weighted positions
        com_sum += mass * local_com
        total_mass += mass

    # Compute overall COM in base coordinates
    center_of_mass_base = com_sum / total_mass
    return center_of_mass_base

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert input quaternion to rotation matrix.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < 1e-10:
        return np.identity(3)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
            (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
            (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
        ),
        dtype=np.float64,
    )

def pd_control(target_q, q, kp, target_dq, dq, kd, g):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd + g


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    # parser.add_argument("position_file", type=str, help="File containing joint position sequence")
    args = parser.parse_args()
    
    config_file = args.config_file
    position_file = "animation/joint_angles_20250228_140351_resampled_reordered_foot_endpoints_joint_pos.txt"
    
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = config["xml_path"]
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        # Get these values if they exist in the config
        lin_vel_scale = config.get("lin_vel_scale", 1.0)
        ang_vel_scale = config.get("ang_vel_scale", 1.0)
        dof_pos_scale = config.get("dof_pos_scale", 1.0)
        dof_vel_scale = config.get("dof_vel_scale", 1.0)
        action_scale = config.get("action_scale", 1.0)
        cmd_scale = np.array(config.get("cmd_scale", [1.0, 1.0, 1.0]), dtype=np.float32)
        
        # Get cmd_init if it exists
        cmd = np.array(config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)

    # Load joint positions from file
    try:
        print(f"Loading joint positions from: {position_file}")
        with open(position_file, 'r') as f:
            lines = f.readlines()
        
        joint_positions = []
        for line in lines:
            # Skip empty lines and comment lines
            if not line.strip() or line.strip().startswith('#'):
                continue
                
            # Parse the line into float values
            values = [float(val) for val in line.strip().split()]
            
            # Check if we have the expected number of values
            if len(values) == NUM_MOTOR:
                joint_positions.append(values)
            else:
                print(f"Warning: Line has {len(values)} values, expected {NUM_MOTOR}. Skipping.")
        
        joint_positions = np.array(joint_positions)
        print(f"Successfully loaded {joint_positions.shape[0]} joint position frames")
        
    except Exception as e:
        print(f"Error loading joint positions file: {e}")
        print("Using default joint angles")
        joint_positions = np.array([default_angles])
        
    # Initialize variables
    target_dof_pos = default_angles.copy()
    frame_index = 0
    total_frames = joint_positions.shape[0]

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    base_body_id = 1

    # Record data for plotting
    lin_vel_data_list = []
    ang_vel_data_list = []
    gravity_b_list = []
    joint_vel_list = []
    position_list = []

    counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:     
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        # mujoco.mj_resetDataKeyframe(m, d, 0)
        time.sleep(2)
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            # transform action to target_dof_pos
            if counter < 700:
                target_dof_pos = default_angles
            else:
                # Get the current joint position target from the loaded sequence
                if frame_index < total_frames:
                    target_dof_pos = joint_positions[frame_index]
                    # Move to the next frame
                    frame_index = (frame_index + 1) % total_frames
                    print(f"Showing frame {frame_index+1}/{total_frames}")

            gravity_torque = np.zeros(12)
            gravity_torque[0] = d.qfrc_bias[0+6]
            gravity_torque[1] = d.qfrc_bias[3+6]
            gravity_torque[2] = d.qfrc_bias[6+6]
            gravity_torque[3] = d.qfrc_bias[9+6]
            gravity_torque[4] = d.qfrc_bias[1+6]
            gravity_torque[5] = d.qfrc_bias[4+6]
            gravity_torque[6] = d.qfrc_bias[7+6]
            gravity_torque[7] = d.qfrc_bias[10+6]
            gravity_torque[8] = d.qfrc_bias[2+6]
            gravity_torque[9] = d.qfrc_bias[5+6]
            gravity_torque[10] = d.qfrc_bias[8+6]
            gravity_torque[11] = d.qfrc_bias[11+6]


            # Apply PD control to reach the target position
            tau = pd_control(target_dof_pos, d.sensordata[:NUM_MOTOR], kps, np.zeros(12), d.sensordata[NUM_MOTOR:NUM_MOTOR + NUM_MOTOR], kds, gravity_torque)
            # print('aaa', d.qfrc_bias)
            # tau = np.zeros(12)
            
            
            d.ctrl[:] = tau
            # print(d.ctrl)

            # Step the physics
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0 and counter > 0:
                # Record data similar to the original script
                qpos = d.sensordata[:12]
                qvel = d.sensordata[12:24]
                
                # Check if these sensor indices are valid for the current model
                try:
                    lin_vel_I = d.sensordata[49:52] if len(d.sensordata) > 51 else np.zeros(3)
                    ang_vel_I = d.sensordata[52:55] if len(d.sensordata) > 54 else np.zeros(3)
                    gravity_b = get_gravity_orientation(d.sensordata[36:40]) if len(d.sensordata) > 39 else np.zeros(3)
                except IndexError:
                    lin_vel_I = np.zeros(3)
                    ang_vel_I = np.zeros(3)
                    gravity_b = np.zeros(3)
                    print("Warning: Some sensor data not available in this model")
                
                
                
            # Update the viewer
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("Simulation complete")