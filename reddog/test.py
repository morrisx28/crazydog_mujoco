import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import argparse
NUM_MOTOR = 12


def pd_control(target_q, q, kp, target_dq, dq, kd, g):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd 


if __name__ == "__main__":
    
    # Load robot model
    m = mujoco.MjModel.from_xml_path("scene_lab.xml")
    d = mujoco.MjData(m)
    m.opt.timestep = 0.02
    base_body_id = 1

    simulation_duration=1000

    counter = 0
    
    with mujoco.viewer.launch_passive(m, d) as viewer:     
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        mujoco.mj_resetDataKeyframe(m, d, 0)
        print("After reset, joint positions:", d.qpos)
        # time.sleep(10)
        while viewer.is_running() and time.time() - start < simulation_duration:

            # gravity_torque = d.qfrc_bias[6:6+12]  # Includes gravity and Coriolis
            # Apply PD control to reach the target position
            # tau = pd_control(target_dof_pos, d.sensordata[:NUM_MOTOR], kps, np.zeros(12), d.sensordata[NUM_MOTOR:NUM_MOTOR + NUM_MOTOR], kds, gravity_torque)
            
            # print('aaa', d.qfrc_bias)
            tau = np.zeros(12)
            tau[0:3] = d.qfrc_bias[3:6]
            tau[3:6] = d.qfrc_bias[0:3]
            tau[6:9] = d.qfrc_bias[9:12]
            tau[9:12] = d.qfrc_bias[6:9]
            tau[0] = d.qfrc_bias[0]
            tau[1] = d.qfrc_bias[3]
            tau[2] = d.qfrc_bias[6]
            tau[3] = d.qfrc_bias[9]
            tau[4] = d.qfrc_bias[1]
            tau[5] = d.qfrc_bias[4]
            tau[6] = d.qfrc_bias[7]
            tau[7] = d.qfrc_bias[10]
            tau[8] = d.qfrc_bias[2]
            tau[9] = d.qfrc_bias[5]
            tau[10] = d.qfrc_bias[8]
            tau[11] = d.qfrc_bias[11]
            
            d.ctrl[:] = tau
            # print(d.ctrl)

            # Step the physics
            mujoco.mj_step(m, d)
                
            # Update the viewer
            viewer.sync()
    
    print("Simulation complete")