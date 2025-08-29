import sys
import os
import torch
import mujoco
import mujoco.viewer
import time
import argparse
import pickle
import numpy as np
from datalogger import DataLogger

# load mujoco model
m = mujoco.MjModel.from_xml_path('xml/genesis/quick_scence.xml')
d = mujoco.MjData(m)

# utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
# # add utils to sys.path
# sys.path.append(utils_path)
# print("path",utils_path)
print(m.opt.timestep)

import gamepad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_sensor_data(sensor_name):
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1:
        raise ValueError(f"Sensor '{sensor_name}' not found in model!")
    start_idx = m.sensor_adr[sensor_id]
    dim = m.sensor_dim[sensor_id]
    sensor_values = d.sensordata[start_idx : start_idx + dim]
    return torch.tensor(
        sensor_values, 
        device=device, 
        dtype=torch.float32
    )

def world2self(quat, v):
    q_w = quat[0] 
    q_vec = quat[1:] 
    v_vec = torch.tensor(v, device=device,dtype=torch.float32)
    a = v_vec * (2.0 * q_w**2 - 1.0)
    b = torch.linalg.cross(q_vec, v_vec) * q_w * 2.0
    c = q_vec * torch.dot(q_vec, v_vec) * 2.0
    result = a - b + c
    return result.to(device)

def get_obs(env_cfg, obs_scales, actions, default_dof_pos, commands=[0.0, 0.0, 0.0, 0.25]):
    commands_scale = torch.tensor(
        [obs_scales["lin_vel"], obs_scales["lin_vel"], 
         obs_scales["ang_vel"], obs_scales["height_measurements"]], 
         device=device, dtype=torch.float32
    )
    base_quat = get_sensor_data("orientation")
    gravity = [0.0, 0.0, -1.0]
    projected_gravity = world2self(base_quat,torch.tensor(gravity, device=device, dtype=torch.float32))
    base_lin_vel = world2self(base_quat, get_sensor_data("base_lin_vel"))
    base_ang_vel = get_sensor_data("base_ang_vel")
    # print("base_lin_vel:", base_lin_vel)
    # print("base_ang_vel:", base_ang_vel)
    # print("commands:", commands)

    dof_pos = torch.zeros(env_cfg["num_actions"], device=device, dtype=torch.float32)    
    for i, dof_name in enumerate(env_cfg["dof_names"]):
        dof_pos[i] = get_sensor_data(dof_name+"_p")[0]
        if i==3:
            break

    dof_vel = torch.zeros(env_cfg["num_actions"], device=device, dtype=torch.float32)
    for i, dof_name in enumerate(env_cfg["dof_names"]):
        dof_vel[i] = get_sensor_data(dof_name+"_v")[0]

    dof_torque = np.zeros(env_cfg["num_actions"])
    for i, dof_name in enumerate(env_cfg["dof_names"]):
        dof_torque[i] = get_sensor_data(dof_name+"_t")[0]


    cmds = torch.tensor(commands, device=device, dtype=torch.float32)

    return torch.cat(
        [
            base_ang_vel * obs_scales["ang_vel"],  # 3
            projected_gravity,  # 3
            cmds * commands_scale,  # 4
            (dof_pos[0:4] - default_dof_pos[0:4]) * obs_scales["dof_pos"],  # 4
            dof_vel * obs_scales["dof_vel"],  # 6
            actions,  # 6
        ],
        axis=-1,
    ), dof_torque

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="quick_wheel-legged-walking-v13") 
    args = parser.parse_args()
    logger = DataLogger('log')

    # path to logs file
    log_dir = os.path.join('pre_train/genesis', args.exp_name)
    cfg_path = os.path.join(log_dir, 'cfgs.pkl')

    # load parameter
    if os.path.exists(cfg_path):
        print("file path:", cfg_path)
        env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
        pos_action_scale = 0.25
        vel_action_scale = 5
        dt = 0.01 # 100 hz for controller
    else:
        print("file not exist:", cfg_path)
        exit()

    # load gamepad
    pad = gamepad.control_gamepad(command_cfg, [1.0, 0.0, 3.14, 0.05])
    commands, reset_flag, plot_flag, startlog_flag = pad.get_commands()
    # commands, reset_flag = np.array([0, 0, 0, 0]), 0

    # load model
    try:
        loaded_policy = torch.jit.load(os.path.join(log_dir, "policy.pt"))
        loaded_policy.eval()  # set to eval mode
        loaded_policy.to(device)
        print("load model success!")
    except Exception as e:
        print(f"load model fail: {e}")
        exit()

    #dof limits
    lower = [env_cfg["dof_limit"][name][0] for name in env_cfg["dof_names"]]
    upper = [env_cfg["dof_limit"][name][1] for name in env_cfg["dof_names"]]
    dof_pos_lower = torch.tensor(lower).to(device)
    dof_pos_upper = torch.tensor(upper).to(device)
        
    # init obs buffer
    history_obs_buf = torch.zeros((obs_cfg["history_length"], obs_cfg["num_slice_obs"]), device=device, dtype=torch.float32)
    slice_obs_buf = torch.zeros(obs_cfg["num_slice_obs"], device=device, dtype=torch.float32)
    obs_buf = torch.zeros((obs_cfg["num_obs"]), device=device, dtype=torch.float32)
    default_dof_pos = torch.tensor(
        [env_cfg["default_joint_angles"][name] for name in env_cfg["dof_names"]],
        device=device,
        dtype=torch.float32)

    print(default_dof_pos)
    print(env_cfg["dof_names"])
    print(env_cfg["joint_action_scale"])
    print(obs_cfg["obs_scales"])
    print(env_cfg["num_actions"])
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            
            actions = loaded_policy(obs_buf)
            actions = torch.clip(actions, -env_cfg["clip_actions"], env_cfg["clip_actions"])
            slice_obs_buf , dof_torque = get_obs(env_cfg=env_cfg, obs_scales=obs_cfg["obs_scales"],
                                    actions=actions, default_dof_pos=default_dof_pos, commands=commands)
            slice_obs_buf = slice_obs_buf.unsqueeze(0)
            obs_buf = torch.cat([history_obs_buf, slice_obs_buf], dim=0).view(-1)

            # update buffer
            if obs_cfg["history_length"] > 1:
                history_obs_buf[:-1, :] = history_obs_buf[1:, :].clone() 
            history_obs_buf[-1, :] = slice_obs_buf 

            # update action
            target_dof_pos = actions[0:4] * pos_action_scale + default_dof_pos[0:4]
            target_dof_vel = actions[4:6] * vel_action_scale
            target_dof_pos = torch.clamp(target_dof_pos, dof_pos_lower[0:4],dof_pos_upper[0:4])

            for i in range(env_cfg["num_actions"]-2):
                d.ctrl[i] = target_dof_pos.detach().cpu().numpy()[i]

            d.ctrl[4] = target_dof_vel.detach().cpu().numpy()[0]
            d.ctrl[5] = target_dof_vel.detach().cpu().numpy()[1]
            # kps = np.array([15, 15, 15, 15, 0, 0])
            # kds = np.array([0.5, 0.5, 0.5, 0.5, 0.3, 0.3])
            # pos_feedback  = np.concatenate([d.sensordata[:4], np.zeros(2)])
            # tau = pd_control(np.concatenate([target_dof_pos.detach().cpu().numpy(), np.zeros(2)]), pos_feedback, 
            #                                 kps, np.concatenate([np.zeros(4), target_dof_vel.detach().cpu().numpy()]), d.sensordata[4:4 + 6], kds)
            # d.ctrl[:] = tau


            base_height = d.qpos[2]

            # Log data
            if startlog_flag == True:
                logger.log(obs_buf, actions, dof_torque, base_height)
            if plot_flag == True:
                logger.plot_and_save()

            commands, reset_flag, plot_flag, startlog_flag = pad.get_commands()
            if reset_flag:
                mujoco.mj_resetData(m, d) 
                logger.clear_buffer()
                
            # simulate in one step
            step_start = time.time()
            for i in range(5):
                mujoco.mj_step(m, d)

            viewer.sync()

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
