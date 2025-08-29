import time
import sys
import numpy as np
import threading
import traceback
import torch
import yaml
import argparse
import matplotlib.pyplot as plt
import csv

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import struct

from estimation_utils import define_state_order, build_x, estimate_z, estimate_body_xy_velocity
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

NUM_MOTORS = 6
class Controller:
    def __init__(self):


        config_file = 'pineapple.yaml'
        with open(f"{config_file}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.dt = config["simulation_dt"]

            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.sit_angles = np.array(config["sit_angles"], dtype=np.float32)
            
            self.cmd_init = np.array(config["cmd_init"], dtype=np.float32)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  


        self.controller_rt = 0.0
        self.is_running = False

        # thread handling
        self.lowCmdWriteThreadPtr = None

        # state
        self.target_dof_pos = self.default_angles.copy()
        self.target_dof_vel = np.zeros(NUM_MOTORS)
        self.qpos = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qvel = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qtau = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.quat = np.zeros(4) # q_w q_x q_y q_z
        self.ang_vel = np.zeros(3)

        self.mode = ''

        # Record data
        self.is_record = True
        log_file = open("state_estimation_log.csv", "w", newline='')
        self.csv_writer = csv.writer(log_file)

        # Load robot model
        urdf_path = "./models/pineapple/pineapple_6dof.urdf"
        mesh_dir = "./models/pineapple/"

        self.model = RobotWrapper.BuildFromURDF(urdf_path, mesh_dir, root_joint = pin.JointModelFreeFlyer()).model
        data = self.model.createData()
        # Write header
        self.csv_writer.writerow(["z_est", "xy_vel_x", "xy_vel_y",
                                "ang_vel_x", "ang_vel_y", "ang_vel_z",
                                "quat_w", "quat_x", "quat_y", "quat_z",
                                "qpos_l_t", "qpos_l_c", "qpos_l_w", "qpos_r_t", "qpos_r_c", "qpos_r_w",
                                "qvel_l_t", "qvel_l_c", "qvel_l_w", "qvel_r_t", "qvel_r_c", "qvel_r_w",
                                "qtau_l_t", "qtau_l_c", "qtau_l_w", "qtau_r_t", "qtau_r_c", "qtau_r_w",
                                "action_l_t", "action_l_c", "action_l_w", "action_r_t", "action_r_c", "action_r_w"])

        # RL related
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load TorchScript policy to GPU
        policy_path = '/home/crazydog/test/deploy_real/pre_train/him/policy_Jul15_15-23-06_.pt'
        self.policy = torch.jit.load(policy_path, map_location=self.device).to(self.device)

        # self.policy = torch.jit.load(policy_path)
        self.counter = 0
        self.decimation = 2
        self.lin_vel_scale = 2.0
        self.ang_vel_scale = 0.25  # 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05

        self.pos_action_scale = 0.5 
        self.vel_action_scale = 5.0 

        self.cmd_scale = np.array([2.0, 2.0, 0.25])
        num_actions = 6
        num_obs = 150
        self.one_step_obs_size = 25
        self.obs_buffer_size = 6

        self.action = np.zeros(num_actions, dtype=np.float32)
        self.obs = np.zeros(num_obs, dtype=np.float32)
        # self.obs_tensor_buf = torch.zeros((1, self.one_step_obs_size * self.obs_buffer_size))
        self.obs_tensor_buf = torch.zeros((1, self.one_step_obs_size * self.obs_buffer_size), device=self.device)

        self.crc = CRC()

    # Public methods
    def Init(self):
        self.InitLowCmd()

        # create publisher #
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        # Init default pos #
        self.Start()

        print("Initial Sucess !!!")

    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation
    

    def Start(self):
        self.is_running = True
        self.lowCmdWriteThreadPtr = threading.Thread(target=self.LowCmdWrite)
        self.lowCmdWriteThreadPtr.start()

    def ShutDown(self):
        self.is_running = False
        self.lowCmdWriteThreadPtr.join()


    # Private methods
    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= self.sit_angles[i]
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        self.get_current_state()
        # print(f'qpos {self.low_state.motor_state[0].q}')
        # quat = self.low_state.imu_state.quaternion
        # ang_vel = self.low_state.imu_state.gyroscope
        # print(f'quat w: {quat[0]} x: {quat[1]} y: {quat[2]} z: {quat[3]}')
        # print(f'ang_vel x: {ang_vel[0]} y: {ang_vel[1]} z: {ang_vel[2]}')
    

    def stand(self):
        self.controller_rt += self.dt
        ## Get into Default Joint pos ##
        if (self.controller_rt < 3.0):
            # Stand up in first 3 second
            # Total time for standing up or standing down is about 1.2s
            phase = np.tanh(self.controller_rt / 1.2)
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = phase * self.default_angles[i] + (
                    1 - phase) * self.sit_angles[i]
                self.low_cmd.motor_cmd[i].kp = 25
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = 0.3
                self.low_cmd.motor_cmd[i].tau = 0.0
    
    def reset_timer(self):
        self.controller_rt = 0.0
    
    def sit(self):
        self.controller_rt += self.dt
        ## Get into Default Joint pos ##
        if (self.controller_rt < 3.0):
            # Stand up in first 3 second
            # Total time for standing up or standing down is about 1.2s
            phase = np.tanh(self.controller_rt / 1.2)
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = phase * self.sit_angles[i] + (
                    1 - phase) * self.qpos[i]
                self.low_cmd.motor_cmd[i].kp = 15
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = 0.3
                self.low_cmd.motor_cmd[i].tau = 0.0
    
    def state_estimation(self, model, ori_body, left_joints, right_joints, left_joints_vel, right_joints_vel):
        z_init = 0.0
        x_init = build_x(model, z_init, ori_body,
                        left_joints, right_joints,
                        left_joints_vel, right_joints_vel)

        z_body = estimate_z(x_init)
        velocity_body_xy = estimate_body_xy_velocity(model, x_init)
        return z_body, velocity_body_xy
    
    def move(self):
        if self.counter % self.decimation == 0 and self.counter > 0:
            self.action = self.step()

            self.target_dof_pos = np.array([self.action[0], self.action[1], 0, self.action[3], self.action[4], 0]) * self.pos_action_scale + self.default_angles
            self.target_dof_vel = np.array([self.action[2], self.action[5]]) * self.vel_action_scale

        for i in range(2):
            self.low_cmd.motor_cmd[i].q = self.target_dof_pos[i]
            self.low_cmd.motor_cmd[i].kp = self.kps[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = self.kds[i]
            self.low_cmd.motor_cmd[i].tau = 0.0

        self.low_cmd.motor_cmd[2].q = 0.0
        self.low_cmd.motor_cmd[2].kp = self.kps[2]
        self.low_cmd.motor_cmd[2].dq = self.target_dof_vel[0]
        self.low_cmd.motor_cmd[2].kd = self.kds[2]
        self.low_cmd.motor_cmd[2].tau = 0.0

        for i in range(2):
            self.low_cmd.motor_cmd[i+3].q = self.target_dof_pos[i+3]
            self.low_cmd.motor_cmd[i+3].kp = self.kps[i+3]
            self.low_cmd.motor_cmd[i+3].dq = 0.0
            self.low_cmd.motor_cmd[i+3].kd = self.kds[i+3]
            self.low_cmd.motor_cmd[i+3].tau = 0.0
        
        self.low_cmd.motor_cmd[5].q = 0.0
        self.low_cmd.motor_cmd[5].kp = self.kps[5]
        self.low_cmd.motor_cmd[5].dq = self.target_dof_vel[1]
        self.low_cmd.motor_cmd[5].kd = self.kds[5]
        self.low_cmd.motor_cmd[5].tau = 0.0

        # Record data
        if self.is_record:
            z_est, xy_vel = self.state_estimation(self.model, self.quat, self.qpos[:3], 
                                              self.qpos[3:6], self.qvel[:3], self.qvel[3:6])
            row = [z_est, xy_vel[0], xy_vel[1], 
                   self.ang_vel[0], self.ang_vel[1], self.ang_vel[2],
                   self.quat[0], self.quat[1], self.quat[2], self.quat[3],
                   self.qpos[0], self.qpos[1], self.qpos[2], self.qpos[3], self.qpos[4], self.qpos[5],
                   self.qvel[0], self.qvel[1], self.qvel[2], self.qvel[3], self.qvel[4], self.qvel[5],
                   self.qtau[0], self.qtau[1], self.qtau[2], self.qtau[3], self.qtau[4], self.qtau[5]]
            self.csv_writer.writerow(row)
        self.counter += 1

    # def step(self):
    #     ## Get qpos qvel
    #     gravity_b = self.get_gravity_orientation(self.quat)
    #     ## Get joystick cmd
    #     cmd = self.cmd_init

    #     obs_list = [
    #         cmd * self.cmd_scale,
    #         self.ang_vel * self.ang_vel_scale,
    #         gravity_b,
    #         (self.qpos[:2] - self.default_angles[:2]) * self.dof_pos_scale,
    #         (self.qpos[3:5] - self.default_angles[3:5]) * self.dof_pos_scale,
    #         self.qvel * self.dof_vel_scale,
    #         self.action.astype(np.float32)
    #     ]

    #     obs_list = [torch.tensor(obs, dtype=torch.float32) for obs in obs_list]

    #     self.obs = torch.cat(obs_list, dim=0).unsqueeze(0)

    #     self.obs_tensor_buf = torch.cat([
    #         self.obs,
    #         self.obs_tensor_buf[:, :-self.one_step_obs_size]
    #     ], dim=1)

    #     self.obs_tensor_buf = torch.clip(self.obs_tensor_buf, -100, 100)

    #     self.action = self.policy(self.obs_tensor_buf).detach().numpy().squeeze()
    #     return self.action
    def step(self):
        gravity_b = self.get_gravity_orientation(self.quat)
        cmd = self.cmd_init

        obs_list = [
            cmd * self.cmd_scale,
            self.ang_vel * self.ang_vel_scale,
            gravity_b,
            (self.qpos[:2] - self.default_angles[:2]) * self.dof_pos_scale,
            (self.qpos[3:5] - self.default_angles[3:5]) * self.dof_pos_scale,
            self.qvel * self.dof_vel_scale,
            self.action.astype(np.float32)
        ]

        obs_list = [torch.tensor(obs, dtype=torch.float32, device=self.device) for obs in obs_list]
        self.obs = torch.cat(obs_list, dim=0).unsqueeze(0)

        self.obs_tensor_buf = torch.cat([
            self.obs,
            self.obs_tensor_buf[:, :-self.one_step_obs_size]
        ], dim=1)

        self.obs_tensor_buf = torch.clamp(self.obs_tensor_buf, -100, 100)

        with torch.no_grad():
            self.action = self.policy(self.obs_tensor_buf).cpu().numpy().squeeze()

        return self.action


    def stand_up(self):
        self.mode = 'stand'
        self.reset_timer()

    def sit_down(self):
        self.mode = 'sit'
        self.reset_timer()
    
    def move_rl(self):
        self.mode = 'move'
        self.reset_timer()
    
    def get_current_state(self):
        for i in range(NUM_MOTORS):
            self.qpos[i] = self.low_state.motor_state[i].q
            self.qvel[i] = self.low_state.motor_state[i].dq
            self.qtau[i] = self.low_state.motor_state[i].tau_est

        
        for i in range(3):
            self.ang_vel[i] = self.low_state.imu_state.gyroscope[i]

        for i in range(4):
            self.quat[i] = self.low_state.imu_state.quaternion[i]

        self.project_gravity = self.get_gravity_orientation(self.quat)
    


    def LowCmdWrite(self):
        
        while self.is_running:
            step_start = time.perf_counter()
            if self.mode == 'stand':
                self.stand()
            elif self.mode == 'sit':
                self.sit()
            elif self.mode == 'move':
                self.move()
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

            time_until_next_step = self.dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        self.ResetParam()
    
        
    def ResetParam(self):
        self.controller_rt = 0
        self.is_running = False

    def plot(self):
        plt.figure(figsize=(14, 16))

        plt.subplot(2, 1, 1)
        for i in range(NUM_MOTORS): 
            plt.plot([step[i] for step in self.torque_data], label=f"Torque Data {i}")
        plt.title(f"History Torque data", fontsize=10, pad=10)  # Added pad for spacing
        plt.legend()
        plt.subplot(2, 1, 2)
        for i in range(NUM_MOTORS):
            plt.plot([step[i] for step in self.pos_cmd], label=f"Position Cmd {i}")
        plt.title(f"History Position cmd", fontsize=10, pad=10)  # Added pad for spacing
        plt.legend()
        plt.tight_layout()
        plt.show()

        self.torque_data = []
        self.pos_cmd = []


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(1, "lo") # default DDS port for pineapple

    controller = Controller()
    controller.Init()

    command_dict = {
        "stand": controller.stand_up,
        "sit": controller.sit_down,
        "move": controller.move_rl,
        "plot": controller.plot,
    }

    while True:        
        try:
            cmd = input("CMD :")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                controller.ShutDown()
                break

        except Exception as e:
            traceback.print_exc()
            break
    sys.exit(-1)     