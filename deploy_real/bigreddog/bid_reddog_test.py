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
import pathlib

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import struct


NUM_MOTORS = 12

class Controller:
    def __init__(self):


        config_file = 'big_reddog_him.yaml'
        with open(f"{config_file}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.dt = config["dt"]
            self.ang_vel_scale = config["ang_vel_scale"]
            policy_path = config["policy_path"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.decimation = config["control_decimation"]
            self.cmd_scale = config["cmd_scale"]
            num_actions = config["num_actions"]
            num_obs = config["num_obs"]
            self.one_step_obs_size = config["one_step_obs_size"]
            self.obs_buffer_size = config["obs_buffer_size"]
            self.action_scale = config["action_scale"]



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

        # Record
        self.ang_vel_data_list = []
        self.gravity_b_list = []
        self.joint_vel_list = []
        self.joint_pos_list = []

        # RL related
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load TorchScript policy to GPU
        # self.policy = torch.jit.load(policy_path, map_location=self.device).to(self.device)

        self.policy = torch.jit.load(policy_path)
        self.counter = 0

        self.action = np.zeros(num_actions, dtype=np.float32)
        self.obs = np.zeros(num_obs, dtype=np.float32)
        # self.obs_tensor_buf = torch.zeros((1, self.one_step_obs_size * self.obs_buffer_size))
        self.obs_tensor_buf = torch.zeros((1, self.one_step_obs_size * self.obs_buffer_size))

        # Chirp data collection
        self.min_freq = 0.1   
        self.max_freq = 10.0  
        self.duration = 20.0  
        self.chirp_amplitude = 1.0
        self.chirp_counter = 0

        self.log_time = []
        self.log_dof_pos = []
        self.log_des_dof_pos = []
        
        self.generate_chirp_profile()

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
        self.update_state()
    

    def stand(self):
        self.controller_rt += self.dt
        ## Get into Default Joint pos ##
        if (self.controller_rt < 3.0):
            # Stand up in first 3 second
            # Total time for standing up or standing down is about 1.2s
            phase = np.tanh(self.controller_rt / 1.2)
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = phase * self.default_angles[i] + (
                    1 - phase) * self.qpos[i]
                self.low_cmd.motor_cmd[i].kp = 40
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = 1
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
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0.0
    
    def move(self):
        if self.counter % self.decimation == 0 and self.counter > 0:
            self.action = self.step()
            self.target_dof_pos = self.default_angles + self.action * self.action_scale


        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].q = self.target_dof_pos[i]
            self.low_cmd.motor_cmd[i].kp = self.kps[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = self.kds[i]
            self.low_cmd.motor_cmd[i].tau = 0.0

        self.counter += 1

    def step(self):
        
        qpos, qvel, ang_vel, quat = self.get_current_state()
        gravity_b = self.get_gravity_orientation(quat)

        self.gravity_b_list.append(gravity_b)
        
        cmd = self.cmd_init

        obs_list = [
            cmd * self.cmd_scale,
            ang_vel * self.ang_vel_scale,
            gravity_b,
            (qpos - self.default_angles) * self.dof_pos_scale,
            qvel * self.dof_vel_scale,
            self.action.astype(np.float32)
        ]

        obs_tensors = [torch.as_tensor(obs, dtype=torch.float32) for obs in obs_list]
        self.obs = torch.cat(obs_tensors, dim=0).unsqueeze(0)


        self.obs_tensor_buf = torch.cat([
            self.obs,
            self.obs_tensor_buf[:, :-self.one_step_obs_size]
        ], dim=1)

        self.obs_tensor_buf = torch.clip(self.obs_tensor_buf, -100, 100)

        # with torch.no_grad():
        #     self.action = self.policy(self.obs_tensor_buf).cpu().numpy().squeeze()
        action = self.policy(self.obs_tensor_buf).detach().numpy().squeeze()

        return action
    
    ## Chirp data collection ##
    def generate_chirp_profile(self):
        sample_rate = 1.0 / self.dt
        num_steps = int(self.duration * sample_rate)
        t = np.linspace(0, self.duration, num_steps)
        f0 = self.min_freq; f1 = self.max_freq
        phase = 2 * np.pi * (f0 * t + ((f1 - f0) / (2 * self.duration)) * t ** 2)
        chirp_signal = np.sin(phase)

        self.chirp_traj = np.zeros((num_steps, 12), dtype=np.float32)
        scales = np.array([0.25, 0.5, 0.75] * 4) * self.chirp_amplitude 
        directions = np.array([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0])

        for i in range(12):
            self.chirp_traj[:, i] = self.default_angles[i] + chirp_signal * scales[i] * directions[i]
        self.num_chirp_steps = num_steps
    
    def chrip_proccess(self):
        # Reset to default pos
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = self.default_angles[i]
            self.low_cmd.motor_cmd[i].kp = 40 
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = 3.0
            self.low_cmd.motor_cmd[i].tau = 0.0
        
        if self.chirp_counter < self.num_chirp_steps:
            target_q = self.chirp_traj[self.chirp_counter]
                
            for i in range(NUM_MOTORS):
                self.low_cmd.motor_cmd[i].q = target_q[i]
                self.low_cmd.motor_cmd[i].kp = self.kps[i] 
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
                self.low_cmd.motor_cmd[i].tau = 0.0

            # Data Collection (Only during Chirp)
            if self.low_state is not None:
                current_q = np.array([self.low_state.motor_state[i].q for i in range(12)])
                # 記錄相對於 Chirp 開始的時間，方便對齊
                self.log_time.append(self.chirp_counter * self.dt) 
                self.log_dof_pos.append(current_q)
                self.log_des_dof_pos.append(target_q.copy())

            self.chirp_counter += 1
            if self.chirp_counter % 500 == 0:
                print(f"Chirp Progress: {self.chirp_counter}/{self.num_chirp_steps}")
        else:
            print("Chirp finished.")
            self.sit_down()
            
    
    def save_data(self):
        if len(self.log_time) == 0:
            return

        print("Saving data...")
        time_tensor = torch.tensor(self.log_time, dtype=torch.float32)
        dof_pos_tensor = torch.tensor(np.array(self.log_dof_pos), dtype=torch.float32)
        des_dof_pos_tensor = torch.tensor(np.array(self.log_des_dof_pos), dtype=torch.float32)

        save_dict = {
            "time": time_tensor,
            "dof_pos": dof_pos_tensor,
            "des_dof_pos": des_dof_pos_tensor
        }

        output_dir = pathlib.Path("data/big_red_dog")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / "chirp_data.pt"
        
        torch.save(save_dict, file_path)
        print(f"Saved to {file_path}")

    def stand_up(self):
        self.mode = 'stand'
        self.reset_timer()

    def sit_down(self):
        self.mode = 'sit'
        self.reset_timer()
    
    def move_rl(self):
        self.mode = 'move'
        self.reset_timer()
    
    def trigger_chirp(self):
        self.mode = 'chirp'
        self.reset_timer()
    
    
    def update_state(self):
        for i in range(NUM_MOTORS):
            self.qpos[i] = self.low_state.motor_state[i].q
            self.qvel[i] = self.low_state.motor_state[i].dq

        
        for i in range(3):
            self.ang_vel[i] = self.low_state.imu_state.gyroscope[i]

        for i in range(4):
            self.quat[i] = self.low_state.imu_state.quaternion[i]

    def get_current_state(self):
        return self.qpos, self.qvel, self.ang_vel, self.quat

    


    def LowCmdWrite(self):
        
        while self.is_running:
            step_start = time.perf_counter()
            if self.mode == 'stand':
                self.stand()
            elif self.mode == 'sit':
                self.sit()
            elif self.mode == 'move':
                self.move()
            elif self.mode == 'chirp':
                self.chrip_proccess()
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

            time_until_next_step = self.dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        self.ResetParam()
    
    def plot(self):
        # Plot the collected data after the simulation ends
        plt.figure(figsize=(18, 20))

        plt.subplot(4, 2, 1)
        for i in range(3): 
            plt.plot([step[i] for step in self.ang_vel_data_list], label=f"Angular Velocity {i}")
        plt.title(f"History Angular Velocity", fontsize=10, pad=10)  # Added pad for spacing
        plt.legend()
        plt.subplot(4, 2, 2)
        for i in range(3):
            plt.plot([step[i] for step in self.gravity_b_list], label=f"Project Gravity {i}")
        plt.title(f"History Project Gravity", fontsize=10, pad=10)  # Added pad for spacing
        plt.legend()
        plt.subplot(4, 2, 3)
        for i in range(NUM_MOTORS):
            plt.plot([step[i] for step in self.joint_pos_list], label=f"Joint Position {i}")
        plt.title(f"History Joint Position", fontsize=10, pad=10)  # Added pad for spacing
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        
    def ResetParam(self):
        self.controller_rt = 0
        self.chirp_counter = 0
        self.is_running = False


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
        "save": controller.save_data,
        "chirp": controller.trigger_chirp,
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
    sys.exit(0)     
