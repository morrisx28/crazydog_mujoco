import time
import sys
import numpy as np
import threading
import traceback
import torch
import yaml
import argparse
from scipy.spatial.transform import Rotation as R

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import unitree_legged_const as go2
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
import struct


class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16

    def set(self, data):
        # wireless_remote
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]

class Controller:
    def __init__(self):

        self.remote_controller = RemoteController()

        config_file = 'go2_lab.yaml'
        with open(f"{config_file}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            policy_path = config["policy_path"]
            dt = config["dt"]
            control_decimation = config["control_decimation"]

            kps = np.array(config["kps"], dtype=np.float32)
            kds = np.array(config["kds"], dtype=np.float32)

            default_angles = np.array(config["default_angles"], dtype=np.float32)

            ang_vel_scale = config["ang_vel_scale"]
            dof_pos_scale = config["dof_pos_scale"]
            dof_vel_scale = config["dof_vel_scale"]
            action_scale = config["action_scale"]
            cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            num_actions = config["num_actions"]
            num_obs = config["num_obs"]
            
            cmd_init = np.array(config["cmd_init"], dtype=np.float32)

        self.dt = dt

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  

        self.default_angles = np.array(default_angles, dtype=float)

        self.stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
            1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
        ],
                                    dtype=float)


        self.startPos = [0.0] * 12
        self.controller_rt = 0.0
        self.is_running = False

        # thread handling
        self.lowCmdWriteThreadPtr = None

        # RL init param
        # policy_path = "/home/csl/robot_model/go2/model/2025-05-11_11-57-25/exported/policy.pt"
        self.policy = torch.jit.load(policy_path)

        self.counter = 0
        self.decimation = control_decimation
        self.kp = kps
        self.kd = kds
        self.ang_vel_scale = ang_vel_scale
        self.action_scale = action_scale
        self.dof_pos_scale = dof_pos_scale
        self.dof_vel_scale = dof_vel_scale

        self.target_dof_pos = self.default_angles.copy()
        self.action = np.zeros(num_actions, dtype=np.float32)
        self.obs = np.zeros(num_obs, dtype=np.float32)
        self.qpos = np.zeros(12, dtype=np.float32)
        self.qvel = np.zeros(12, dtype=np.float32)
        self.cmd_init = cmd_init

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


        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)
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

    def get_linear_vel(self, lin_acc):
        lin_acc[2] - 9.8
        self.lin_vel += lin_acc

        return self.lin_vel

    def Start(self):
        # self.lowCmdWriteThreadPtr = RecurrentThread(
        #     interval=self.dt, target=self.LowCmdWrite, name="writebasiccmd"
        # )
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
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowCmdWrite(self):
        
        while self.is_running:
            if controller.remote_controller.button[KeyMap.A] == 1:
                break
            ## Stand up at first 3 sec ##
            step_start = time.perf_counter()
            self.controller_rt += self.dt
            ## Get into Default Joint pos ##
            if (self.controller_rt < 3.0):
                # Stand up in first 3 second
                # Total time for standing up or standing down is about 1.2s
                phase = np.tanh(self.controller_rt / 1.2)
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = phase * self.default_angles[i] + (
                        1 - phase) * self.stand_down_joint_pos[i]
                    self.low_cmd.motor_cmd[i].kp = 50
                    self.low_cmd.motor_cmd[i].dq = 0.0
                    self.low_cmd.motor_cmd[i].kd = 3.5
                    self.low_cmd.motor_cmd[i].tau = 0.0
            ## Start RL controller ##
            else:
                self.counter += 1
                if self.counter % self.decimation == 0 and self.counter > 0:
                    self.action = self.step()
                
                if self.counter < 500:
                    self.target_dof_pos = self.default_angles
                else:
                    self.target_dof_pos = self.cmdRemapping(self.action) * self.action_scale + self.default_angles
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = self.target_dof_pos[i]
                    self.low_cmd.motor_cmd[i].kp = self.kp[i]
                    self.low_cmd.motor_cmd[i].dq = 0.0
                    self.low_cmd.motor_cmd[i].kd = self.kd[i]
                    self.low_cmd.motor_cmd[i].tau = 0.0

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

            time_until_next_step = self.dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        self.ResetParam()
    
    def step(self):
        ## Get qpos qvel
        for i in range(12):
            self.qpos[i] = self.low_state.motor_state[i].q
            self.qvel[i] = self.low_state.motor_state[i].dq
        ## Get imu state
        quat = self.low_state.imu_state.quaternion
        ang_vel = self.low_state.imu_state.gyroscope
        gravity_b = self.get_gravity_orientation(quat)
        ## Get joystick cmd
        cmd = [self.remote_controller.ly, self.remote_controller.lx * -1, self.remote_controller.rx * -1]
        ## To obs
        qpos_obs = self.jointIDRemapping(self.qpos)
        qvel_obs = self.jointIDRemapping(self.qvel)

        self.obs[:3] = ang_vel 
        self.obs[3:6] = gravity_b
        self.obs[6:9] = cmd
        self.obs[9:21] = (qpos_obs - self.jointIDRemapping(self.default_angles)) * self.dof_pos_scale
        self.obs[21:33] = qvel_obs * self.dof_vel_scale
        self.obs[33:45] = self.action

        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        return self.action
    
    def jointIDRemapping(self, qpos):
        return np.array([qpos[go2.LegID["FL_0"]], qpos[go2.LegID["FR_0"]], qpos[go2.LegID["RL_0"]], qpos[go2.LegID["RR_0"]],
                qpos[go2.LegID["FL_1"]], qpos[go2.LegID["FR_1"]], qpos[go2.LegID["RL_1"]], qpos[go2.LegID["RR_1"]],
                qpos[go2.LegID["FL_2"]], qpos[go2.LegID["FR_2"]], qpos[go2.LegID["RL_2"]], qpos[go2.LegID["RR_2"]],
                ])
    
    def cmdRemapping(self, q_cmd):
        return np.array([q_cmd[go2.LabID["FR_0"]], q_cmd[go2.LabID["FR_1"]], q_cmd[go2.LabID["FR_2"]], q_cmd[go2.LabID["FL_0"]],
                q_cmd[go2.LabID["FL_1"]], q_cmd[go2.LabID["FL_2"]], q_cmd[go2.LabID["RR_0"]], q_cmd[go2.LabID["RR_1"]],
                q_cmd[go2.LabID["RR_2"]], q_cmd[go2.LabID["RL_0"]], q_cmd[go2.LabID["RL_1"]], q_cmd[go2.LabID["RL_2"]],
                ])
        
    def ResetParam(self):
        self.controller_rt = 0
        self.is_running = False


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ## For Sim2Real ##
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ## For Sim2Sim ##
        ChannelFactoryInitialize(1, "lo")

    controller = Controller()
    controller.Init()

    command_dict = {
        "st": controller.Start,
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