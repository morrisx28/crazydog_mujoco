import time
import sys
import numpy as np
import threading
import traceback
import yaml
import matplotlib.pyplot as plt
import csv
from math import atan2, sqrt, pi

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import struct
import gamepad


NUM_MOTORS = 8

class Controller:
    def __init__(self):


        config_file = 'compal_amr.yaml'
        with open(f"{config_file}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.dt = config["dt"]
            self.cmd_scale = config["cmd_scale"]

            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.w_base = config["wheelbase"]
            self.t_width = config["trackwidth"]
            
            self.cmd_init = np.array(config["cmd_init"], dtype=np.float32)

        self.pad = gamepad.control_gamepad( 3, [-1.0, 1.0], [-1.0, 1.0], [-3.14, 3.14], [1.0, 1.0, 3.14, 0.05])
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

        self.R = sqrt(self.w_base**2 + self.t_width**2)

        # Wheel order: FL, FR, RL, RR
        self.wheel_positions = {
            'FL': [self.w_base/2, self.t_width/2],
            'FR': [self.w_base/2, -self.t_width/2],
            'RL': [-self.w_base/2, self.t_width/2],
            'RR': [-self.w_base/2, -self.t_width/2],
        }

        self.mode = ''

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
            self.low_cmd.motor_cmd[i].q= self.default_angles[i]
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        self.update_state()
    
    def normalize(self, angle, speed):
        angle = ((angle + pi) % (2*pi)) - pi

        if angle > pi/2:
            angle -= pi
            speed *= -1
        elif angle < -pi/2:
            angle += pi
            speed *= -1

        return angle, speed
    
    def compute(self, cmd_vel):
        vx, vy, wz = cmd_vel
        # Only use planar control
        w_z = wz

        wheel_angles = []
        wheel_speeds = []

        for name in ['FL', 'FR', 'RL', 'RR']:
            x_offset, y_offset = self.wheel_positions[name]

            # Relative rotational velocity at wheel position
            delta_vx = -w_z * y_offset
            delta_vy = w_z * x_offset

            total_vx = vx + delta_vx
            total_vy = vy + delta_vy

            speed = sqrt(total_vx**2 + total_vy**2)
            angle = atan2(total_vy, total_vx)
            angle, speed = self.normalize(angle, speed)

            wheel_speeds.append(speed)
            wheel_angles.append(angle)

        return wheel_angles, wheel_speeds
    
    def move(self):
        command, reset_flag, plot_flag, startlog_flag = self.pad.get_commands()

        wheel_ang, wheel_speed = self.compute(command * self.cmd_scale)
        pos_cmd = np.concatenate([wheel_ang, np.zeros(4)])
        vel_cmd = np.concatenate([np.zeros(4), wheel_speed])

        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].q = pos_cmd[i]
            self.low_cmd.motor_cmd[i].kp = self.kps[i]
            self.low_cmd.motor_cmd[i].dq = vel_cmd[i]
            self.low_cmd.motor_cmd[i].kd = self.kds[i]
            self.low_cmd.motor_cmd[i].tau = 0.0
    
    def move_robot(self):
        self.mode = 'move'
    
    
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
            if self.mode == 'move':
                self.move()
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

            time_until_next_step = self.dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        self.ResetParam()
    
    
        
    def ResetParam(self):
        self.is_running = False


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(1, "lo") # default DDS port for amr

    controller = Controller()
    controller.Init()

    command_dict = {
        "move": controller.move_robot,
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