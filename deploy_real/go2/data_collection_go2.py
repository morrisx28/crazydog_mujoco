import time
import sys
import numpy as np
import threading
import traceback
import torch
import yaml
import struct
import pathlib

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

class KeyMap:
    R1 = 0; L1 = 1; start = 2; select = 3; R2 = 4; L2 = 5
    F1 = 6; F2 = 7; A = 8; B = 9; X = 10; Y = 11
    up = 12; right = 13; down = 14; left = 15

class RemoteController:
    def __init__(self):
        self.lx = 0; self.ly = 0; self.rx = 0; self.ry = 0
        self.button = [0] * 16

    def set(self, data):
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
        
        # --- Config ---
        config_file = 'go2_pace.yaml'
        try:
            with open(f"{config_file}", "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                self.dt = config.get("dt")
                self.kps = np.array(config.get("kps"), dtype=np.float32)
                self.kds = np.array(config.get("kds"), dtype=np.float32)
                self.default_angles = np.array(config.get("default_angles"), dtype=np.float32)
        except Exception:
            print(f"Failed to load config from {config_file}.")

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  

        self.stand_down_joint_pos = np.array([
            -0.047, 1.22, -2.44, 0.047, 1.22, -2.44, 
            -0.047, 1.22, -2.44, 0.047, 1.22, -2.44
        ], dtype=float)

        self.controller_rt = 0.0
        self.is_running = False
        self.lowCmdWriteThreadPtr = None
        self.crc = CRC()

        # --- Control Flags ---
        self.chirp_started = False  # 是否已經開始 Chirp
        self.ready_to_chirp = False # 是否已經站好，準備接受指令

        # --- Chirp Settings ---
        self.min_freq = 0.1   
        self.max_freq = 10.0  
        self.duration = 20.0  
        self.chirp_amplitude = 1.0
        
        self.generate_chirp_profile()

        # Data logging buffers
        self.log_time = []
        self.log_dof_pos = []
        self.log_des_dof_pos = []

    def generate_chirp_profile(self):
        sample_rate = 1.0 / self.dt
        num_steps = int(self.duration * sample_rate)
        t = np.linspace(0, self.duration, num_steps)
        f0 = self.min_freq; f1 = self.max_freq
        phase = 2 * np.pi * (f0 * t + ((f1 - f0) / (2 * self.duration)) * t ** 2)
        chirp_signal = np.sin(phase)

        self.chirp_traj = np.zeros((num_steps, 12), dtype=np.float32)
        scales = np.array([0.25, 0.5, 0.75] * 4) * self.chirp_amplitude 
        directions = np.array([1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0])

        for i in range(12):
            self.chirp_traj[:, i] = self.default_angles[i] + chirp_signal * scales[i] * directions[i]
        self.num_chirp_steps = num_steps

    def Init(self):
        self.InitLowCmd()
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
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
        print("Initial Success !!!")

    def Start(self):
        if not self.is_running:
            self.is_running = True
            self.lowCmdWriteThreadPtr = threading.Thread(target=self.LowCmdWrite)
            self.lowCmdWriteThreadPtr.start()
            print("Controller Started. Robot will Stand Up.")

    def TriggerChirp(self):
        """ Trigger Chirp """
        if self.ready_to_chirp:
            print(">>> Chirp Triggered by User! Starting collection...")
            self.chirp_started = True
        else:
            print(">>> Robot is not ready yet (Still standing up). Please wait.")

    def ShutDown(self):
        self.is_running = False
        if self.lowCmdWriteThreadPtr is not None:
            self.lowCmdWriteThreadPtr.join()
        
        self.save_data()

    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE; self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF; self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].kp = 0; self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kd = 0; self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowCmdWrite(self):
        chirp_counter = 0
        stand_up_duration = 3.0
        
        last_freq_print_time = time.perf_counter()
        loop_count = 0

        while self.is_running:
            if self.remote_controller.button[KeyMap.A] == 1:
                print("Emergency Stop by User (Button A)")
                break
                
            step_start = time.perf_counter()

            # Calculate and print frequency
            loop_count += 1
            if step_start - last_freq_print_time >= 1.0:
                actual_freq = loop_count / (step_start - last_freq_print_time)
                print(f"Control Frequency: {actual_freq:.2f} Hz")
                loop_count = 0
                last_freq_print_time = step_start

            self.controller_rt += self.dt

            # --- Phase 1: Stand Up (0 ~ 3秒) ---
            if self.controller_rt < stand_up_duration:
                phase = np.tanh(self.controller_rt / 1.2)
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = phase * self.default_angles[i] + (1 - phase) * self.stand_down_joint_pos[i]
                    self.low_cmd.motor_cmd[i].kp = 40 
                    self.low_cmd.motor_cmd[i].dq = 0.0
                    self.low_cmd.motor_cmd[i].kd = 3.0
                    self.low_cmd.motor_cmd[i].tau = 0.0
                
            # --- Phase 2: Hold & Wait for Trigger ---
            elif not self.chirp_started:
                if not self.ready_to_chirp:
                    self.ready_to_chirp = True
                    print("\n[INFO] Robot Stood Up. Holding Position.")
                    print("[INFO] Please type 'chirp' to start data collection.\nCMD :", end='', flush=True)

                # 保持目前的 default_angles
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = self.default_angles[i]
                    self.low_cmd.motor_cmd[i].kp = 40 # 保持較硬的剛性
                    self.low_cmd.motor_cmd[i].dq = 0.0
                    self.low_cmd.motor_cmd[i].kd = 3.0
                    self.low_cmd.motor_cmd[i].tau = 0.0

            # --- Phase 3: Execute Chirp ---
            elif chirp_counter < self.num_chirp_steps:
                target_q = self.chirp_traj[chirp_counter]
                
                for i in range(12):
                    self.low_cmd.motor_cmd[i].q = target_q[i]
                    self.low_cmd.motor_cmd[i].kp = self.kps[i] 
                    self.low_cmd.motor_cmd[i].dq = 0.0
                    self.low_cmd.motor_cmd[i].kd = self.kds[i]
                    self.low_cmd.motor_cmd[i].tau = 0.0

                # Data Collection (Only during Chirp)
                if self.low_state is not None:
                    current_q = np.array([self.low_state.motor_state[i].q for i in range(12)])
                    # 記錄相對於 Chirp 開始的時間，方便對齊
                    self.log_time.append(chirp_counter * self.dt) 
                    self.log_dof_pos.append(current_q)
                    self.log_des_dof_pos.append(target_q.copy())

                chirp_counter += 1
                if chirp_counter % 500 == 0:
                    print(f"Chirp Progress: {chirp_counter}/{self.num_chirp_steps}")

            # --- Phase 4: Finished ---
            else:
                print("Chirp finished.")
                break

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)

            time_until_next_step = self.dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        self.ResetParam()
        if len(self.log_time) > 0:
            self.save_data()
        
        print("\nControl Loop Ended. Press Enter to exit.")

    def ResetParam(self):
        self.controller_rt = 0
        self.is_running = False
        self.chirp_started = False
        self.ready_to_chirp = False

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

        output_dir = pathlib.Path("data/go2")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / "chirp_data.pt"
        
        torch.save(save_dict, file_path)
        print(f"Saved to {file_path}")

if __name__ == '__main__':
    print("WARNING: Ensure robot is HANGING UP.")
    input("Press Enter to continue...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(1, "lo")

    controller = Controller()
    controller.Init()

    # --- 新增 chirp 指令 ---
    command_dict = {
        "st": controller.Start,
        "chirp": controller.TriggerChirp, 
    }

    while True:        
        try:
            cmd = input("CMD (st, chirp, exit): ")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                controller.ShutDown()
                break
        except Exception as e:
            traceback.print_exc()
            break
    sys.exit(-1)