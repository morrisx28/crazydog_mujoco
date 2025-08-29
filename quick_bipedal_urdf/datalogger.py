import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch



class DataLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.data = {
            'base_lin_vel': [],
            'base_ang_vel': [],
            'projected_gravity': [],
            'commands': [],
            'dof_pos': [],
            'dof_vel': [],
            'actions': [],
            'torques': [],
            'timestamps': [],
            'base_height': []
        }
        self.start_time = time.time()

    def log(self, obs, actions, torques, base_height):
        current_time = time.time() - self.start_time
        self.data['base_lin_vel'].append(obs[:3].detach().cpu().numpy())  # 3
        self.data['base_ang_vel'].append(obs[3:6].detach().cpu().numpy())  # 3
        self.data['projected_gravity'].append(obs[6:9].detach().cpu().numpy())  # 3
        self.data['commands'].append(obs[9:13].detach().cpu().numpy())  # 4
        self.data['dof_pos'].append(obs[13:17].detach().cpu().numpy())  # 4
        self.data['dof_vel'].append(obs[17:23].detach().cpu().numpy())  # 6
        self.data['actions'].append(actions.detach().cpu().numpy())  # 6
        self.data['torques'].append(torques.copy())
        self.data['base_height'].append(base_height)
        self.data['timestamps'].append(current_time)

    def plot_and_save(self):
        if len(self.data['timestamps']) < 2:
            return

        # Convert lists to numpy arrays
        timestamps = np.array(self.data['timestamps'])
        base_lin_vel = np.stack(self.data['base_lin_vel'])
        base_ang_vel = np.stack(self.data['base_ang_vel'])
        projected_gravity = np.stack(self.data['projected_gravity'])
        commands = np.stack(self.data['commands'])
        dof_pos = np.stack(self.data['dof_pos'])
        dof_vel = np.stack(self.data['dof_vel'])
        actions = np.stack(self.data['actions'])
        torques = np.stack(self.data['torques'])

        # Create figure with multiple subplots
        fig, axes = plt.subplots(8, 1, figsize=(12, 30), sharex=True)
        
        # 1. Base Linear Velocity
        for i in range(3):
            axes[0].plot(timestamps, base_lin_vel[:, i], label=f'Lin Vel {["x", "y", "z"][i]}')
        axes[0].set_title('Base Linear Velocity')
        axes[0].set_ylabel('m/s')
        axes[0].legend()
        axes[0].grid(True)

        # 2. Base Angular Velocity
        for i in range(3):
            axes[1].plot(timestamps, base_ang_vel[:, i], label=f'Ang Vel {["x", "y", "z"][i]}')
        axes[1].set_title('Base Angular Velocity')
        axes[1].set_ylabel('rad/s')
        axes[1].legend()
        axes[1].grid(True)

        # 3. Projected Gravity
        for i in range(3):
            axes[2].plot(timestamps, projected_gravity[:, i], label=f'Grav {["x", "y", "z"][i]}')
        axes[2].set_title('Projected Gravity')
        axes[2].set_ylabel('unit')
        axes[2].legend()
        axes[2].grid(True)

        # 4. Commands
        labels = ['lin_x', 'lin_y', 'ang_z', 'height']
        for i in range(4):
            axes[3].plot(timestamps, commands[:, i], label=labels[i])
        axes[3].set_title('Commands')
        axes[3].set_ylabel('value')
        axes[3].legend()
        axes[3].grid(True)

        # 5. DoF Positions
        for i in range(4):
            axes[4].plot(timestamps, dof_pos[:, i], label=f'DoF Pos {i}')
        axes[4].set_title('DoF Positions')
        axes[4].set_ylabel('rad')
        axes[4].legend()
        axes[4].grid(True)

        # 6. DoF Velocities
        for i in range(6):
            axes[5].plot(timestamps, dof_vel[:, i], label=f'DoF Vel {i}')
        axes[5].set_title('DoF Velocities')
        axes[5].set_ylabel('rad/s')
        axes[5].legend()
        axes[5].grid(True)

        # 7. Actions
        for i in range(6):
            axes[6].plot(timestamps, actions[:, i], label=f'Action {i}')
        axes[6].set_title('Actions')
        axes[6].set_ylabel('value')
        axes[6].legend()
        axes[6].grid(True)

        # 8. Torques
        for i in range(6):
            axes[7].plot(timestamps, torques[:, i], label=f'Torque {i}')
        axes[7].set_title('Joint Torques')
        axes[7].set_xlabel('Time (s)')
        axes[7].set_ylabel('N·m')
        axes[7].legend()
        axes[7].grid(True)

        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, f'plot_resulte.png')
        plt.savefig(plot_path, dpi=100)
        plt.close()

        print("save image")
        # Clear data
        self.data = {k: [] for k in self.data.keys()}
        self.start_time = time.time()
    
    def plot_velocity_traking(self):

        if  len(self.data['timestamps']) < 2:
            return

        timestamps = np.array(self.data['timestamps'])
        base_lin_vel = np.stack(self.data['base_lin_vel'])
        base_ang_vel = np.stack(self.data['base_ang_vel'])
        commands = np.stack(self.data['commands'])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: X-direction linear velocity vs command
        ax1.plot(timestamps, base_lin_vel[:, 0]/2, label='X Velocity (m/s)', color='blue')
        ax1.plot(timestamps, commands[:, 0]/2, label='X Command', color='orange', linestyle='--')
        ax1.set_title('X-Direction Linear Velocity vs Command')
        ax1.set_ylabel('Linear Velocity (m/s)')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Z-direction angular velocity vs command
        ax2.plot(timestamps, base_ang_vel[:, 2]/0.5, label='Z Angular Velocity (rad/s)', color='green')
        ax2.plot(timestamps, commands[:, 2]/0.5, label='Z Command', color='red', linestyle='--')
        ax2.set_title('Z-Direction Angular Velocity vs Command')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.legend()
        ax2.grid(True)

        # Adjust layout and save
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, f'vel_tracking.png')
        plt.savefig(plot_path, dpi=100)
        plt.close()

    def plot_high_tracking(self):

        if  len(self.data['timestamps']) < 2:
            return

        timestamps = np.array(self.data['timestamps'])
        commands = np.stack(self.data['commands'])          # shape: (n_steps, 4)
        base_height = np.array(self.data['base_height'])    # shape: (n_steps,)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot observed height (from base_link z-position)
        ax.plot(timestamps, base_height, label='Observed Height (m)', color='blue')
        
        # Plot commanded height (fourth component of commands)
        ax.plot(timestamps, commands[:, 3]/5, label='Commanded Height', color='orange', linestyle='--')
        
        # Customize plot
        ax.set_title('Robot Height vs Commanded Height')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Height (m)')
        ax.legend()
        ax.grid(True)

        # Save plot
        plot_path = os.path.join(self.save_dir, f'height_vs_cmd.png')
        plt.savefig(plot_path, dpi=100)
        plt.close()


    def clear_buffer(self):
        """Clear all stored data and reset the start time."""
        self.data = {k: [] for k in self.data.keys()}
        self.start_time = time.time()

