import pygame
import numpy as np
class control_gamepad:
    def __init__(self,num_commands: int, lin_vel_x_range: np.array, lin_vel_y_range: np.array, 
                 ang_vel_range: np.array, height_target_range: np.array,command_scale=None):
        pygame.init()
        pygame.joystick.init()
        self.use_gamepad = True

        # get number of joystick
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("no gamepad,open keyboard window")
            self.use_gamepad = False
            screen_width = 300
            screen_height = 480
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            image_path = "pictures/gamepad_key.png"
            image_center = (160, 240)
            pygame.display.set_caption("COntrol pannel (This use your keyboard)")
            try:
                image_surface = pygame.image.load(image_path)
            except pygame.error as e:
                print(f"Cant load image: {image_path}")
                print(e)
                pygame.quit()
                exit()
            image_rect = image_surface.get_rect()
            image_rect.center = image_center
            self.screen.fill((255, 255, 255)) 
            self.screen.blit(image_surface, image_rect)
            pygame.display.flip() 
        else:
            # deafult setting for first joystick
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"link gamepad: {self.joystick.get_name()}")
        self.num_commands = num_commands
        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_range = ang_vel_range
        self.height_target_range = height_target_range
        self.commands = np.zeros(self.num_commands)
        self.command_scale = command_scale
        if self.command_scale is None:
            self.command_scale = [1.0, 1.0, 1.0 ,0.05]
        self.startlog_flag = False
    
    def get_commands(self):
        pygame.event.pump()
        reset_flag = False
        plot_flag = False
        if self.use_gamepad:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.JOYBUTTONDOWN:
                    # print(f"按钮 {event.button} 被按下。")
                    match event.button:
                        case 0:
                            reset_flag=True
                # elif event.type == pygame.JOYBUTTONUP:
                #     print(f"按钮 {event.button} 被释放。")
                elif event.type == pygame.JOYAXISMOTION:
                    # print(f"轴 {event.axis},{event.value}")
                    match event.axis:
                        case 1: #ly 
                            self.commands[0] = -event.value * self.command_scale[0]
                        case 2: #rx 
                            self.commands[2] = -event.value * self.command_scale[2]
                        case 3: #lt increase height 
                            self.commands[3] += (event.value+1) * self.command_scale[3] 
                        case 4: #ry reduce height
                            self.commands[3] -= (event.value+1) * self.command_scale[3]
        else:
            quiet_walking = 1.0
            for event in pygame.event.get():  
                if event.type == pygame.QUIT:  
                    running = False
                elif event.type == pygame.KEYDOWN:  
                    match event.key:
                        case pygame.K_w:
                            self.commands[0] = self.command_scale[0] * quiet_walking
                        case pygame.K_s:
                            self.commands[0] = -self.command_scale[0] * quiet_walking
                        case pygame.K_a:
                            self.commands[1] = self.command_scale[1] * quiet_walking
                        case pygame.K_d:
                            self.commands[1] = -self.command_scale[1] * quiet_walking
                        case pygame.K_q:
                            self.commands[2] = +self.command_scale[2] * quiet_walking
                        case pygame.K_e:
                            self.commands[2] = -self.command_scale[2] * quiet_walking
                        case pygame.K_SPACE:
                            self.commands[3] = +self.command_scale[3] * 100.0
                        case pygame.K_c:
                            self.commands[3] = -self.command_scale[3] * 100.0
                        case pygame.K_LCTRL:
                            self.commands[3] = -self.command_scale[3] * 100.0
                        case pygame.K_r:
                            reset_flag=True
                        case pygame.K_p:
                            plot_flag=True
                        case pygame.K_t:
                            self.startlog_flag=True
                            print("start log")
                        case pygame.K_LSHIFT:
                            quiet_walking=0.2
                elif event.type == pygame.KEYUP:  
                    match event.key:
                        case pygame.K_w:
                            self.commands[0] = 0
                        case pygame.K_s:
                            self.commands[0] = 0
                        case pygame.K_a:
                            self.commands[1] = 0
                        case pygame.K_d:
                            self.commands[1] = 0
                        case pygame.K_q:
                            self.commands[2] = 0
                        case pygame.K_e:
                            self.commands[2] = 0
                if plot_flag == True:
                    self.startlog_flag = False
        self.commands_clip()
        return self.commands,reset_flag,plot_flag,self.startlog_flag
    
    def commands_clip(self):
        # lin_vel_x
        if self.commands[0] < self.lin_vel_x_range[0] * self.command_scale[0]:
            self.commands[0] = self.lin_vel_x_range[0] * self.command_scale[0]
        elif self.commands[0] > self.lin_vel_x_range[1] * self.command_scale[0]:
            self.commands[0] = self.lin_vel_x_range[1] * self.command_scale[0]

        #lin_vel_y
        if self.commands[1] < self.lin_vel_y_range[0] * self.command_scale[1]:
            self.commands[1] = self.lin_vel_y_range[0] * self.command_scale[1]
        elif self.commands[1] > self.lin_vel_y_range[1] * self.command_scale[1]:
            self.commands[1] = self.lin_vel_y_range[1] * self.command_scale[1]

        #ang_vel
        if self.commands[2] < self.ang_vel_range[0] * self.command_scale[2]:
            self.commands[2] = self.ang_vel_range[0] * self.command_scale[2]
        elif self.commands[2] > self.ang_vel_range[1] * self.command_scale[2]:
            self.commands[2] = self.ang_vel_range[1] * self.command_scale[2]

        #base_heigh
        if self.commands[3] < self.height_target_range[0]:
            self.commands[3] = self.height_target_range[0]
        elif self.commands[3] > self.height_target_range[1]:
            self.commands[3] = self.height_target_range[1]