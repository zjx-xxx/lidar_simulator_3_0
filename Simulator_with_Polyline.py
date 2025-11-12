import os
import sys
import math
import random
import datetime
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
import time
import pandas as pd


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# def map_exp(x, factor):
#     return 0.5/(math.exp(- x * factor * x * factor) + 0.5)


#PID控制器
class PID:
    def __init__(self, Kp, Ki, Kd, output_limit=None, integral_limit=None, map=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.map = map

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time  # 计算时间间隔
        if dt == 0:  # 避免除零错误
            dt = 1e-6
        self.integral += error * dt  # 积分项累积
        if self.integral_limit:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)  # 限制积分项
        derivative = (error - self.prev_error) / dt  # 计算微分项
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative  # PID 公式
        if self.output_limit:
            output = np.clip(output, -self.output_limit, self.output_limit)  # 限制输出，防止过大
        # if self.map:
        #     output = map_exp(output, 70)
        self.prev_error = error
        self.last_time = current_time
        return output

class Simulator(tk.Tk):
    # ====== 几何段定义：直线段/圆弧段 ======
    class LineSeg:
        __slots__ = ("p0", "p1", "L", "t0", "t1")

        def __init__(self, p0, p1):
            self.p0 = np.array(p0, dtype=float) # 起点、终点坐标
            self.p1 = np.array(p1, dtype=float)
            self.L = float(np.linalg.norm(self.p1 - self.p0)) # 长度
            self.t0 = 0.0 # 参数起点、终点
            self.t1 = 1.0

    class ArcSeg:
        __slots__ = ("O", "r", "th0", "th1", "ccw", "T0", "T1", "L")

        def __init__(self, O, r, th0, th1, ccw, T0, T1):
            self.O = np.array(O, dtype=float) # 圆心坐标
            self.r = float(r) # 半径
            self.th0 = float(th0) # 起始角（弧度）
            self.th1 = float(th1) # 终止角（弧度）
            self.ccw = bool(ccw) # 方向，顺逆时针
            self.T0 = np.array(T0, dtype=float) # 圆弧起点（切点）
            self.T1 = np.array(T1, dtype=float) # 圆弧终点（切点）

            # 弧长计算
            def wrap(a):
                while a < -np.pi: a += 2 * np.pi
                while a > np.pi: a -= 2 * np.pi
                return a

            d = wrap(self.th1 - self.th0)
            self.L = abs(d) * self.r

    @staticmethod
    def _ang_wrap(a): # 角度计算
        while a < -np.pi: a += 2 * np.pi
        while a > np.pi: a -= 2 * np.pi
        return a

    def __init__(self):
        super().__init__()
        self.title("Simulator")

        # 定义软件界面
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        middle_frame = tk.Frame(main_frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.simulation_canvas = tk.Canvas(left_frame, bg="lightgray", width=800, height=800)
        self.simulation_canvas.pack(fill=tk.BOTH, expand=True)

        # self.canvas_root = self.simulation_canvas.winfo_toplevel()

        self.lidar_canvas = tk.Canvas(middle_frame, bg="lightgray", width=500, height=500)
        self.lidar_canvas.pack(side=tk.TOP)

        log_frame = tk.Frame(middle_frame)
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            log_frame, state="disabled", width=50, height=10, font=("等线", 12, "bold")
        )
        self.log_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        log_text_frame = tk.Frame(log_frame)
        log_text_frame.pack(side=tk.TOP, fill=tk.X)
        log_text_frame.grid_columnconfigure(0, weight=1)
        log_text_frame.grid_columnconfigure(1, weight=1)

        # 折线段参数（折现循迹方案）
        self.completed_polyline_segments = []  # 已完成的折线段（绿色）
        self.follow_step_counter = 0
        self.current_target_point = None

        self.slope = 0


        self.timer_label = tk.Label(
            log_text_frame, text=f"用时: {0:>5d}", font=("等线", 12, "bold")
        )
        self.timer_label.grid(row=0, column=0, pady=5)

        self.collision_count_label = tk.Label(
            log_text_frame, text=f"碰撞: {0:>5d}", font=("等线", 12, "bold")
        )
        self.collision_count_label.grid(row=0, column=1, pady=5)

        self.car_control_velocity_label = tk.Label(
            log_text_frame, text=f"速度: {0:>5d}", font=("等线", 12, "bold")
        )
        self.car_control_velocity_label.grid(row=1, column=0, pady=5)

        self.road_label = tk.Label(
            log_text_frame, text=f"路径分类预测: 无", font=("等线", 12, "bold")
        )
        self.road_label.grid(row=2, column=0, pady=5)

        self.car_control_steering_label = tk.Label(
            log_text_frame, text=f"实际转角: {0:>5d}", font=("等线", 12, "bold")
        )
        self.car_control_steering_label.grid(row=3, column=0, pady=5)

        self.speed_input_label = tk.Label(
            log_text_frame, text="自动速度:", font=("等线", 12, "bold")
        )
        self.speed_input_label.grid(row=1, column=1, pady=5)
        self.speed_input = tk.Entry(log_text_frame, font=("等线", 12, "bold"), width=6)
        self.speed_input.insert(0, "50")
        self.speed_input.grid(row=1, column=2, pady=5)

        self.predicted_angle_label = tk.Label(
            log_text_frame, text="预测舵机角度: 无", font=("等线", 12, "bold")
        )
        self.predicted_angle_label.grid(row=3, column=1, pady=5)

        logo_frame = tk.Frame(right_frame)
        logo_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(right_frame)
        button_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor=tk.N)

        serial_frame = tk.Frame(right_frame)
        serial_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        button_width = 20
        button_height = 2

        # 存储数据记录的序号
        self.recording = False
        try:
            with open('index.txt', 'r') as f:
                self.dataindex = int(f.read())
        except FileNotFoundError:
            self.dataindex = 0
        self.data_dir = "./mydata/raw"

        # 控件打包
        load_default_map_button = tk.Button(
            button_frame,
            text="加载默认地图",
            command=self.load_default_map,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        load_default_map_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        load_map_button = tk.Button(
            button_frame,
            text="加载地图",
            command=self.load_map,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        load_map_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        toggle_map_edit_button = tk.Button(
            button_frame,
            text="开启/关闭地图编辑",
            command=self.toggle_map_edit,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        toggle_map_edit_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        save_map_button = tk.Button(
            button_frame,
            text="保存地图",
            command=self.save_map,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        save_map_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        set_car_pose_button = tk.Button(
            button_frame,
            text="设置小车位置/朝向",
            command=self.set_car_pose,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        set_car_pose_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        toggle_clear_data_button = tk.Button(
            button_frame,
            text="清除数据",
            command=self.toggle_clear_data,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        toggle_clear_data_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        clear_log_button = tk.Button(
            button_frame,
            text="清除日志",
            command=self.clear_log,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        clear_log_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        exit_button = tk.Button(
            button_frame,
            text="退出程序",
            command=self.quit,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        exit_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        undo_polyline_button = tk.Button(
            button_frame,
            text="撤销上一个折线点",
            command=self.undo_last_polyline_point,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        undo_polyline_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        delete_polyline_data_button = tk.Button( # 可能没用？
            button_frame,
            text="撤销折线循迹记录",
            command=self.delete_last_polyline_data,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        delete_polyline_data_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        polyline_button = tk.Button(
            button_frame,
            text="绘制折线路径",
            command=self.toggle_polyline_drawing,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        polyline_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        follow_polyline_button = tk.Button(
            button_frame,
            text="开启折线循迹",
            command=self.toggle_polyline_following,
            width=button_width,
            height=button_height,
            font=("等线", 12, "bold")
        )
        follow_polyline_button.pack(side=tk.TOP, fill=tk.BOTH, pady=5)

        self.start_dataindex = None # 用来记录「在采集轨迹数据时的起始帧索引与结束帧索引」
        self.end_dataindex = None

        logo1_image = Image.open(resource_path("data/logo1.png"))
        logo1_image = ImageTk.PhotoImage(logo1_image)
        logo1_label = tk.Label(logo_frame, image=logo1_image)
        logo1_label.image = logo1_image
        logo1_label.pack()

        logo2_image = Image.open(resource_path("data/logo2.png"))
        logo2_image = ImageTk.PhotoImage(logo2_image)
        logo2_label = tk.Label(logo_frame, image=logo2_image)
        logo2_label.image = logo2_image
        logo2_label.pack()

        self.grid = np.zeros((40, 40), dtype=np.uint8)

        # 新增扰动参数

        # 转向扰动
        self.steering_perturbation_enabled = False  # 是否启用转向扰动
        self.steering_perturbation_prob = 0.02  # 扰动发生概率 (20%)
        self.steering_perturbation_counter = 0  # 扰动计数器
        self.steering_perturbation_interval = 10  # 扰动最小间隔(更新次数)
        self.steering_perturbation_range = 8  # 转向扰动范围(度)

        # 位移扰动
        self.position_perturbation_enabled = False  # 是否启用位移扰动
        self.position_perturbation_prob = 0.02  # 扰动发生概率 (10%)
        self.position_perturbation_counter = 0  # 扰动计数器
        self.position_perturbation_interval = 20  # 扰动最小间隔(更新次数)
        self.position_perturbation_range = 15  # 位移扰动范围(像素)

        self.in_map_edit = False
        self.in_keyboard_control = False
        self.in_auto_control = False
        self.in_timing = False
        self.in_set_car_pose = False
        self.data = np.zeros((1, 360))

        self.key_state = {'w': False, 's': False, 'a': False, 'd': False}
        self.key_press_time = {}

        self.car = None
        self.car_pose = None
        self.car_control_value = np.zeros((2, 1), dtype=np.float32)
        self.road = -1
        self.car_size = 30
        self.car_image = Image.open(resource_path("data/car.png")).resize((self.car_size, self.car_size))
        self.car_photo = ImageTk.PhotoImage(self.car_image)

        self.path_pts = None  # 折线采样点数组 (N,2)
        self.path_s = None  # 折线累计弧长数组 (N,)

        self.velocity_command = ""
        self.steering_command = ""
        self.command_state = -1
        self.deltatheta = 0

        self.map_type = 0
        self.simulation_time_interval = 0.02
        self.lidar_scan_time_interval = 0.2
        self.time_interval = 0.2
        self.lidar_result = []

        self.collision_count = 0
        self.collided_obstacles = set()

        self.timestamp = None
        self.load_default_map()
        self.update()
        self.lidar_scan()
        self.direction = None
        self.grid_size = 20

        # 自动控制状态标志初始化
        self.in_auto_control = False
        self.polyline_following = False
        self.polyline_points = []
        self.polyline_index = 0
        self.drawing_polyline = False

        self.curr_seg_idx = 0

        self.polyline_record_start = None
        self.polyline_record_end = None
        self.polyline_segments = []  # 保存多个折线段

        self.passed_points = []  # 已跟随的点（绿色）
        self.current_follow_point = None  # 当前跟随的前视点（红色）
        self.steering_pid = PID(Kp=6.0, Ki=0.1, Kd=0.2, output_limit=50, map=1)

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def draw_map(self): # 这里存储了numpy坐标和canvas实际坐标的对照关系
        self.simulation_canvas.delete("all")
        grid_size = 20
        for row in range(40):
            for col in range(40):
                x1 = col * grid_size
                y1 = row * grid_size
                x2 = x1 + grid_size
                y2 = y1 + grid_size
                if self.grid[row][col] == 1:  # 障碍
                    self.simulation_canvas.create_rectangle(
                        x1, y1, x2, y2, fill="orange", outline="gray"
                    )
                elif self.grid[row][col] == 2:  # 终点
                    self.simulation_canvas.create_rectangle(
                        x1, y1, x2, y2, fill="red", outline="gray"
                    )
                else:  # 空地
                    self.simulation_canvas.create_rectangle(
                        x1, y1, x2, y2, fill="gray", outline="gray"
                    )

    def update(self): # 这里可以修改更新频率，更新时间和频率
        start_time = time.time()
        self.update_velocity()
        self.update_car()
        end_time = time.time()
        delay_time = self.simulation_time_interval - (end_time - start_time)
        self.after(int(max(delay_time, 0) * 1000), self.update) # 通过对自己的回调来控制时钟

    def update_car(self):
        if self.car is not None:
            grid_size = 20

            # 小车运行的运动学更新
            next_car_pose = self.car_pose.copy()
            next_car_pose[2, 0] += self.car_control_value[0, 0] / (0.75 * self.car_size) * np.tan(
                np.deg2rad(self.car_control_value[1, 0])
            ) * self.simulation_time_interval
            next_car_pose[0, 0] += self.car_control_value[0, 0] * np.cos(
                next_car_pose[2, 0]) * self.simulation_time_interval
            next_car_pose[1, 0] += self.car_control_value[0, 0] * np.sin(
                next_car_pose[2, 0]) * self.simulation_time_interval
            car_x = next_car_pose[0, 0]
            car_y = next_car_pose[1, 0]
            car_radius = self.car_size / 2
            new_collided_obstacles = set()

            # 小车的碰撞检测
            for row in range(40):
                for col in range(40):
                    if self.grid[row][col] == 1:  # 障碍物存在
                        # 计算障碍物的边界
                        obstacle_left = col * grid_size
                        obstacle_right = (col + 1) * grid_size
                        obstacle_top = row * grid_size
                        obstacle_bottom = (row + 1) * grid_size
                        # 找到小车中心与障碍物最近点
                        nearest_x = max(obstacle_left, min(car_x, obstacle_right))
                        nearest_y = max(obstacle_top, min(car_y, obstacle_bottom))
                        # 计算小车中心到障碍物边界的距离
                        dx = car_x - nearest_x
                        dy = car_y - nearest_y
                        # 如果距离小于小车半径，则发生碰撞
                        if (dx * dx + dy * dy) < (car_radius * car_radius):
                            new_collided_obstacles.add((row, col))
                            # 限制小车位置，使其与障碍物相切
                            distance = (dx ** 2 + dy ** 2) ** 0.5
                            if distance > 0:  # 避免除以零
                                overlap = car_radius - distance
                                car_x += (dx / distance) * overlap
                                car_y += (dy / distance) * overlap
            # 如果有新碰撞，更新碰撞次数
            if new_collided_obstacles and not self.collided_obstacles:
                self.collision_count += 1
            if not (self.collided_obstacles and new_collided_obstacles):
                self.car_pose[0, 0] = car_x
                self.car_pose[1, 0] = car_y
                self.car_pose[2, 0] = next_car_pose[2, 0]
            self.collided_obstacles = new_collided_obstacles
            self.collision_count_label.config(text=f"碰撞: {self.collision_count:>5d}")
            if self.in_auto_control:
                if self.check_collision(self.car_pose[0, 0], self.car_pose[1, 0], "destination"):
                    self.log("到达终点")
                    self.close_auto_control()
                self.timer_label.config(text=f"用时: {int(time.time() - self.timestamp):>5d}")
            self.draw_car()

    def load_default_map(self):
        if self.in_auto_control:
            self.log("请先关闭自动控制")
            return
        if self.in_keyboard_control:
            self.log("请先关闭键盘控制")
            return
        if self.in_map_edit:
            self.close_map_edit()
        self.grid = np.load(resource_path("data/map_default.npy"))
        self.draw_map()
        self.log("加载默认地图")
        if self.car is not None:
            self.simulation_canvas.delete(self.car)
            self.car = None

    def load_map(self):
        if self.in_auto_control:
            self.log("请先关闭自动控制")
            return
        if self.in_keyboard_control:
            self.log("请先关闭键盘控制")
            return
        if self.in_map_edit:
            self.close_map_edit()
        file_path = filedialog.askopenfilename(
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        if file_path:
            self.grid = np.load(file_path)
            self.draw_map()
            self.log(f"从{file_path}成功加载地图")
            if self.car is not None:
                self.simulation_canvas.delete(self.car)
                self.car = None

    def toggle_map_edit(self):
        if self.in_map_edit:
            self.close_map_edit()
        else:
            self.open_map_edit()

    def open_map_edit(self):
        if self.in_keyboard_control:
            self.log("请先关闭键盘控制")
            return
        if self.in_auto_control:
            self.log("请先关闭自动控制")
            return
        self.in_map_edit = True
        self.simulation_canvas.bind("<Button-1>", self.handle_map_edit_click)
        self.simulation_canvas.bind("<ButtonPress-3>", self.handle_set_destination_click)
        self.simulation_canvas.bind("<B1-Motion>", self.handle_map_edit_drag)
        self.simulation_canvas.bind("<B3-Motion>", self.handle_set_destination_drag)
        self.log("开启地图编辑")

    def close_map_edit(self):
        self.in_map_edit = False
        self.simulation_canvas.unbind("<Button-1>")
        self.simulation_canvas.unbind("<ButtonPress-3>")
        self.simulation_canvas.unbind("<B1-Motion>")
        self.simulation_canvas.unbind("<B3-Motion>")
        self.log("关闭地图编辑")

    def handle_map_edit_drag(self, event):
        grid_size = 20
        row = event.y // grid_size
        col = event.x // grid_size
        if 0 <= row < 40 and 0 <= col < 40:
            if self.map_type == 1 and self.grid[row][col] != 2:
                self.grid[row][col] = 1
                if self.car is not None:
                    if self.check_collision(self.car_pose[0, 0], self.car_pose[1, 0]):
                        self.grid[row][col] = 0
            elif self.map_type == 0 and self.grid[row][col] != 2:
                self.grid[row][col] = 0
            self.draw_single_block(row, col)

    def handle_map_edit_click(self, event):
        grid_size = 20  # 计算每个格子的像素大小
        row = event.y // grid_size
        col = event.x // grid_size
        if 0 <= row < 40 and 0 <= col < 40:
            if self.grid[row][col] == 0:
                self.grid[row][col] = 1
                self.map_type = 1
                if self.car is not None:
                    if self.check_collision(self.car_pose[0, 0], self.car_pose[1, 0]):
                        self.grid[row][col] = 0
            elif self.grid[row][col] == 1:
                self.grid[row][col] = 0
                self.map_type = 0
            else:
                self.map_type = 2
            self.draw_single_block(row, col)

    def handle_set_destination_click(self, event):
        grid_size = 20
        row = event.y // grid_size
        col = event.x // grid_size
        if 0 <= row < 40 and 0 <= col < 40:
            if self.grid[row][col] == 0:
                self.grid[row][col] = 2
                self.map_type = 2
            elif self.grid[row][col] == 1:
                self.map_type = 1
            else:
                self.grid[row][col] = 0
                self.map_type = 0
            self.draw_single_block(row, col)

    def handle_set_destination_drag(self, event):
        grid_size = 20
        row = event.y // grid_size
        col = event.x // grid_size
        if 0 <= row < 40 and 0 <= col < 40:
            if self.map_type == 2 and self.grid[row][col] != 1:
                self.grid[row][col] = 2
            elif self.map_type == 0 and self.grid[row][col] != 1:
                self.grid[row][col] = 0
            self.draw_single_block(row, col)

    def draw_single_block(self, row, col):
        grid_size = 20
        x1 = col * grid_size
        y1 = row * grid_size
        x2 = x1 + grid_size
        y2 = y1 + grid_size
        if self.grid[row][col] == 1:
            color = "orange"
        elif self.grid[row][col] == 2:
            color = "red"
        else:
            color = "gray"
        self.simulation_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

    def save_map(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
        )
        if file_path:
            np.save(file_path, self.grid)
            self.log(f"地图保存到: {file_path}")

    def set_car_pose(self):
        if self.in_auto_control:
            self.log("请先关闭自动控制")
            return
        if self.in_keyboard_control:
            self.log("请先关闭键盘控制")
            return
        if self.in_map_edit:
            self.log("请先关闭地图编辑")
            return
        if self.car is not None:
            self.simulation_canvas.delete(self.car)
            self.car = None
        self.log("设置小车位置/朝向")
        self.simulation_canvas.bind("<Button-1>", self.handle_set_car_pose_click)

    def check_collision(self, car_x, car_y, collision_type="obstacle"):
        grid_size = 20
        car_radius = self.car_size / 2
        map_type = 1 if collision_type == "obstacle" else 2
        for row in range(40):
            for col in range(40):
                if self.grid[row][col] == map_type:  # 障碍物存在
                    nearest_x = max(col * grid_size, min(car_x, (col + 1) * grid_size))
                    nearest_y = max(row * grid_size, min(car_y, (row + 1) * grid_size))
                    # 计算小车中心到障碍物边界的距离
                    dx = car_x - nearest_x
                    dy = car_y - nearest_y
                    # 如果距离小于小车半径，则发生碰撞
                    if (dx * dx + dy * dy) < (car_radius * car_radius):
                        return True
        return False

    def handle_set_car_pose_click(self, event):
        if not self.in_set_car_pose:
            if self.check_collision(event.x, event.y):
                self.log("小车位置与障碍物冲突，请重新设置")
                return
            self.car_pose = np.array([event.x, event.y, 0], dtype=np.float32).reshape(3, 1)
            self.in_set_car_pose = True
        else:
            self.car_pose[2, 0] = np.arctan2(event.y - self.car_pose[1, 0], event.x - self.car_pose[0, 0])
            self.in_set_car_pose = False
            self.draw_car()
            self.simulation_canvas.unbind("<Button-1>")

    def draw_car(self):
        if self.car is not None:
            self.simulation_canvas.delete(self.car)

        car_image = self.car_image.rotate(np.rad2deg(-self.car_pose[2, 0] - np.pi / 2), expand=True)
        self.car_photo = ImageTk.PhotoImage(car_image)
        self.car = self.simulation_canvas.create_image(
            self.car_pose[0, 0],
            self.car_pose[1, 0],
            image=self.car_photo,
            anchor="center"
        )

    # 键盘控制
    def toggle_keyboard_control(self):
        if self.in_keyboard_control:
            self.close_keyboard_control()
        else:
            self.open_keyboard_control()

    def open_keyboard_control(self):
        if self.in_auto_control:
            self.log("请先关闭自动控制")
            return
        if self.car is None:
            self.log("请先设置小车位置/朝向")
            return
        if self.in_map_edit:
            self.close_map_edit()
        self.in_keyboard_control = True
        self.collision_count = 0
        self.bind("<KeyPress>", self.on_key_press)
        self.bind("<KeyRelease>", self.on_key_release)
        self.log("开启键盘控制")

    def close_keyboard_control(self):
        self.in_keyboard_control = False
        self.simulation_canvas.unbind("<KeyPress>")
        self.simulation_canvas.unbind("<KeyRelease>")
        self.log("关闭键盘控制")

    def on_key_press(self, event):
        key = event.keysym.lower()
        if key in self.key_state and not self.key_state[key]:
            self.key_state[key] = True
            self.key_press_time[key] = time.time()

    def on_key_release(self, event):
        key = event.keysym.lower()
        if key in self.key_state:
            self.key_state[key] = False

    def update_velocity(self): # 上传所有的控制量，包括转角、速度
        # 在计算控制量之前添加转向扰动
        if (self.steering_perturbation_enabled and
                self.recording and
                self.steering_perturbation_counter >= self.steering_perturbation_interval and
                random.random() < self.steering_perturbation_prob):

            # 生成随机扰动角度
            steering_perturb = random.uniform(-self.steering_perturbation_range, self.steering_perturbation_range)

            # === 基于车头与目标方向夹角的类高斯缩放扰动 ===
            car_x, car_y = self.car_pose[0, 0], self.car_pose[1, 0]
            theta_car = self.car_pose[2, 0]  # 当前航向（弧度）

            # 当前目标点（例如你的 path_pts 或 polyline_points）
            target_x, target_y = self.path_pts[self.polyline_index]
            dx = target_x - car_x
            dy = target_y - car_y

            # ---- 计算车头与目标方向夹角 ----
            target_theta = np.arctan2(dy, dx)
            # wrap到(-pi, pi]
            angle_diff = (target_theta - theta_car + np.pi) % (2 * np.pi) - np.pi
            angle_diff_deg = np.degrees(angle_diff)  # 转成角度便于理解

            # ---- 高斯缩放：夹角越大扰动越小 ----
            sigma = 25.0  # 控制衰减范围 (度)
            min_scale = 0.3  # 弯道时扰动下限
            max_scale = 1.0  # 正对目标时扰动上限

            scale = min_scale + (max_scale - min_scale) * np.exp(
                - (angle_diff_deg ** 2) / (2 * sigma ** 2)
            )

            # ---- 应用扰动 ----
            self.car_control_value[1, 0] += steering_perturb * scale

            # 日志记录
            self.log(f"施加转向扰动: {steering_perturb:.2f}°")

            # 重置计数器
            self.steering_perturbation_counter = 0
        else:
            self.steering_perturbation_counter += 1

        # 在计算控制量之前添加位移扰动
        if (self.position_perturbation_enabled and
                self.recording and
                self.position_perturbation_counter >= self.position_perturbation_interval and
                random.random() < self.position_perturbation_prob):

            # === 生成随机位移扰动（按车头-目标夹角缩放）===
            car_x, car_y = self.car_pose[0, 0], self.car_pose[1, 0]
            theta_car = self.car_pose[2, 0]

            # 当前目标点
            target_x, target_y = self.path_pts[self.polyline_index]
            dx = target_x - car_x
            dy = target_y - car_y

            # ---- 计算车头与目标方向夹角 ----
            target_theta = math.atan2(dy, dx)
            angle_diff = (target_theta - theta_car + math.pi) % (2 * math.pi) - math.pi
            angle_diff_deg = math.degrees(angle_diff)

            # ---- 类高斯缩放系数（夹角越大扰动越小）----
            sigma = 25.0  # 控制扰动衰减范围（单位：度）
            min_scale = 0.3  # 急转弯时扰动最小比例
            max_scale = 1.0  # 直行时扰动最大比例
            gauss_scale = min_scale + (max_scale - min_scale) * math.exp(
                - (angle_diff_deg ** 2) / (2 * sigma ** 2)
            )

            # ---- 生成随机方向与距离（乘上高斯缩放）----
            angle = random.uniform(0, 2 * math.pi)
            base_distance = random.uniform(3, self.position_perturbation_range)
            distance = base_distance * gauss_scale

            # ---- 应用位移扰动 ----
            self.car_pose[0, 0] += distance * math.cos(angle)
            self.car_pose[1, 0] += distance * math.sin(angle)

            # 日志记录
            self.log(f"施加位移扰动: 方向={math.degrees(angle):.1f}°, 距离={distance:.2f}px")

            # 重置计数器
            self.position_perturbation_counter = 0
        else:
            self.position_perturbation_counter += 1


        if self.in_keyboard_control:
            for key, pressed in self.key_state.items():
                if pressed:
                    press_duration = time.time() - self.key_press_time.get(key, time.time())
                    self.key_press_time[key] = time.time()
                    if key == 'w':
                        self.car_control_value[0, 0] += press_duration * 200
                    elif key == 's':
                        self.car_control_value[0, 0] -= press_duration * 200
                    elif key == 'a':
                        self.car_control_value[1, 0] -= press_duration * 30
                    elif key == 'd':
                        self.car_control_value[1, 0] += press_duration * 30

        if self.in_keyboard_control:
            velocity_decrease_flag = True
            steering_decrease_flag = True
            for key, pressed in self.key_state.items():
                if pressed:
                    if key in ['w', 's']:
                        velocity_decrease_flag = False
                    if key in ['a', 'd']:
                        steering_decrease_flag = False
            if velocity_decrease_flag:
                self.car_control_value[0, 0] *= 0.95
            if steering_decrease_flag:
                self.car_control_value[1, 0] *= 0.85
        else:
            self.car_control_value[0, 0] *= 0.95
            self.car_control_value[1, 0] *= 0.85

        self.car_control_value[0, 0] = np.maximum(np.minimum(self.car_control_value[0, 0], 200), -200)
        self.car_control_value[1, 0] = np.maximum(np.minimum(self.car_control_value[1, 0], 30), -30)

        self.car_control_velocity_label.config(text=f"速度: {int(self.car_control_value[0, 0]):>5d}")
        self.car_control_steering_label.config(text=f"实际转角: {int(self.car_control_value[1, 0]):>5d}")

    def lidar_scan(self):
        start_time = time.time()
        if self.car is not None:
            max_distance = 200
            x1, y1 = self.car_pose[0, 0], self.car_pose[1, 0]
            theta = self.car_pose[2, 0]
            self.deltatheta = self.deltatheta + theta
            self.lidar_result = []
            for angle_increment in range(360):
                angle = theta + np.deg2rad(angle_increment)
                x2 = x1 + max_distance * np.cos(angle)
                y2 = y1 + max_distance * np.sin(angle)
                distance = self.lidar_detect(x1, y1, x2, y2)
                self.lidar_result.append((angle_increment, distance))
            self.draw_lidar()
        end_time = time.time()
        delay_time = self.lidar_scan_time_interval - (end_time - start_time)
        self.after(int(max(delay_time, 0) * 1000), self.lidar_scan)
        # self.lidar_result.append(self.car_control_value[1,0])

        df = pd.DataFrame(self.lidar_result, columns=['degree', 'distance'])
        if len(df) != 0 and self.recording:
            # 取原始distance列
            second_column = df['distance']
            # 当前转向角（取整）
            steering_value = int(self.car_control_value[1, 0])
            # 当前斜率（整数）
            slope_value = int(self.slope)
            # 合并为一个Series
            combined = pd.concat([
                second_column,
                pd.Series([steering_value], name="steering"),
                pd.Series([slope_value], name="slope")
            ], ignore_index=True)
            # 生成文件路径
            file_path = './mydata/raw'
            while os.path.exists(file_path+f'/data{self.dataindex}.csv'):
                self.dataindex += 1
            # 保存为一列（或一行）
            if file_path and not os.path.exists(file_path):
                os.makedirs(file_path)
            combined.to_csv(file_path + f'/data{self.dataindex}.csv', index=False, header=False)
            # 更新索引记录
            self.dataindex += 1
            with open('index.txt', 'w') as f:
                f.write(str(self.dataindex))

    def _nearest_on_polyline(self, P, pts, start_idx, window=30):
        """
        在以 start_idx 为中心的局部窗内，找 P 到折线最近的小段。
        返回：idx, t, Q, tan, nor
            idx: 段起点索引（该段是 pts[idx] -> pts[idx+1]）
            t  : 段内参数 [0,1]
            Q  : 最近点坐标 (2,)
            tan: 段单位切向 (2,)
            nor: 段单位法向 (2,) = R90(tan)
        """
        n = len(pts)
        if n < 2:
            raise ValueError("path_pts too short")

        i0 = max(0, start_idx - window)
        i1 = min(n - 2, start_idx + window)

        best = None
        best_pack = None
        for i in range(i0, i1 + 1):
            A = pts[i]
            B = pts[i + 1]
            v = B - A
            L2 = float(np.dot(v, v)) + 1e-9
            t = float(np.dot(P - A, v) / L2)
            t_clamped = max(0.0, min(1.0, t))
            Q = A + t_clamped * v
            d2 = float(np.dot(P - Q, P - Q))
            if (best is None) or (d2 < best):
                best = d2
                tan = v / (np.sqrt(L2) + 1e-9)
                nor = np.array([-tan[1], tan[0]], dtype=float)
                best_pack = (i, t_clamped, Q, tan, nor)
        return best_pack

    def _advance_along_polyline(self, pts, s, idx, t, Ld):
        """
        从“idx 段内参数 t 对应的位置”出发，沿折线前推弧长 Ld，返回前视点 Q_los。
        """
        # 当前弧长位置
        A = pts[idx]
        B = pts[idx + 1]
        v = B - A
        L = float(np.linalg.norm(v)) + 1e-9
        s_here = s[idx] + t * L
        s_target = s_here + max(0.0, Ld)

        # 二分/顺序推进：这里用顺序推进，足够简单
        n = len(pts)
        if s_target >= s[-1]:
            return pts[-1].copy()

        # 找到 s_target 落在哪一段
        lo, hi = idx, n - 2
        # 直接顺序（代码简单，性能也够用）
        k = idx
        while k < n - 1 and s[k + 1] < s_target:
            k += 1
        # 在线性插值
        seg_len = (s[k + 1] - s[k]) + 1e-9
        lam = float((s_target - s[k]) / seg_len)
        return (1 - lam) * pts[k] + lam * pts[k + 1]

    def toggle_polyline_drawing(self):
        # 关闭绘制：收尾当前段，但不清空历史段
        if self.drawing_polyline:
            self.simulation_canvas.unbind("<Button-1>")
            self.drawing_polyline = False
            if self.current_segment and len(self.current_segment) >= 2:
                self.polyline_segments.append(self.current_segment[:])
            self.current_segment = None
            self.log("关闭折线路径绘制")
            self.redraw_polyline()
            return

        # 开始绘制：开启新的一段
        self.current_segment = []
        self.polyline_index = 0
        self.simulation_canvas.bind("<Button-1>", self.add_polyline_point)
        self.drawing_polyline = True
        self.log("点击画布开始绘制【一段】折线路径；再次点击按钮结束该段")

    def add_polyline_point(self, event):
        if self.current_segment is None:
            self.current_segment = []

        tx, ty = self.find_target_point(event)
        # 注意：用 is None，避免 0 被误判
        if tx is None or ty is None:
            self.log("未找到有效道路中线，已忽略点击")
            return

        # 第一个点：把“起点”设在道路中线
        if len(self.current_segment) == 0:
            self.current_segment.append((tx, ty))
            self.redraw_polyline()
            return

        # 其余点：从末点 -> (tx,ty) 生成采样（直线 + 1/4 椭圆弧 + 直线）
        self.process_target_point(tx, ty)

        # 关键：不要 append (event.x, event.y)！
        self.redraw_polyline()

    def process_target_point(self, target_x, target_y):
        """
        从 current_segment 的末点出发，沿主方向走到最近路口（宽度突增），
        在路口处用 1/4 椭圆弧转到副方向，再走直线到 (target_x, target_y)。
        约束：一次点击只允许转一个直角弯；若检测到多路口或信息不足则拒绝本次。
        最终把采样点以 ~10px 的间距追加到 self.current_segment。
        """

        import math, numpy as np

        # ---------- 基本检查 ----------
        if not self.current_segment or len(self.current_segment) == 0:
            self.log("当前没有起点，请先添加一个折线点")
            return

        # 末点（起点）
        p0x, p0y = map(float, self.current_segment[-1])
        tx, ty = float(target_x), float(target_y)
        gs = int(getattr(self, "grid_size", 20))
        H, W = getattr(self, "grid", np.zeros((40, 40))).shape
        ds = 20.0  # 每 ~10px 采样一个点（与你原代码一致）

        # 取 raycast 函数（兼容 _raycast_axis / raycast_axis）
        raycast = getattr(self, "_raycast_axis", None)
        if raycast is None:
            raycast = getattr(self, "raycast_axis", None)
        if raycast is None:
            self.log("缺少射线函数 _raycast_axis/raycast_axis")
            return

        # ---------- 小工具 ----------
        def corridor_width_and_mid(x, y):
            """返回水平/竖直两方向的宽度与中线点 ((wh, (mxh,myh)), (wv, (mxv,myv)))；若某向无墙对则宽度为 +inf。"""
            L = raycast(x, y, 'x', -1)
            R = raycast(x, y, 'x', +1)
            U = raycast(x, y, 'y', -1)
            D = raycast(x, y, 'y', +1)
            if L is not None and R is not None:
                wh = abs(float(R[0]) - float(L[0]))
                mid_h = (0.5 * (float(R[0]) + float(L[0])), float(y))
            else:
                wh, mid_h = float('inf'), None
            if U is not None and D is not None:
                wv = abs(float(D[1]) - float(U[1]))
                mid_v = (float(x), 0.5 * (float(D[1]) + float(U[1])))
            else:
                wv, mid_v = float('inf'), None
            return (wh, mid_h), (wv, mid_v)

        def find_first_junction_from(x, y, axis, sign, max_steps=2000):
            """
            沿 axis='x'/'y' 以像素步长 step 前进，检测“与运动正交的宽度”是否出现**明显突增**。
            返回：(corner_point(x,y), width_enter, width_exit)，若失败返回 (None, None, None)。
            判据采用 相对阈值 + 绝对阈值：ratio>=1.35 且 (w_new - w_old) >= 0.6*grid_size
            """
            step = 1.0
            # 初始宽度（正交方向）
            (wh, _), (wv, _) = corridor_width_and_mid(x, y)
            w_old = wv if axis == 'x' else wh
            if not np.isfinite(w_old):
                return (None, None, None)

            # 逐步前进
            cx, cy = float(x), float(y)
            for i in range(int(max_steps)):
                if axis == 'x':
                    cx += sign * step
                else:
                    cy += sign * step

                # 越界
                r = int(cy // gs)
                c = int(cx // gs)
                if not (0 <= r < H and 0 <= c < W):
                    break

                (wh, _), (wv, _) = corridor_width_and_mid(cx, cy)
                w_new = wv if axis == 'x' else wh
                if not np.isfinite(w_new):
                    continue

                # 判据：相对 + 绝对 突增
                rel_ok = (w_new / max(1e-6, w_old)) >= 1.35
                abs_ok = (w_new - w_old) >= (0.6 * gs)
                if rel_ok and abs_ok:
                    # 路口角点取“前一位置”（避免跨越多格）
                    corner = (cx - sign * step, cy) if axis == 'x' else (cx, cy - sign * step)
                    return (corner, w_old, w_new)

                w_old = w_new

            return (None, None, None)

        def append_line_samples(p_from, p_to, step=ds):
            """p_from->p_to 线段采样（不重复起点）"""
            x0, y0 = p_from
            x1, y1 = p_to
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 1e-6:
                return
            n = max(1, int(L // step))
            xs = np.linspace(x0, x1, n + 1)[1:]
            ys = np.linspace(y0, y1, n + 1)[1:]
            for xk, yk in zip(xs, ys):
                self.current_segment.append((float(xk), float(yk)))

        def sample_quarter_ellipse_arc(corner, dir_enter, dir_exit, a, b, n=5):
            """
            生成 1/4 轴对齐椭圆弧，满足：
              - 与进入水平/竖直线 C^1 切接；
              - 不经过角点 corner=(x0,y0)；
              - a,b > 0 分别取进入/退出通道半宽 * 安全系数。
            屏幕坐标：x→右正, y→下正。dir_* ∈ {'E','W','N','S'} 且相互正交。
            返回：[(x1,y1), ...] 采样点（不含直线段端点，避免重复）
            """
            x0, y0 = corner

            # 中心与 [tA,tB]
            if dir_enter == 'E' and dir_exit == 'S':
                cx, cy = x0, y0 + b
                tA, tB = 3 * math.pi / 2, 2 * math.pi
            elif dir_enter == 'E' and dir_exit == 'N':
                cx, cy = x0, y0 - b
                tA, tB = math.pi / 2 , 0.0
            elif dir_enter == 'W' and dir_exit == 'S':
                cx, cy = x0, y0 + b
                tA, tB = 3 * math.pi / 2, math.pi
            elif dir_enter == 'W' and dir_exit == 'N':
                cx, cy = x0, y0 - b
                tA, tB = math.pi / 2, math.pi
            elif dir_enter == 'N' and dir_exit == 'E':
                cx, cy = x0 + a, y0
                tA, tB = math.pi, 3 * math.pi / 2
            elif dir_enter == 'N' and dir_exit == 'W':
                cx, cy = x0 - a, y0
                tA, tB = 0, -1 * math.pi / 2
            elif dir_enter == 'S' and dir_exit == 'E':
                cx, cy = x0 + a, y0
                tA, tB = math.pi, math.pi / 2
            elif dir_enter == 'S' and dir_exit == 'W':
                cx, cy = x0 - a, y0
                tA, tB = 0, math.pi / 2
            else:
                return []

            # 采样：弧端与直线端点重合，避免重复 corner/端点，这里取(1..n-1)
            th = np.linspace(tA, tB, max(2, n + 1))
            xs = cx + a * np.cos(th)
            ys = cy + b * np.sin(th)
            pts = list(zip(xs[1:-1], ys[1:-1]))  # 去掉两端，防止与直线重复
            return [(float(x), float(y)) for x, y in pts]

        def cardinal_from_motion(dx, dy, axis):
            """根据主/副方向与符号给出 E/W/N/S """
            if axis == 'x':
                return 'E' if dx >= 0 else 'W'
            else:
                return 'S' if dy >= 0 else 'N'

        # ---------- 特殊情形：与末点严格同 x 或同 y（直线到达） ----------
        eps = 1e-6
        if abs(tx - p0x) < eps or abs(ty - p0y) < eps:
            append_line_samples((p0x, p0y), (tx, ty), ds)
            return

        # ---------- 决定“主方向→副方向”的转弯顺序 ----------
        # 优先沿当前末段方向；如果没有末段，按 |dx| vs |dy| 决定
        if len(self.current_segment) >= 2:
            px2, py2 = map(float, self.current_segment[-2])
            vx_prev = p0x - px2
            vy_prev = p0y - py2
            if abs(vx_prev) >= abs(vy_prev):
                primary = 'x'
                secondary = 'y'
            else:
                primary = 'y'
                secondary = 'x'
        else:
            dx0, dy0 = (tx - p0x), (ty - p0y)
            if abs(dx0) >= abs(dy0):
                primary = 'x'
                secondary = 'y'
            else:
                primary = 'y'
                secondary = 'x'

        # 主/副方向行进的符号（朝向目标）
        sgn_x = +1 if (tx - p0x) >= 0 else -1
        sgn_y = +1 if (ty - p0y) >= 0 else -1

        # ---------- 第1段：从 p0 沿主方向走到最近路口 ----------
        if primary == 'x':
            corner1, w_enter, w_exit = find_first_junction_from(p0x, p0y, 'x', sgn_x)
            if corner1 is None:
                # 若一路无路口，允许直接走到“与目标同 x”并转弯（但仍保证只转一次）
                corner1 = (tx, p0y)
                # 估个进入/退出宽度（就地测一次）
                (wh, _), (wv, _) = corridor_width_and_mid(*corner1)
                w_enter = wv  # primary='x'，正交宽度是竖直宽度
                w_exit = w_enter
            # 记录直线段
            append_line_samples((p0x, p0y), corner1, ds)
            # 椭圆弧 enter/exit 方向
            dir_enter = cardinal_from_motion(sgn_x, 0.0, 'x')
            dir_exit = cardinal_from_motion(0.0, (ty - corner1[1]), 'y')
            self.log(f"directions: {dir_enter}, {dir_exit}")
            # 计算椭圆半轴（进入半宽/退出半宽，各乘安全系数）
            # 进入方向的半轴取“进入通道的半宽”，退出方向同理
            if not np.isfinite(w_enter) or not np.isfinite(w_exit):
                self.log("无法评估路口宽度，取消本次转弯")
                return
            (w_exit, _), (_, _) = corridor_width_and_mid(tx, ty)  # 水平半轴
            a = 0.5 * w_exit   # 水平半轴
            b = 0.5 * w_enter  # 垂直半轴
            self.log(f"a={a}, b={b}")
            # 椭圆弧采样
            arc_pts = sample_quarter_ellipse_arc(corner1, dir_enter, dir_exit, a, b, n=5)
            for (xk, yk) in arc_pts:
                self.current_segment.append((float(xk), float(yk)))
            # 第3段：从弧终点到目标（副方向→终点）
            if len(arc_pts) > 0:
                p2 = arc_pts[-1]
            else:
                p2 = corner1
            append_line_samples(p2, (tx, ty), ds)

        else:  # primary == 'y'
            corner1, w_enter, w_exit = find_first_junction_from(p0x, p0y, 'y', sgn_y)
            if corner1 is None:
                corner1 = (p0x, ty)
                (wh, _), (wv, _) = corridor_width_and_mid(*corner1)
                w_enter = wh  # primary='y'，正交宽度是水平宽度
                w_exit = w_enter
            append_line_samples((p0x, p0y), corner1, ds)
            dir_enter = cardinal_from_motion(0.0, sgn_y, 'y')
            dir_exit = cardinal_from_motion((tx - corner1[0]), 0.0, 'x')
            self.log(f"directions: {dir_enter}, {dir_exit}")
            if not np.isfinite(w_enter) or not np.isfinite(w_exit):
                self.log("无法评估路口宽度，取消本次转弯")
                return
            (_, _), (w_exit, _) = corridor_width_and_mid(tx, ty)  # 水平半轴
            a = 0.5 * w_enter  # 水平半轴
            b = 0.5 * w_exit  # 垂直半轴
            self.log(f"a={a}, b={b}")
            arc_pts = sample_quarter_ellipse_arc(corner1, dir_enter, dir_exit, a, b, n=5)
            for (xk, yk) in arc_pts:
                self.current_segment.append((float(xk), float(yk)))
            if len(arc_pts) > 0:
                p2 = arc_pts[-1]
            else:
                p2 = corner1
            append_line_samples(p2, (tx, ty), ds)

        # ---------- 防跨越多个路口（软约束） ----------
        # 简单检查：起点到 corner1 的直线路径上，统计“宽度突增”的次数，大于1则提示。
        # （为控制复杂度，这里不做强硬回退；如需严格限制，可将本次操作回滚）
        def count_widen_along(p_from, axis, sign):
            cx, cy = p_from
            step = 1.0
            (wh, _), (wv, _) = corridor_width_and_mid(cx, cy)
            w_old = wv if axis == 'x' else wh
            if not np.isfinite(w_old): return 99
            cnt = 0
            for _ in range(2000):
                if axis == 'x':
                    cx += sign * step
                else:
                    cy += sign * step
                r = int(cy // gs)
                c = int(cx // gs)
                if not (0 <= r < H and 0 <= c < W): break
                (wh, _), (wv, _) = corridor_width_and_mid(cx, cy)
                w_new = wv if axis == 'x' else wh
                if not np.isfinite(w_new): continue
                if (w_new / max(1e-6, w_old)) >= 1.35 and (w_new - w_old) >= 0.6 * gs:
                    cnt += 1
                w_old = w_new
                # 到达 corner1 停止
                if abs(cx - corner1[0]) < step * 0.6 and abs(cy - corner1[1]) < step * 0.6:
                    break
            return cnt

        # 仅用于日志告警，不做强制回退
        if primary == 'x':
            cnt = count_widen_along((p0x, p0y), 'x', sgn_x)
        else:
            cnt = count_widen_along((p0x, p0y), 'y', sgn_y)
        if cnt > 1:
            self.log(f"警告：检测到可能跨越了 {cnt} 个路口（建议缩短一次点击的跨度）")

    def raycast_axis(self, x, y, axis='x', sign=+1, max_cells=2000):
        gs = int(self.grid_size)
        H, W = self.grid.shape
        r = int(y // gs)
        c = int(x // gs)
        if not (0 <= r < H and 0 <= c < W):
            return None

        for k in range(0, max_cells):
            rr = r + (k if axis == 'y' else 0) * sign
            cc = c + (k if axis == 'x' else 0) * sign
            if not (0 <= rr < H and 0 <= cc < W):
                break
            if self.grid[rr, cc] == 1:
                if axis == 'x':
                    # 水平射线：命中列 cc 的墙格
                    # sign < 0（向左）：取该墙格的“右边界” (cc+1)*gs
                    # sign > 0（向右）：取该墙格的“左边界”  cc*gs
                    px = ((cc + 1) if sign < 0 else cc) * gs
                    py = y
                else:
                    # 垂直射线：命中行 rr 的墙格
                    # sign < 0（向上）：取该墙格的“下边界” (rr+1)*gs
                    # sign > 0（向下）：取该墙格的“上边界”  rr*gs
                    px = x
                    py = ((rr + 1) if sign < 0 else rr) * gs
                return float(px), float(py)
        return None

    def find_target_point(self, event):
        x, y = int(event.x), int(event.y)
        gs = int(self.grid_size)
        H, W = self.grid.shape

        # 四向最近墙
        L = self.raycast_axis(x, y, 'x', -1)
        R = self.raycast_axis(x, y, 'x', +1)
        U = self.raycast_axis(x, y, 'y', -1)
        D = self.raycast_axis(x, y, 'y', +1)

        # 需要左右成对 或 上下成对，才能定义通道
        horiz_ok = (L is not None) and (R is not None)
        vert_ok = (U is not None) and (D is not None)

        if not horiz_ok and not vert_ok:
            self.log("请将光标移动到道路内部（两侧/上下都有墙）")
            return None, None
        self.log(f"{abs(R[0] - L[0])}, {D[1] - U[1]}")
        # 计算水平/竖直两个方向的通道宽度与中线
        if horiz_ok and abs(R[0] - L[0]) <= 105 < abs(D[1] - U[1]):
            # 道路沿竖直方向
            mid_v = ((R[0] + L[0]) * 0.5, y)
        elif vert_ok and abs(R[0] - L[0]) > 105 >= abs(D[1] - U[1]):
            mid_v = (x, (D[1] + U[1]) * 0.5)
        else:
            self.log("请将光标移动到道路上，而不是路口上")
            return None, None
        target_x, target_y = mid_v

        # 容错：门洞太窄或异常
        if not np.isfinite(target_x) or not np.isfinite(target_y):
            self.log("未能确定有效的道路中线，请调整光标位置")
            return None, None

        return float(target_x), float(target_y)

    def toggle_polyline_following(self):
        if not self.polyline_following:
            self.start_polyline_following()
        else:
            self.close_polyline_following()

    def start_polyline_following(self):
        self.recording = True
        # 把“当前段”（若在绘制中）也纳入临时总段
        self.polyline_record_start = self.dataindex
        # --- 新增：准备可跟随路径 ---
        self._prepare_path_for_follow(ds=20.0)
        if self.path_pts is None or len(self.path_pts) < 2:
            self.log("路径为空或太短，无法循迹")
            return
        self.log(f"path_pts的长度{len(self.path_pts)}")
        self.curr_idx = 0
        if not hasattr(self, "speed_pid"):
            self.speed_pid = PID(Kp=1.0, Ki=0.0, Kd=0.2, output_limit=10, integral_limit=5)

        self.polyline_following = True
        self.follow_step_counter = 0
        self.followed_path_points = []
        self.log("开启折线循迹（基于离散折线）")
        self.follow_polyline()

    def _prepare_path_for_follow(self, ds=20.0):
        """把 polyline_segments (+ current_segment) 合并为均匀采样的 path_pts/path_s。"""
        segs = self.polyline_segments[:]
        if self.current_segment and len(self.current_segment) >= 2:
            segs.append(self.current_segment)
            self.log(f"current_segment:{len(self.current_segment)}")
            self.log(f"polyline_segment:{len(segs[0])}")

        if not segs:
            self.path_pts = None
            self.path_s = None
            return

        pts = []

        def densify(A, B, step):
            Ax, Ay = A
            Bx, By = B
            L = math.hypot(Bx - Ax, By - Ay)
            if L < 1e-6:
                return []
            n = max(1, int(L // step))
            xs = np.linspace(Ax, Bx, n + 1)[1:]  # 不重复起点
            ys = np.linspace(Ay, By, n + 1)[1:]
            return list(zip(xs, ys))

        # 逐段合并 + 细分
        last = None
        for seg in segs:
            for i, P in enumerate(seg):
                P = (float(P[0]), float(P[1]))
                if last is None:
                    pts.append(P)
                    last = P
                else:
                    # 先补一段 last->P 的均匀采样
                    pts.extend(densify(last, P, ds))
                    last = P

        # 去重/防抖（相邻重复点）
        cleaned = [pts[0]]
        for q in pts[1:]:
            if (q[0] - cleaned[-1][0]) ** 2 + (q[1] - cleaned[-1][1]) ** 2 > 1e-6:
                cleaned.append(q)

        self.path_pts = np.array(cleaned, dtype=float)
        # 累计弧长
        dif = np.diff(self.path_pts, axis=0)
        segL = np.hypot(dif[:, 0], dif[:, 1])
        self.path_s = np.concatenate([[0.0], np.cumsum(segL)])

    def close_polyline_following(self):
        self.polyline_following = False
        self.recording = False
        self.log("关闭折线循迹")
        self.car_control_value[:, 0] = 0
        self.polyline_record_end = self.dataindex
        self.log(f"折线循迹结束，记录数据索引：{self.polyline_record_start} 到 {self.polyline_record_end}")

        # 清空路径容器（蓝色历史段、当前段、已完成段/跟随轨迹）
        self.polyline_segments.clear()  # 历史折线段
        self.current_segment = []  # 当前未收尾的折线段
        if hasattr(self, "completed_polyline_segments"):
            self.completed_polyline_segments.clear()  # 若你用绿色标记已完成段
        if hasattr(self, "followed_path_points"):
            self.followed_path_points.clear()  # 已走过的轨迹点（若用于可视化）

        # 清空用于循迹的采样路径与状态
        self.path_pts = None
        self.path_s = None
        self.polyline_index = 0
        self.current_target_point = None

        # 触发重绘（只画地图和车，不再画折线）
        self.redraw_polyline()

    def follow_polyline(self):
        if not self.polyline_following:
            return
        if self.polyline_index >= len(self.path_pts):
            self.close_polyline_following()
            self.log("self.polyline_index >= len(self.path_pts)")
            return

        car_x, car_y = self.car_pose[0, 0], self.car_pose[1, 0]
        theta_car = self.car_pose[2, 0]



        # 取当前目标与向量
        target_x, target_y = self.path_pts[self.polyline_index]
        dx = target_x - car_x
        dy = target_y - car_y
        distance_to_target = np.hypot(dx, dy)
        self.current_target_point = (target_x, target_y)

        # 车头方向
        car_face_x = np.cos(theta_car)
        car_face_y = np.sin(theta_car)

        # === 记录轨迹及路径方向信息 ===
        self.follow_step_counter += 1
        if self.follow_step_counter % 3 == 0:
            self.followed_path_points.append((car_x, car_y))

        if self.polyline_index < len(self.path_pts)-1:
            self.slope = -1 * int(max(abs((self.path_pts[self.polyline_index+1][1] - self.path_pts[self.polyline_index][1])), 1e-2)/max(abs((self.path_pts[self.polyline_index+1][0] - self.path_pts[self.polyline_index][0])), 1e-2))

        # —— 背向跳点（保留 while）——
        # 仅在“确实在身后（点积<0）且距离足够远”时跳，避免起步就一路跳到末端
        max_hops = 10
        hops = 0
        dot_product = np.dot([car_face_x, car_face_y], [dx, dy])

        while (dot_product < 0.0 and distance_to_target > 15.0
               and self.polyline_index < len(self.path_pts) - 1
               and hops < max_hops):
            self.polyline_index += 1
            target_x, target_y = self.path_pts[self.polyline_index]
            dx = target_x - car_x
            dy = target_y - car_y
            distance_to_target = np.hypot(dx, dy)
            dot_product = np.dot([car_face_x, car_face_y], [dx, dy])
            hops += 1

        # —— 前视距离（保持你原有逻辑）——
        speed_control = self.speed_pid.compute(distance_to_target)
        speed_control = np.clip(speed_control, -10, 10)

        base_lookahead = 10.0
        variable_component = 0.5 * abs(speed_control)
        lookahead_distance = np.clip(base_lookahead + variable_component, 10, 25)

        if distance_to_target > 1e-5:
            unit_dx = dx / distance_to_target
            unit_dy = dy / distance_to_target
            actual_lookahead = min(lookahead_distance, distance_to_target)
            lookahead_x = car_x + unit_dx * actual_lookahead
            lookahead_y = car_y + unit_dy * actual_lookahead
        else:
            lookahead_x, lookahead_y = target_x, target_y

        self.current_follow_point = (lookahead_x, lookahead_y)

        # —— 航向误差与转向 ——
        dx_la = lookahead_x - car_x
        dy_la = lookahead_y - car_y
        target_theta = np.arctan2(dy_la, dx_la)
        heading_error = (target_theta - theta_car + np.pi) % (2 * np.pi) - np.pi
        heading_error = np.rad2deg(heading_error)

        steering_control = self.steering_pid.compute(heading_error)
        speed_mapped = np.interp(speed_control, [-10, 10], [-80, 80])

        self.car_control_value[0, 0] = speed_mapped
        self.car_control_value[1, 0] = steering_control

        # —— 到点切换（保持原逻辑）——
        if distance_to_target < 10 and self.polyline_index < len(self.path_pts) - 1:
            self.polyline_index += 1
            target_x, target_y = self.path_pts[self.polyline_index]
            dx = target_x - car_x
            dy = target_y - car_y

        # 可视化与调度
        self.draw_follow_points()
        self.after(50, self.follow_polyline)

        # —— 终点判定（保持原逻辑）——
        if self.polyline_index >= len(self.path_pts) - 1:
            final_dx = self.path_pts[-1][0] - car_x
            final_dy = self.path_pts[-1][1] - car_y
            final_dist = np.hypot(final_dx, final_dy)
            if final_dist < 5:
                self.close_polyline_following()
                return

    def draw_follow_points(self):
        self.simulation_canvas.delete("follow_point")

        # （可选）画最终跟随用的 path_pts
        if getattr(self, "path_pts", None) is not None and len(self.path_pts) >= 2:
            for i in range(1, len(self.path_pts)):
                x1, y1 = self.path_pts[i - 1]
                x2, y2 = self.path_pts[i]
                self.simulation_canvas.create_line(
                    x1, y1, x2, y2, fill="#66aaff", width=2, tags="follow_point"
                )

        # 实际走过轨迹：绿色
        for i in range(1, len(self.followed_path_points)):
            x1, y1 = self.followed_path_points[i - 1]
            x2, y2 = self.followed_path_points[i]
            self.simulation_canvas.create_line(
                x1, y1, x2, y2, fill='green', width=2, tags="follow_point"
            )

        # 前视点：红
        if self.current_follow_point:
            x, y = self.current_follow_point
            self.simulation_canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5, fill='red', outline='white', width=2, tags="follow_point")

        # 最近点：黄
        if self.current_target_point:
            x, y = self.current_target_point
            self.simulation_canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4, fill='yellow', outline='black', width=1, tags="follow_point")

    def lidar_detect(self, x1, y1, x2, y2):
        grid_size = 20
        x1_grid, y1_grid = int(x1 // grid_size), int(y1 // grid_size)
        x2_grid, y2_grid = int(x2 // grid_size), int(y2 // grid_size)
        dx = x2 - x1
        dy = y2 - y1

        x_crossings = []
        y_crossings = []
        min_distance = 200
        if dx != 0:
            x_steps = range(min(x1_grid, x2_grid) + 1, max(x1_grid, x2_grid) + 1)
            x_crossings = [x * grid_size for x in x_steps]
        if dy != 0:
            y_steps = range(min(y1_grid, y2_grid) + 1, max(y1_grid, y2_grid) + 1)
            y_crossings = [y * grid_size for y in y_steps]
        for xc in x_crossings:
            t = (xc - x1) / dx
            yc = y1 + t * dy
            row = math.floor(yc / grid_size) if dy >= 0 else math.ceil(yc / grid_size) - 1
            col = math.floor(xc / grid_size) if dx >= 0 else math.ceil(xc / grid_size) - 1
            if 0 <= row < 40 and 0 <= col < 40 and self.grid[row][col] == 1:
                distance = np.sqrt((xc - x1) ** 2 + (yc - y1) ** 2)
                min_distance = min(distance, min_distance)
        for yc in y_crossings:
            t = (yc - y1) / dy
            xc = x1 + t * dx
            row = math.floor(yc / grid_size) if dy >= 0 else math.ceil(yc / grid_size) - 1
            col = math.floor(xc / grid_size) if dx >= 0 else math.ceil(xc / grid_size) - 1
            if 0 <= row < 40 and 0 <= col < 40 and self.grid[row][col] == 1:
                distance = np.sqrt((xc - x1) ** 2 + (yc - y1) ** 2)
                min_distance = min(distance, min_distance)
        return int(min_distance)

    def draw_lidar(self):
        self.lidar_canvas.delete("all")
        center_x, center_y = 250, 250  # 假设雷达画布的中心
        radius = 220  # 圆的半径，缩小到220

        self.lidar_canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            outline="black",
        )

        for i in range(12):
            angle = i * 30
            end_x = center_x + radius * np.cos(np.deg2rad(angle))
            end_y = center_y + radius * np.sin(np.deg2rad(angle))
            self.lidar_canvas.create_line(
                center_x, center_y, end_x, end_y, fill="black"
            )

        for angle, distance in self.lidar_result:
            if distance == 200:
                continue
            angle = np.deg2rad(angle - 90)
            end_x = center_x + 1.1 * distance * np.cos(angle)
            end_y = center_y + 1.1 * distance * np.sin(angle)
            self.lidar_canvas.create_oval(
                end_x - 1, end_y - 1, end_x + 1, end_y + 1, fill="red", outline="red"
            )

        arrow_length = 20
        end_arrow_x = center_x
        end_arrow_y = center_y - arrow_length
        self.lidar_canvas.create_line(
            center_x,
            center_y,
            end_arrow_x,
            end_arrow_y,
            arrow=tk.LAST,
            fill="blue",
            width=2,
        )

        for i in range(12):
            angle = i * 30
            distance = next((d for a, d in self.lidar_result if a == angle), None)
            if distance is not None:
                text_x = center_x + (radius + 20) * np.cos(np.deg2rad(angle - 90))
                text_y = center_y + (radius + 20) * np.sin(np.deg2rad(angle - 90))
                self.lidar_canvas.create_text(
                    text_x, text_y, text=f"{distance}", fill="blue"
                )


    def toggle_clear_data(self):
        if not os.listdir(self.data_dir):
            popup = tk.Toplevel(self)  # 创建一个新的窗口
            popup.title("清除数据")
            popup.geometry("200x100")

            label = tk.Label(popup, text="文件夹已经为空")
            label.pack(pady=10)

            close_button = tk.Button(popup, text="关闭", command=popup.destroy)
            close_button.pack()
        else:
            popup = tk.Toplevel(self)  # 创建一个新的窗口
            popup.title("清除数据")
            popup.geometry("200x100")

            label = tk.Label(popup, text="您确定要删除文件夹中的所有数据吗？")
            label.pack(pady=10)

            enter_button = tk.Button(popup, text="确认", command=self.clear_data)
            enter_button.pack()

    def clear_data(self):
        directory = self.data_dir
        for filename in os.listdir(directory):
            if filename != "raw.md":
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):  # 仅删除文件，不删除子目录
                    os.remove(file_path)
        self.log("已经删除所有数据")
        return

    def clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    def undo_last_polyline_point(self):
        # 先撤销当前段
        if self.current_segment and len(self.current_segment) > 0:
            self.current_segment.pop()
            self.redraw_polyline()
            self.log("已撤销当前段最近一个折线点")
            return
        # 再撤销历史段
        if self.polyline_segments:
            if len(self.polyline_segments[-1]) > 1:
                self.polyline_segments[-1].pop()
                if len(self.polyline_segments[-1]) < 2:
                    self.polyline_segments.pop()
            else:
                self.polyline_segments.pop()
            self.redraw_polyline()
            self.log("已撤销上一段的最近一个折线点")
            return
        self.log("没有可以撤销的折线点")

    # --- 重绘路径 ---
    def redraw_polyline(self):
        self.draw_map()
        # 历史段（蓝色）
        for seg in self.polyline_segments:
            for i in range(1, len(seg)):
                x1, y1 = seg[i - 1]
                x2, y2 = seg[i]
                self.simulation_canvas.create_line(x1, y1, x2, y2,
                                                   fill="blue", width=3)
        # 当前段（蓝色，未收尾）
        if self.current_segment:
            for i in range(1, len(self.current_segment)):
                x1, y1 = self.current_segment[i - 1]
                x2, y2 = self.current_segment[i]
                self.simulation_canvas.create_line(x1, y1, x2, y2,
                                                   fill="blue", width=3)
        if self.car:
            self.draw_car()

    def delete_last_polyline_data(self):
        if self.polyline_record_start is None or self.polyline_record_end is None:
            self.log("没有找到可以删除的折线循迹记录段")
            return
        deleted = 0
        for i in range(self.polyline_record_start, self.polyline_record_end):
            file_path = f"./mydata/raw/data{i}.csv"
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted += 1
        self.log(
            f"已删除 {deleted} 个文件（data{self.polyline_record_start} 到 data{self.polyline_record_end - 1}）")
        self.polyline_record_start = None
        self.polyline_record_end = None


if __name__ == "__main__":
    Simulator().mainloop()
