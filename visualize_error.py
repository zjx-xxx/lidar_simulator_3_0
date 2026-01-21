import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
# 用 TkAgg 后端嵌入到 Tk 窗口
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


LIDAR_POINTS = 360  # 前 360 列是 LiDAR 距离


class LidarViewerGUI:
    def __init__(self, master):
        self.master = master
        master.title("LiDAR 点云可视化 (n×364 CSV)")

        # 数据占位
        self.data = None
        self.lidar = None
        self.y_true = None
        self.y_pred = None
        self.N = 0
        self.index = 0

        # 角度数组 (0° ~ 359°)
        self.angles_deg = np.arange(LIDAR_POINTS)
        self.angles_rad = np.deg2rad(90 - self.angles_deg)

        # ================== 上方控制区 ==================
        control_frame = tk.Frame(master)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.btn_open = tk.Button(control_frame, text="打开 CSV 文件", command=self.open_file)
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_prev = tk.Button(control_frame, text="上一条", command=self.prev_sample, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = tk.Button(control_frame, text="下一条", command=self.next_sample, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        # 样本索引滑块
        self.slider = tk.Scale(control_frame, from_=0, to=0, orient=tk.HORIZONTAL,
                               label="样本索引", command=self.on_slider_change, state=tk.DISABLED)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # ================== 中间 Matplotlib 画布 ==================
        fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.set_title("尚未加载数据")

        # 初始化一个空散点
        self.scatter = self.ax.scatter([], [])

        self.canvas = FigureCanvasTkAgg(fig, master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ================== 下方信息标签 ==================
        info_frame = tk.Frame(master)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.label_info = tk.Label(info_frame, text="请先打开一个 n×364 的 CSV 文件。")
        self.label_info.pack(side=tk.LEFT)

    # ------------- 数据加载与处理 -------------
    def open_file(self):
        path = filedialog.askopenfilename(
            title="选择 n×364 CSV 文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            data = np.loadtxt(path, delimiter=",")
        except Exception as e:
            messagebox.showerror("加载失败", f"读取 CSV 时出错：\n{e}")
            return

        if data.ndim == 1:
            data = data.reshape(1, -1)

        N, D = data.shape
        if D < LIDAR_POINTS + 2:
            messagebox.showerror(
                "格式错误",
                f"列数不足，期望至少 {LIDAR_POINTS + 2} 列（360 LiDAR + 2 标签），实际 {D} 列。"
            )
            return

        # 假设：前 360 列 LiDAR，倒数两列 y_true, y_pred
        self.data = data
        self.lidar = data[:, :LIDAR_POINTS]
        self.y_true = data[:, -2]
        self.y_pred = data[:, -1]
        self.N = N
        self.index = 0

        # 更新滑块与按钮状态
        self.slider.config(from_=0, to=self.N - 1, state=tk.NORMAL)
        self.slider.set(0)
        self.btn_prev.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.NORMAL)

        # 设置坐标范围：根据所有样本 LiDAR 距离的最大值
        max_r = float(np.max(self.lidar))
        max_r = max(max_r, 10.0)  # 至少保证一个基础尺度
        self.ax.set_xlim(-max_r, max_r)
        self.ax.set_ylim(-max_r, max_r)

        # 更新第一条样本
        self.update_plot()
        self.label_info.config(text=f"文件已加载：{path}，样本数 N = {N}，列数 D = {D}")

    def lidar_to_xy(self, lidar_row):
        r = lidar_row
        x = r * np.cos(self.angles_rad)
        y = r * np.sin(self.angles_rad)
        return x, y

    # ------------- 绘图更新 -------------
    def update_plot(self):
        if self.data is None or self.N == 0:
            return

        idx = int(self.index)
        idx = max(0, min(self.N - 1, idx))
        self.index = idx

        x, y = self.lidar_to_xy(self.lidar[idx])

        # 更新散点数据
        offsets = np.column_stack((x, y))
        self.scatter.set_offsets(offsets)

        err = abs(self.y_true[idx] - self.y_pred[idx])
        self.ax.set_title(
            f"Sample {idx+1}/{self.N}\n"
            f"Y_true = {self.y_true[idx]:.2f}°,  "
            f"Y_pred = {self.y_pred[idx]:.2f}°\n"
            f"Error = {err:.2f}°"
        )

        # 触发重绘
        self.canvas.draw_idle()

        # 更新下方信息
        self.label_info.config(
            text=f"当前样本：{idx+1}/{self.N}    "
                 f"Y_true = {self.y_true[idx]:.2f}°,  "
                 f"Y_pred = {self.y_pred[idx]:.2f}°,  "
                 f"Error = {err:.2f}°"
        )

    # ------------- GUI 控件回调 -------------
    def prev_sample(self):
        if self.N == 0:
            return
        self.index = (self.index - 1) % self.N
        self.slider.set(self.index)  # 顺带触发 slider 回调 -> update_plot()

    def next_sample(self):
        if self.N == 0:
            return
        self.index = (self.index + 1) % self.N
        self.slider.set(self.index)

    def on_slider_change(self, val):
        if self.N == 0:
            return
        self.index = int(float(val))
        self.update_plot()


def main():
    root = tk.Tk()
    app = LidarViewerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
