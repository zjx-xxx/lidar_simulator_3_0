import numpy as np
import tkinter as tk
import pandas as pd
import os
import re

def draw_lidar_from_csv(csv_files):
    last_type_label = None
    last_direction_label = None
    index = 0
    total_files = len(csv_files)
    if total_files == 0:
        print("没有找到 CSV 文件。")
        return

    root = tk.Tk()
    root.title("Lidar Labeling: 路口类型 + 转向方向")
    # 在 root = tk.Tk() 之后，保存按钮之前或之后都行
    angle_file_var = tk.StringVar(value="")
    slope_file_var = tk.StringVar(value="")

    tk.Label(root, text="文件中的转角(°)：").pack()
    tk.Label(root, textvariable=angle_file_var, fg="blue").pack()
    tk.Label(root, text="文件中的斜率：").pack()
    tk.Label(root, textvariable=slope_file_var, fg="blue").pack()

    # 输入框：路径类型（0-3）
    tk.Label(root, text="路径类型（0:直行, 1:拐弯, 2:T型, 3:十字）").pack()
    type_entry = tk.Entry(root)
    type_entry.pack()

    # 输入框：转向方向（0-左，1-右，2-前）
    tk.Label(root, text="转向方向（0:前, 1:左, 2:右）").pack()
    direction_entry = tk.Entry(root)
    direction_entry.pack()

    type_entry.focus_force()
    save_button = tk.Button(root, text="保存标签")
    save_button.pack()

    canvas_size = 500
    lidar_canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
    lidar_canvas.pack()

    center_x, center_y = canvas_size // 2, canvas_size // 2
    radius = 220
    lidar_canvas.create_oval(
        center_x - radius, center_y - radius, center_x + radius, center_y + radius, outline="black"
    )

    def update_canvas(data):
        lidar_canvas.delete("points")
        for angle in range(360):
            distance = data[angle]
            if distance > 0:
                radian = np.deg2rad(angle - 90)
                end_x = center_x + 1.1 * distance * np.cos(radian)
                end_y = center_y + 1.1 * distance * np.sin(radian)
                lidar_canvas.create_oval(
                    end_x - 1, end_y - 1, end_x + 1, end_y + 1, fill="red", outline="red", tags="points"
                )

    def load_next_file():
        nonlocal index
        if index >= total_files:
            print("所有文件标注完成。")
            root.destroy()
            return

        csv_file = csv_files[index]
        type_entry.delete(0, tk.END)
        direction_entry.delete(0, tk.END)
        type_entry.focus_force()

        try:
            data = pd.read_csv(f'./mydata/raw/{csv_file}', header=None).values.flatten()
            if len(data) < 361:
                raise ValueError("CSV 文件数据不完整")
            # data: flatten 后的一列
            # 只用前360个点画图
            update_canvas(data[:360])

            # 如果有第361、362行，显示为“文件中的转角/斜率”
            angle_in_file = data[360] if len(data) >= 361 else None
            slope_in_file = data[361] if len(data) >= 362 else None
            angle_file_var.set("" if angle_in_file is None else f"{float(angle_in_file):.6g}")
            slope_file_var.set("" if slope_in_file is None else f"{float(slope_in_file):.6g}")

        except Exception as e:
            print(f"无法读取 {csv_file}: {e}")
            index += 1
            load_next_file()
            return

    def save_label(event=None):
        nonlocal index, last_type_label, last_direction_label
        type_label = type_entry.get().strip()
        direction_label = direction_entry.get().strip()

        # 如果为空则继承上一次的值
        if not type_label and last_type_label is not None:
            type_label = last_type_label
        if not direction_label and last_direction_label is not None:
            direction_label = last_direction_label

        csv_file = csv_files[index]

        # 舍弃条件
        if type_label == "9" or direction_label == "9":
            os.remove(f'./mydata/raw/{csv_file}')
            print(f"文件 {csv_file} 被删除（舍弃）")
            index += 1
            load_next_file()
            return

        # 合法性检查
        if type_label not in {'0', '1', '2', '3'}:
            print(f"无效路径类型输入: {type_label}")
            return
        if direction_label not in {'0', '1', '2'}:
            print(f"无效转向方向输入: {direction_label}")
            return

        try:
            data = pd.read_csv(f'./mydata/raw/{csv_file}', header=None).values.flatten()
            if len(data) < 360:
                print("CSV 文件数据不足 360 行")
                return

            lidar_360 = data[:361].astype(float)#若只GPT，还有转角

            # 你的两个输入：type_label / direction_label（已在上面做了校验）
            data_with_labels = np.append(lidar_360, [int(type_label), int(direction_label)]).reshape(-1, 1)  # 360+2=362
            df = pd.DataFrame(data_with_labels)
            save_path = f'./mydata/labeled/classification/{csv_file}'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False, header=False)
            os.remove(f'./mydata/raw/{csv_file}')
            print(f"已保存：{csv_file}，路径类型={type_label}，方向={direction_label}")

            # 保存这次输入为上一次输入
            last_type_label = type_label
            last_direction_label = direction_label

        except Exception as e:
            print(f"保存标签时出错: {e}")
            return

        index += 1
        load_next_file()

    save_button.config(command=save_label)
    type_entry.bind("<Return>", lambda e: direction_entry.focus_set())
    direction_entry.bind("<Return>", save_label)
    root.bind("<Return>", save_label)

    load_next_file()
    root.mainloop()


# 获取所有文件
csv_files = []
raw_data_dir = './mydata/raw'

if os.path.exists(raw_data_dir):
    file_names = os.listdir(raw_data_dir)
else:
    print(f"目录 {raw_data_dir} 不存在！")
    file_names = []

pattern = r'^data(\d+)\.csv$'
matching_files = []

for file in file_names:
    match = re.match(pattern, file)
    if match:
        file_index = int(match.group(1))
        matching_files.append((file_index, file))

matching_files.sort()
csv_files = [file for _, file in matching_files]

draw_lidar_from_csv(csv_files)
