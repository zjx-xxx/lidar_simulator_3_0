import os
import re
import pandas as pd
import shutil
import datetime

# 读取单列363行数据，转置为1行363列，同时清洗转角、路径类型和转向标签
def clean_and_transpose_single_column(df):
    row = df.T.values.flatten()

    # 提取三个标签：倒数第3列是转角 direction，倒数第2列是路径类型 type，最后一列是转向 towards
    label_dir = str(row[-3])
    label_type = str(row[-2])
    label_towards = str(row[-1])

    # 清洗两个个标签，只保留第一个数字，找不到设为0
    def extract_label(val):
        match = re.search(r'[0-9]', val)
        return int(match.group(0)) if match else 0

    row[-2] = extract_label(label_type)
    row[-1] = extract_label(label_towards)

    # 其他列尝试转为数字
    cleaned_row = []
    for i, val in enumerate(row):
        if i in [len(row) - 3, len(row) - 2, len(row) - 1]:
            cleaned_row.append(int(row[i]))
        else:
            try:
                num = float(val)
                if num.is_integer():
                    num = int(num)
                cleaned_row.append(num)
            except:
                cleaned_row.append(float('nan'))
    return pd.DataFrame([cleaned_row])

# 原始标注文件夹
mydata_dir = os.path.join(os.getcwd(), 'labeled/classification')

# 获取文件列表
file_names = os.listdir(mydata_dir) if os.path.exists(mydata_dir) else []
pattern = r'^data(\d+)\.csv$'
matching_files = sorted(
    [f for f in file_names if re.match(pattern, f)],
    key=lambda x: int(re.search(r'\d+', x).group())
)

all_data = []

print(matching_files)

# 加载旧 Data.csv（如果存在）
if os.path.exists("./Data.csv") and os.path.getsize("./Data.csv") > 0:
    try:
        df_existing = pd.read_csv("./Data.csv", header=None)
        all_data.append(df_existing)
        print("已加载旧 Data.csv")
    except pd.errors.EmptyDataError:
        print("Data.csv 文件存在但为空，跳过加载")
elif not os.path.exists("./Data.csv"):
    # 文件不存在，创建空文件
    with open("./Data.csv", "w") as f:
        pass
    print("未发现 Data.csv，已创建空文件")
else:
    print("Data.csv 文件为空，跳过加载")

# 读取并清洗每个标注文件（应为363行×1列）
for file_name in matching_files:
    file_path = os.path.join(mydata_dir, file_name)
    try:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] != 1 or df.shape[0] != 363:
            print(f"{file_name} 不是单列363行数据，跳过")
            continue
        df_cleaned = clean_and_transpose_single_column(df)
        all_data.append(df_cleaned)
        print(f"{file_name} 已处理并添加")
    except Exception as e:
        print(f"{file_name} 处理出错: {e}")

# 合并数据
if all_data:
    merged_data = pd.concat(all_data, axis=0, ignore_index=True)
    merged_data.to_csv('./Data.csv', header=False, index=False)

    # 先整体打乱
    merged_data = merged_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # 划分训练集和测试集（10%测试）
    test_size = 0.1
    test_count = int(len(merged_data) * test_size)

    Test = merged_data.iloc[:test_count, :]
    Train = merged_data.iloc[test_count:, :]

    # 拆分输入与标签
    X_train = Train.iloc[:, :-3]
    Y_dir_train = Train.iloc[:, -3:-2]      # 转角 direction
    Y_type_train = Train.iloc[:, -2:-1]     # 路径类型 type
    Y_towards_train = Train.iloc[:, -1:]   # 转向 towards

    X_test = Test.iloc[:, :-3]
    Y_dir_test = Test.iloc[:, -3:-2]
    Y_type_test = Test.iloc[:, -2:-1]
    Y_towards_test = Test.iloc[:, -1:]

    # 创建输出目录
    os.makedirs('./direction', exist_ok=True)
    os.makedirs('./type', exist_ok=True)
    os.makedirs('./towards', exist_ok=True)

    # 保存训练集
    X_train.to_csv('./X_train.csv', header=False, index=False)
    Y_dir_train.to_csv('./direction/Y_train.csv', header=False, index=False)
    Y_type_train.to_csv('./type/Y_train.csv', header=False, index=False)
    Y_towards_train.to_csv('./towards/Y_train.csv', header=False, index=False)

    # 保存测试集
    X_test.to_csv('./X_test.csv', header=False, index=False)
    Y_dir_test.to_csv('./direction/Y_test.csv', header=False, index=False)
    Y_type_test.to_csv('./type/Y_test.csv', header=False, index=False)
    Y_towards_test.to_csv('./towards/Y_test.csv', header=False, index=False)

    print("训练集和测试集已打乱并保存")
else:
    print("没有数据合并，未生成 Data.csv")

# 归档原始文件
now = datetime.datetime.now()
suffix = now.strftime("%m_%d_%H_%M")
archive_folder = f'./labeledData/{suffix}'
os.makedirs(archive_folder, exist_ok=True)

for file_name in matching_files:
    src_path = os.path.join(mydata_dir, file_name)
    if os.path.exists(src_path):
        shutil.move(src_path, archive_folder)
        print(f"{file_name} 已移动到 {archive_folder}")
