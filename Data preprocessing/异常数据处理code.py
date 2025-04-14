import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('D:\\back-up\\pro1\\40万-newprt.csv')

# 步骤1：过滤经纬度在合理范围内的数据
df = df[(df['LAT'] >= -90) & (df['LAT'] <= 90) & (df['LON'] >= -180) & (df['LON'] <= 180)]

# 步骤2：排除非正常航行状态的数据
df = df[~df['Status'].isin([1, 4, 6])]

# 步骤3：删除带有空值的数据行，并消除重复数据
df = df.dropna().drop_duplicates(keep='first')

# 步骤4：删除轨迹点过少的船舶
min_points = 5  # 假设最少轨迹点数为5
df = df.groupby('MMSI').filter(lambda x: len(x) >= min_points)

# 将BaseDateTime转换为日期时间类型
df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])

# 按MMSI分组并计算时间间隔
df['TimeDiff'] = df.groupby('MMSI')['BaseDateTime'].diff()

# 筛选出时间间隔大于两小时的船舶轨迹
time_threshold = pd.Timedelta(hours=0.1)
selected_trajectories = df[df['TimeDiff'] > time_threshold]['MMSI'].unique()

# 为这些轨迹重新分配未使用的MMSI
max_mmsi = df['MMSI'].max()
new_mmsi = max_mmsi + 1
for mmsi in selected_trajectories:
    df.loc[df['MMSI'] == mmsi, 'MMSI'] = new_mmsi
    new_mmsi += 1

# 步骤5：筛选航速低于30节的数据
df = df[df['SOG'] < 30]

# 将结果保存为csv文件
csv_path = 'D:\\back-up\\pro1\\40万-newprt_处理后.csv'
df.to_csv(csv_path)