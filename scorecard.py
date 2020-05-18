from data_to_model import dtom
import pandas as pd
from datetime import datetime
import os

t = '20200511143436'
t = datetime.now().strftime('%Y%m%d%H%M%S') if t is None else t
if not os.path.exists('model_result//wuboyu_modelresult_{}'.format(t)):
    os.makedirs('model_result//wuboyu_modelresult_{}'.format(t))
path = 'model_result//wuboyu_modelresult_{}//test{}'.format(t, t)
data = pd.read_csv('all_data//data.csv')

y_label = 'Creditability'
special_value = [-9999, -1111]  # 特殊值
exclude = []  # 不包含的列
bestks_k = 0
break_type = 1
bin_rate_min = 0
train_perc = 0.7  # 训练集比例
sv_perc = 0.8  # 单一值阀值
num_bins = 20
min_num_bins = 3  # 最小箱数
max_num_bins = 10  # 最大箱数
bad_value = 1  # 坏人的标志
closed_on_right = True
good_value = 0  # 好人的标志
replace_value = 1
comb_type = 'combinning'
woe_stand = 'monotonous'
seed = 1234  # 随机种子
sample_weights = [1, 1]
p_min = 0.05
p0 = 580
pdo = 50
iv_min = 0.02
iv_max = 100
score_k = 10
theta = None
corr_t = 0.8

dtom(path, data, y_label, special_value, exclude, bestks_k, break_type, bin_rate_min, train_perc, sv_perc, num_bins,
     min_num_bins, max_num_bins, bad_value, closed_on_right, good_value, replace_value, comb_type, woe_stand, seed,
     corr_t, iv_min, iv_max, p0, sample_weights, score_k, p_min, theta, pdo)
