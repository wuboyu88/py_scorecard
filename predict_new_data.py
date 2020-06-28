from data_to_bin import dtob
from bin_to_model import btom
from common_util import *
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
manuel_breakpoints_dict = None
# manuel_breakpoints_dict = {'AccountBalance': [1, 1.9, 3.1, 4]}  # 其中1和4分别为训练集的最小值和最大值

info_data_to_bin = dtob(path=path, data=data, y_label=y_label, special_value=special_value, exclude=exclude,
                        bestks_k=bestks_k, break_type=break_type, bin_rate_min=bin_rate_min, train_perc=train_perc,
                        sv_perc=sv_perc, num_bins=num_bins, min_num_bins=min_num_bins, max_num_bins=max_num_bins,
                        bad_value=bad_value, closed_on_right=closed_on_right, good_value=good_value,
                        replace_value=replace_value, comb_type=comb_type, woe_stand=woe_stand, seed=seed,
                        manuel_breakpoints_dict=manuel_breakpoints_dict)

name, glm_fit = btom(path=path, info_data_to_bin=info_data_to_bin, corr_t=corr_t, seed=seed, iv_min=iv_min,
                     iv_max=iv_max, p0=p0, sample_weights=sample_weights, score_k=score_k, p_min=p_min, theta=theta,
                     pdo=pdo)


def predict_new_data(new_data, info_data_to_bin, name, glm_fit):
    new_data = deepcopy(new_data)
    new_data_all = deepcopy(new_data)
    train_index = info_data_to_bin['train_index']
    trainbreakpoints = {k: v['breakpoints'] for k, v in train_index.items()}
    new_data = new_data[[y_label] + list(trainbreakpoints.keys())]

    new_index = allvarindexfortest(new_data.iloc[:, 1:], new_data.iloc[:, 0], trainbreakpoints=trainbreakpoints,
                                   closed_on_right=closed_on_right, break_type=break_type, bad_value=bad_value,
                                   good_value=good_value, special_value=special_value, bin_rate_min=bin_rate_min,
                                   replace_value=replace_value)

    new_index = {k: v for k, v in new_index.items() if k in name}
    new_woematrix = pd.concat([v['sampleinfo']['woe'].to_frame(name=k) for k, v in new_index.items()], axis=1)
    new_data_all['probability'] = glm_fit.predict(new_woematrix)
    return new_data_all


new_data = pd.read_csv('all_data//testdata.csv')
predict_new_data(new_data, info_data_to_bin, name, glm_fit)
