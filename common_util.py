import pandas as pd
import numpy as np
from copy import deepcopy
import pandas.api.types as types
from openpyxl import load_workbook, drawing


def varinfo(var, special_value):
    """
    变量信息
    :param var:
    :param special_value:
    :return:
    """
    x = var[~var.isna()]
    re = pd.DataFrame(index=[var.name],
                      columns=['Min', '1stQu', 'Median', 'Mean', '3rdQu', 'Max', 'Std', 'Mode', 'Na(%)',
                               'Singlemaxrate(%)'] + ['{}(%)'.format(ele) for ele in special_value])
    if types.is_numeric_dtype(x):
        x = var[~var.isin(special_value)]

        if len(x) == 0:
            re['Na(%)'] = [round(len(x[x.isna()]) * 100 / len(var), 4)]

        else:
            summary = x.describe().to_dict()
            re['Min'] = [round(summary['min'], 4)]
            re['1stQu'] = [round(summary['25%'], 4)]
            re['Median'] = [round(summary['50%'], 4)]
            re['Mean'] = [round(summary['mean'], 4)]
            re['3rdQu'] = [round(summary['75%'], 4)]
            re['Max'] = [round(summary['max'], 4)]
            re['Std'] = [round(summary['std'], 4)]
            re['Mode'] = [x.mode()[0]]
            re['Na(%)'] = [round(len(x[x.isna()]) * 100 / len(var), 4)]
            re['Singlemaxrate(%)'] = [round(x.value_counts().max() * 100 / len(var), 4)]

        if len(special_value) > 0:
            for ele in special_value:
                re['{}(%)'.format(ele)] = [round(len(var[var == ele]) * 100 / len(var), 4)]
    else:
        re['Mode'] = [x.mode()[0]]
        re['Na(%)'] = [round(len(var[var.isna()]) * 100 / len(var), 4)]
        re['Singlemaxrate(%)'] = [round(x.value_counts().max() * 100 / len(var), 4)]
        if len(special_value) > 0:
            for ele in special_value:
                re['{}(%)'.format(ele)] = [round(len(var[var == str(ele)]) * 100 / len(var), 4)]

    return re


def is_single_var(var, sv_perc):
    """
    单一值判断
    :param var: pandas.core.series.Series
    :param sv_perc: float, 单一值比率阈值
    :return boolean
    """
    if var.value_counts().max() >= sv_perc * len(var):
        return True
    else:
        return False


def single_var_name(data, sv_perc):
    """
    提取满足单一值条件的变量名
    :param data: pandas.core.frame.DataFrame
    :param sv_perc: 0.8
    :return [col1, col2]
    """
    columns = data.columns
    res = []
    for ele in columns:
        if is_single_var(data[ele], sv_perc):
            res.append(ele)
    return res


def f(var):
    """
    判断处理-1111和-9999之外的值的unique是否大于1
    :param var: pandas.core.series.Series
    :return boolean
    """
    if var[~var.isin([-1111, -9999])].nunique() <= 1:
        return False
    else:
        return True


def is_monotonous(var):
    """
    单调性判断（非严格）
    :param var: pandas.core.series.Series
    :return boolean
    """
    var = var[~var.isna()]
    if len(var) == 1:
        return True
    else:
        signd = np.sign(np.diff(var))
        if all(signd >= 0) or all(signd <= 0):
            return True
        else:
            return False


def is_quadratic(var):
    """
    单调性判断（严格）
    :param var: pandas.core.series.Series
    :return boolean
    """
    var = var[~var.isna()]
    if len(var) <= 2:
        return True
    else:
        signd = np.sign(np.diff(var))
        signd2 = np.sign(np.diff(signd))
        if all(signd2 >= 0) or all(signd2 <= 0):
            return True
        else:
            return False


def equfreq(var, num_bins=10):
    """
    等频分箱分位点
    :param var: pandas.core.series.Series
    :param num_bins: 箱数
    :return list
    """
    var = var[~var.isna()]
    var = sorted(var)
    bincount = round(len(var) / num_bins)
    points = var[0:1] + list(np.array(var)[bincount * np.arange(1, num_bins) - 1]) + var[len(var) - 1:]
    if points[0] == points[1]:
        points[0] -= 1
    return points


def equdist(var, num_bins=10):
    """
    等距分箱分位点
    :param var: pandas.core.series.Series
    :param num_bins: 箱数
    :return list
    """
    var = var[~var.isna()]
    var = sorted(var)
    points = np.linspace(min(var), max(var), num_bins + 1)
    return points


def get_one_best_ks(y, var, bad_value, good_value):
    """
    获取一个bestks点
    :param y: pandas.core.series.Series
    :param var: pandas.core.series.Series
    :param bad_value: 坏人的标志
    :param good_value: 好人的标志
    :return list
    """
    if sorted(y.unique()) != sorted([bad_value, good_value]):
        return 'y has other values beside bad_value and good_value'
    # 归并排序对大小相同的元素能够保持排序前的顺序，默认的快速排序则不能
    y = y[var.sort_values(kind='mergesort').index]
    var = var.sort_values()
    is_bad = y.apply(lambda x: 1 if x == bad_value else 0)
    is_good = 1 - is_bad
    bad_acc_num = np.cumsum(is_bad)
    good_acc_num = np.cumsum(is_good)
    bad_acc_per = bad_acc_num / sum(is_bad)
    good_acc_per = good_acc_num / sum(is_good)
    ks = abs(bad_acc_per - good_acc_per)
    return var[ks.idxmax()]


def get_var_best_ks(y, var, bad_value, good_value, bestks_k):
    """
    获取2^bestks_k - 1个bestks点,最终有2^bestks_k箱
    :param y: pandas.core.series.Series
    :param var: pandas.core.series.Series
    :param bad_value: 坏人的标志
    :param good_value: 好人的标志
    :param bestks_k: int
    :return list
    """
    y = y[~var.isna()]
    var = var[~var.isna()]
    bestks = get_one_best_ks(y=y, var=var, bad_value=bad_value, good_value=good_value)
    if bestks_k == 1:
        return bestks
    elif 1 < bestks_k <= len(y) - 1:
        for _ in range(1, bestks_k):
            if isinstance(bestks, list):
                bestks2 = [-np.Inf] + bestks + [np.Inf]
            else:
                bestks2 = [-np.Inf] + [bestks] + [np.Inf]
            len_bestks2 = len(bestks2)
            for j in range(len_bestks2 - 1):
                bestks2 += [get_one_best_ks(y=y[(var > bestks2[j]) & (var <= bestks2[j + 1])],
                                            var=var[(var > bestks2[j]) & (var <= bestks2[j + 1])],
                                            bad_value=bad_value,
                                            good_value=good_value)]

            bestks = sorted(bestks2)
            bestks = bestks[1:-1]
        return bestks
    elif bestks_k < 1:
        return 'bestks_k must be more than 0'
    else:
        return 'bestks_k must be less than number of sample - 1'


def quantile_for_bin(y, var, bad_value, good_value, break_type=1, breakpoints=None, special_value=None, num_bins=10,
                     bestks_k=3):
    """
    获取分位点
    :param y:
    :param var:
    :param bad_value:
    :param good_value:
    :param break_type: int, 1：等频分箱；2：等距分箱；3：bestks分箱
    :param breakpoints:
    :param special_value:
    :param num_bins:
    :param bestks_k:
    :return:
    """
    special_value = [] if special_value is None else special_value
    if breakpoints is not None:
        return breakpoints
    if len(y) != len(var):
        return 'The length of y must be equal to the length of var.'
    var2 = var[~var.isin(special_value)]
    y2 = y[~var.isin(special_value)]
    if break_type == 1:
        # 等频分位点
        breakpoints = equfreq(var2, num_bins=num_bins)
    elif break_type == 2:
        # 等距分位点
        breakpoints = equdist(var2, num_bins=num_bins)
    elif break_type == 3:
        # bestks分位点
        if bestks_k <= 0:
            return 'besks_k must be more than 0'

        breakpoints = get_var_best_ks(y=y2, bad_value=bad_value, good_value=good_value, var=var2, bestks_k=bestks_k)
        breakpoints = [min(var2)] + breakpoints + [max(var2)]

    breakpoints = sorted(list(set(breakpoints)))
    return breakpoints


# 指标计算函数类
def iv(a, b, replace_value):
    """
    计算iv
    :param a: int
    :param b: int
    :param replace_value: int
    :return float
    """
    if a * b == 0:
        a = replace_value if a == 0 else a
        b = replace_value if b == 0 else b
    return (a - b) * np.log(a / b)


def woe(a, b, replace_value):
    """
    计算woe
    :param a: int
    :param b: int
    :param replace_value: int
    :return: float
    """

    if a * b == 0:
        a = replace_value if a == 0 else a
        b = replace_value if b == 0 else b
    return np.log(a / b)


def psi(train_bin_count_per, test_bin_count_per):
    """
    计算psi
    :param train_bin_count_per:
    :param test_bin_count_per:
    :return:
    """
    train_bin_count_per = 0 if np.isnan(train_bin_count_per) else train_bin_count_per
    test_bin_count_per = 0 if np.isnan(test_bin_count_per) else test_bin_count_per
    if train_bin_count_per != test_bin_count_per and train_bin_count_per != 0 and test_bin_count_per != 0:
        train_diff_test_bin_count_per = train_bin_count_per - test_bin_count_per
        rate_train_diff_test_bin_count_per = train_bin_count_per / test_bin_count_per
        return train_diff_test_bin_count_per * np.log(rate_train_diff_test_bin_count_per)
    else:
        return 0


def binning(y, var, break_type=1, bad_value=1, good_value=0, breakpoints=None, special_value=None, bestks_k=3,
            num_bins=10, bin_rate_min=0, replace_value=1, closed_on_right=True):
    """
    通用单变量分箱
    #### 分箱函数类 ####
    # 分箱：
    # 入参：data为数据集（第一列为目标变量，其他为连续自变量）；
    #       var为自变量；num_bins为等分箱数；如果usequa = T，使用默认分位点；
    #       如果usequa = F，自行输入参数分位点向量fwd，fwd的第一个元素要比变量var的最小值小
    # 出参（list）：共四个元素：1、向量WOE：变量var的每个元素的woe；
    #                           2、变量var的分箱信息数据框index：num_bins行（箱）9列，列信息依次为：该箱的最小值、
    #                              最大值、好样本数、好样本占比、坏样本数、坏样本占比、样本数、woe、iv
    #                           3、变量var的IV
    #                           4、变量var的分位点quantile
    #                           5、变量var的num_binsS
    :param y:
    :param var:
    :param break_type:
    :param bad_value:
    :param good_value:
    :param breakpoints:
    :param special_value:
    :param bestks_k:
    :param num_bins:
    :param bin_rate_min:
    :param replace_value:
    :param closed_on_right:
    :return:
    """
    special_value = [] if special_value is None else special_value
    nsample = len(y)
    numallgood = len(y[y == good_value])
    numallbad = len(y[y == bad_value])
    breakpoints = quantile_for_bin(y=y, var=var, break_type=break_type, bad_value=bad_value, good_value=good_value,
                                   num_bins=num_bins, breakpoints=breakpoints, bestks_k=bestks_k,
                                   special_value=special_value)
    data_bin = deepcopy(var)
    tmpbin = pd.cut(data_bin[~data_bin.isin(special_value)], bins=breakpoints, precision=8, right=closed_on_right,
                    include_lowest=True)

    data_bin[~data_bin.isin(special_value)] = tmpbin
    data_bin[data_bin.isin(special_value)] = data_bin[data_bin.isin(special_value)].apply(
        lambda x: pd.Interval(x, x, closed='both'))

    index = tmpbin.cat.categories.tolist() + [pd.Interval(ele, ele, closed='both') for ele in special_value]

    card = pd.DataFrame(index=index)
    tmp_card = pd.merge(data_bin, y, left_index=True, right_index=True)
    var_name = data_bin.name
    label_name = y.name
    card.loc[:, 'good.count'] = tmp_card[tmp_card[label_name] == good_value].groupby(var_name)[label_name].count()
    card.loc[:, 'bad.count'] = tmp_card[tmp_card[label_name] == bad_value].groupby(var_name)[label_name].count()
    card.fillna(0, inplace=True)

    # 保证特殊值分箱永远在最前面
    part1 = card[card.index.isin([pd.Interval(ele, ele, closed='both') for ele in special_value])]
    part2 = card[~card.index.isin([pd.Interval(ele, ele, closed='both') for ele in special_value])]
    card = pd.concat([part1.sort_index(), part2.sort_index()])

    card['count'] = card['good.count'] + card['bad.count']
    card['binning_type'] = 'range'
    card.loc[card.index[0:len(special_value)], 'binning_type'] = 'single_value'
    card = card[[card.columns[-1]] + list(card.columns[0:-1])]
    card['good.perc'] = card['good.count'] / card['count']
    card['bad.perc'] = card['bad.count'] / card['count']
    card['count.perc'] = card['count'] / nsample
    card['count.perc.acc'] = np.cumsum(card['count.perc'])

    card['good.perc.all'] = card['good.count'] / nsample
    card['bad.perc.all'] = card['bad.count'] / nsample
    card['good.perc.allgood'] = card['good.count'] / numallgood
    card['bad.perc.allbad'] = card['bad.count'] / numallbad
    card['good.perc.allgood.acc'] = np.cumsum(card['good.perc.allgood'])
    card['bad.perc.allbad.acc'] = np.cumsum(card['bad.perc.allbad'])
    # woe计算坏好比
    card['woe'] = card.apply(lambda x: woe(x['bad.perc.allbad'], x['good.perc.allgood'], replace_value), axis=1)
    card[card['count.perc'] < bin_rate_min]['woe'] = 0
    card['iv'] = card.apply(lambda x: iv(x['bad.perc.allbad'], x['good.perc.allgood'], replace_value), axis=1)
    card['ks'] = abs(card['bad.perc.allbad.acc'] - card['good.perc.allgood.acc'])
    card['label'] = ['bin{}'.format(i) for i in range(1, len(card) + 1)]
    card = card[[card.columns[-1]] + list(card.columns[0:-1])]
    card.fillna(0, inplace=True)
    vardetail = dict()
    sampleinfo = deepcopy(var)
    sampleinfo = sampleinfo.to_frame(name='var')
    sampleinfo['bin'] = data_bin
    sampleinfo['woe'] = list(card.loc[data_bin, 'woe'])
    vardetail['sampleinfo'] = sampleinfo
    vardetail['card'] = card
    vardetail['breakpoints'] = breakpoints
    varindex = pd.DataFrame({'IV': [card['iv'].sum()], 'num_binsS': [card['ks'].max()], 'numbin': [len(card)]})
    varindex.index = [var.name]
    vardetail['varindex'] = varindex
    return vardetail


def onecombine(y, var, varindex, special_value=None, break_type=1, bin_rate_min=0, bad_value=1, good_value=0,
               num_bins=10, replace_value=1, comb_type='combinning', closed_on_right=True):
    """
    合箱
    :param y:
    :param var:
    :param varindex:
    :param special_value:
    :param break_type:
    :param bin_rate_min:
    :param bad_value:
    :param good_value:
    :param num_bins:
    :param replace_value:
    :param comb_type: str, 'combinning' or 'decreasing'
    :param closed_on_right:
    :return:
    """
    special_value = [] if special_value is None else special_value
    if comb_type == 'combinning':
        binwoe = deepcopy(varindex['card']['woe'])
        binwoe = binwoe.iloc[len(special_value):]
        diffbinwoe = abs(np.diff(binwoe))
        breakpoints = deepcopy(varindex['breakpoints'])
        del breakpoints[np.argmin(diffbinwoe) + 1]
        varindex = binning(y=y, var=var, break_type=break_type, bad_value=bad_value, good_value=good_value,
                           breakpoints=breakpoints, special_value=special_value, bestks_k=0, num_bins=num_bins,
                           bin_rate_min=bin_rate_min, replace_value=replace_value, closed_on_right=closed_on_right)

    elif comb_type == 'decreasing':
        num_bins = len(varindex['breakpoints']) - 2
        varindex = binning(y=y, var=var, break_type=break_type, bad_value=bad_value, good_value=good_value,
                           breakpoints=None, special_value=special_value, bestks_k=0, num_bins=num_bins,
                           bin_rate_min=bin_rate_min, replace_value=replace_value, closed_on_right=closed_on_right)

    else:
        return 'comb_type has no {}'.format(comb_type)

    return varindex


def finalvarindex(y, var, bad_value=1, good_value=0, num_bins=10, min_num_bins=3, max_num_bins=10, breakpoints=None,
                  bestks_k=0, special_value=None, break_type=1, bin_rate_min=0, replace_value=1, comb_type='combinning',
                  woe_stand='monotonous', closed_on_right=True):
    """
    训练集单变量分箱
    :param y:
    :param var:
    :param bad_value:
    :param good_value:
    :param num_bins:
    :param min_num_bins:
    :param max_num_bins:
    :param breakpoints:
    :param bestks_k:
    :param special_value:
    :param break_type:
    :param bin_rate_min:
    :param replace_value:
    :param comb_type:
    :param woe_stand: str, monotonous or quadratic
    :param closed_on_right:
    :return:
    """
    special_value = [] if special_value is None else special_value
    varindex = binning(y=y, var=var, break_type=break_type, bad_value=bad_value, good_value=good_value,
                       breakpoints=breakpoints, special_value=special_value, bestks_k=bestks_k, num_bins=num_bins,
                       bin_rate_min=bin_rate_min, replace_value=replace_value, closed_on_right=closed_on_right)
    binwoe = deepcopy(varindex['card']['woe'])
    binwoe = binwoe.iloc[len(special_value):]

    if breakpoints is not None:
        return varindex

    elif woe_stand == 'monotonous':
        while (len(binwoe) > min_num_bins and (not is_monotonous(binwoe))) or len(binwoe) > max_num_bins:
            varindex = onecombine(y, var, varindex, special_value=special_value, break_type=break_type,
                                  bin_rate_min=bin_rate_min, bad_value=bad_value, good_value=good_value,
                                  num_bins=num_bins, replace_value=replace_value, comb_type=comb_type,
                                  closed_on_right=closed_on_right)
            binwoe = deepcopy(varindex['card']['woe'])
            binwoe = binwoe.iloc[len(special_value):]

    elif woe_stand == 'quadratic':
        while (len(binwoe) > min_num_bins and (not is_quadratic(binwoe))) or len(binwoe) > max_num_bins:
            varindex = onecombine(y, var, varindex, special_value=special_value, break_type=break_type,
                                  bin_rate_min=bin_rate_min, bad_value=bad_value, good_value=good_value,
                                  num_bins=num_bins, replace_value=replace_value, comb_type=comb_type,
                                  closed_on_right=closed_on_right)
            binwoe = deepcopy(varindex['card']['woe'])
            binwoe = binwoe.iloc[len(special_value):]

    else:
        return 'woe_stand is quadratic or monotonous'
    return varindex


def allvarindex(datax, y, bad_value=1, good_value=0, num_bins=10, min_num_bins=3, max_num_bins=10, bestks_k=0,
                special_value=None, bin_rate_min=0, replace_value=1, break_type=1, comb_type='combinning',
                woe_stand='monotonous', closed_on_right=True, manuel_breakpoints_dict=None):
    """
    训练集全部变量分箱
    :param datax:
    :param y:
    :param bad_value:
    :param good_value:
    :param num_bins:
    :param min_num_bins:
    :param max_num_bins:
    :param bestks_k:
    :param special_value:
    :param bin_rate_min:
    :param replace_value:
    :param break_type:
    :param comb_type:
    :param woe_stand:
    :param closed_on_right:
    :param manuel_breakpoints_dict
    :return:
    """
    special_value = [] if special_value is None else special_value
    manuel_breakpoints_dict = dict() if manuel_breakpoints_dict is None else manuel_breakpoints_dict
    var_index = dict()
    for ele in datax.columns:
        var = datax[ele]
        breakpoints = manuel_breakpoints_dict.get(ele)
        var_index[ele] = finalvarindex(y=y, var=var, bad_value=bad_value, good_value=good_value, num_bins=num_bins,
                                       min_num_bins=min_num_bins, max_num_bins=max_num_bins, breakpoints=breakpoints,
                                       bestks_k=bestks_k, special_value=special_value, break_type=break_type,
                                       bin_rate_min=bin_rate_min, replace_value=replace_value, comb_type=comb_type,
                                       woe_stand=woe_stand, closed_on_right=closed_on_right)

    return var_index


def binningfortest(y, testinfo, special_value=None, bin_rate_min=0, break_type=1, bad_value=1, good_value=0,
                   replace_value=1, closed_on_right=True):
    """
    测试集单变量分箱
    :param y:
    :param testinfo:
    :param special_value:
    :param bin_rate_min:
    :param break_type:
    :param bad_value:
    :param good_value:
    :param replace_value:
    :param closed_on_right:
    :return:
    """
    special_value = [] if special_value is None else special_value
    var = deepcopy(testinfo['var'])
    breakpoints = deepcopy(testinfo['breakpoints'])
    # 切分点最大/小值（训练集最大/小值）修改为测试集最大/小值;注：特殊值除外
    breakpoints[-1] = np.Inf
    breakpoints[0] = -np.Inf
    varindex = binning(y=y, var=var, break_type=break_type, bad_value=bad_value, good_value=good_value,
                       breakpoints=breakpoints, special_value=special_value, bin_rate_min=bin_rate_min,
                       replace_value=replace_value, closed_on_right=closed_on_right)

    return varindex


def allvarindexfortest(datax, y_test, trainbreakpoints, closed_on_right, break_type, bad_value=1, good_value=0,
                       special_value=None, bin_rate_min=0, replace_value=1):
    """
    测试集全部变量分箱
    :param datax:
    :param y_test:
    :param trainbreakpoints:
    :param closed_on_right:
    :param break_type:
    :param bad_value:
    :param good_value:
    :param special_value:
    :param bin_rate_min:
    :param replace_value:
    :return:
    """
    special_value = [] if special_value is None else special_value
    var_index = dict()
    for ele in datax.columns:
        testinfo = dict()
        testinfo['var'] = datax[ele]
        testinfo['breakpoints'] = trainbreakpoints[ele]

        var_index[ele] = binningfortest(y_test, testinfo, special_value=special_value, bin_rate_min=bin_rate_min,
                                        break_type=break_type, bad_value=bad_value, good_value=good_value,
                                        replace_value=replace_value, closed_on_right=closed_on_right)

    return var_index


def filter_var(train_woematrix, corr_t):
    k = 0
    while k + 1 < train_woematrix.shape[1]:
        corr = train_woematrix.iloc[:, k:].corr(method='spearman').iloc[0][1:]
        dename = corr[abs(corr) > corr_t].index.tolist()
        name = [ele for ele in train_woematrix if ele not in dename]

        if len(dename) > 0:
            train_woematrix = train_woematrix[name]
        else:
            k += 1

    return train_woematrix


def write_df_dict_to_sheet(df_dict, writer, sheet_name, line_spacing=5):
    """
    :param df_dict:
    :param writer:
    :param sheet_name:
    :param line_spacing: 行间距
    :return:
    """
    # Get the xlsxwriter workbook and worksheet objects
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    # Add a format
    df_format = workbook.add_format({'bg_color': 'green',
                                     'font_color': 'white',
                                     'border': 1,
                                     'bold': True})

    name_format = workbook.add_format({'bg_color': 'gray',
                                       'font_color': 'white',
                                       'border': 1,
                                       'bold': True})

    startrow = 0
    for k, v in df_dict.items():
        worksheet.write(startrow, 0, k, name_format)
        v.to_excel(writer, sheet_name=sheet_name, startrow=startrow + 1, startcol=0)
        for col_num, value in enumerate(v.columns.values):
            worksheet.write(startrow + 1, col_num + 1, value, df_format)

        for row_num, value in enumerate(v.index):
            worksheet.write(row_num + startrow + 2, 0, str(value), df_format)
        worksheet.write(startrow + 1, 0, None, df_format)
        startrow += len(v) + 2 + line_spacing


def write_dict_to_excel(path, info):
    """
    :param path:
    :param info:
    :return:
    """
    if max([len(ele) for ele in info.keys()]) > 31:
        transname = ['name{}'.format(ele) for ele in range(len(info))]
        info['name'] = pd.DataFrame({'oriname': list(info.keys()), 'transname': transname})

    writer = pd.ExcelWriter(path, engine='xlsxwriter')

    for k, v in info.items():
        if isinstance(v, pd.DataFrame):
            df_dict = {k: v}
        else:
            df_dict = v
        write_df_dict_to_sheet(df_dict=df_dict, writer=writer, sheet_name=k)
    writer.save()


def insert_image_to_excel(path, sheet_name, image_path, anchor):
    """
    :param path:
    :param sheet_name:
    :param image_path:
    :param anchor:
    :return:
    """
    workbook = load_workbook(path)
    ws = workbook[sheet_name]
    image = drawing.image.Image(image_path)
    image.width /= 5
    image.height /= 5
    ws.add_image(image, anchor)
    workbook.save(path)
