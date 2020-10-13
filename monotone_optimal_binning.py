# inspired by git project https://github.com/jstephenj14/Monotonic-WOE-Binning-Algorithm
# and thesis "Monotone optimal binning algorithm for credit risk modeling"


# #################算法原理##################
# 1.按变量值对数据进行排序，将样本划分到最细的bin，比如每个样本都有自己的bin。
# 2.第一次合箱（保证单调性）：至上而下用两个指针（i，j）控制是否要合箱；
#   2.1如果j对应的bad_rate比i对应的bad_rate低，则i向下走；
#   2.2否则，则说明需要合并i和j这两箱，合并生成新的箱作为i，j往下走一个，
#      判断此时的j对应的bad_rate比i对应的bad_rate低还是高，如果高则继续2.2，否则go to 2.1
#   2.3直到i或j走完结束
# 3.第二次合箱（保证两箱的bad_rate有显著差异，显著水平（置信程度）通过p_value控制）
#   3.1如果单箱样本数小于给定值
#   3.2如果单项坏样本数小于给定值
#   3.3如果任意两箱的bad_rate的p_value大于给定值
#   3.4上述三个条件满足其一则进行合箱操作
# #################算法原理##################

import os
import pandas as pd
import scipy.stats as stats
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 130)
warnings.filterwarnings("ignore")
os.getcwd()


class Binning(BaseEstimator, TransformerMixin):

    def __init__(self, y, n_threshold, y_threshold=1, p_threshold=1, sign=False, closed='right'):
        """
        # 如果不管怎么分箱都是不单调的，那么就会只分一箱
        :param y: 标签列
        :param n_threshold: 每一箱至少有多少个样本，比如要求单箱占比至少5%，则取样本数*0.05
        :param y_threshold: 是每一箱至少有多少个坏样本，至少设置成1，否则没办法计算woe
        :param p_threshold: 是p_value为多少的时候认为是相关的，介于0到1之间，值越大，条件越宽松，分的箱数越多，其实就是把可能分布相近的箱保持不变
        :param sign: False表示变量值越大，样本越坏，比如额度使用率越大，样本越坏
        """
        self.n_threshold = n_threshold
        self.y_threshold = y_threshold
        self.p_threshold = p_threshold
        self.y = y
        self.sign = sign

        self.init_summary = pd.DataFrame()
        self.bin_summary = pd.DataFrame()
        self.pvalue_summary = pd.DataFrame()
        self.dataset = pd.DataFrame()
        self.woe_summary = pd.DataFrame()

        self.column = None
        self.total_iv = None
        self.bins = None
        self.closed = closed
        self.bucket = True if closed == 'right' else False

    def generate_summary(self):

        self.init_summary = self.dataset.groupby([self.column]).agg({self.y: ["mean", "std", "size"]}).rename(
            {"mean": "means", "size": "nsamples", "std": "std_dev"}, axis=1)

        self.init_summary.columns = self.init_summary.columns.droplevel(level=0)

        self.init_summary = self.init_summary[["means", "nsamples", "std_dev"]]
        self.init_summary = self.init_summary.reset_index()

        self.init_summary["del_flag"] = 0
        self.init_summary["std_dev"] = self.init_summary["std_dev"].fillna(0)

        self.init_summary = self.init_summary.sort_values([self.column], ascending=not self.bucket)

    def combine_bins(self):
        summary = self.init_summary.copy()

        while True:
            i = 0
            summary = summary[summary.del_flag != 1]
            summary = summary.reset_index(drop=True)
            if (self.bucket and self.sign) or (not self.bucket and not self.sign):
                while True:

                    j = i + 1

                    if j >= len(summary):
                        break

                    if summary.iloc[j].means > summary.iloc[i].means:
                        i = i + 1
                        continue
                    else:
                        while True:
                            n = summary.iloc[j].nsamples + summary.iloc[i].nsamples
                            m = (summary.iloc[j].nsamples * summary.iloc[j].means +
                                 summary.iloc[i].nsamples * summary.iloc[i].means) / n

                            if n == 2:
                                s = np.std([summary.iloc[j].means, summary.iloc[i].means])
                            else:
                                # 参考https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
                                s = np.sqrt(((summary.iloc[j].nsamples - 1) * (summary.iloc[j].std_dev ** 2) +
                                             (summary.iloc[i].nsamples - 1) * (summary.iloc[i].std_dev ** 2) +
                                             summary.iloc[j].nsamples * (m - summary.iloc[j].means) ** 2 +
                                             summary.iloc[i].nsamples * (m - summary.iloc[i].means) ** 2) / (n - 1))

                            summary.loc[i, "nsamples"] = n
                            summary.loc[i, "means"] = m
                            summary.loc[i, "std_dev"] = s
                            summary.loc[j, "del_flag"] = 1

                            j = j + 1
                            if j >= len(summary):
                                break
                            if summary.loc[j, "means"] > summary.loc[i, "means"]:
                                i = j
                                break
                    if j >= len(summary):
                        break
                dels = np.sum(summary["del_flag"])
                if dels == 0:
                    break
            else:
                while True:

                    j = i + 1

                    if j >= len(summary):
                        break

                    if summary.iloc[j].means < summary.iloc[i].means:
                        i = i + 1
                        continue
                    else:
                        while True:
                            n = summary.iloc[j].nsamples + summary.iloc[i].nsamples
                            m = (summary.iloc[j].nsamples * summary.iloc[j].means +
                                 summary.iloc[i].nsamples * summary.iloc[i].means) / n

                            if n == 2:
                                s = np.std([summary.iloc[j].means, summary.iloc[i].means])
                            else:
                                # 参考https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
                                s = np.sqrt(((summary.iloc[j].nsamples - 1) * (summary.iloc[j].std_dev ** 2) +
                                             (summary.iloc[i].nsamples - 1) * (summary.iloc[i].std_dev ** 2) +
                                             summary.iloc[j].nsamples * (m - summary.iloc[j].means) ** 2 +
                                             summary.iloc[i].nsamples * (m - summary.iloc[i].means) ** 2) / (n - 1))

                            summary.loc[i, "nsamples"] = n
                            summary.loc[i, "means"] = m
                            summary.loc[i, "std_dev"] = s
                            summary.loc[j, "del_flag"] = 1

                            j = j + 1
                            if j >= len(summary):
                                break
                            if summary.loc[j, "means"] < summary.loc[i, "means"]:
                                i = j
                                break
                    if j >= len(summary):
                        break
                dels = np.sum(summary["del_flag"])
                if dels == 0:
                    break

        self.bin_summary = summary.copy()

    def calculate_pvalues(self):
        summary = self.bin_summary.copy()
        while True:
            summary["means_lead"] = summary["means"].shift(-1)
            summary["nsamples_lead"] = summary["nsamples"].shift(-1)
            summary["std_dev_lead"] = summary["std_dev"].shift(-1)

            summary["est_nsamples"] = summary["nsamples_lead"] + summary["nsamples"]
            summary["est_means"] = (summary["means_lead"] * summary["nsamples_lead"] +
                                    summary["means"] * summary["nsamples"]) / summary["est_nsamples"]

            summary["est_std_dev2"] = ((summary["nsamples_lead"] - 1) * summary["std_dev_lead"] ** 2 +
                                       (summary["nsamples"] - 1) * summary["std_dev"] ** 2) / (
                                              summary["est_nsamples"] - 2)

            # 其实这里写错了，应该是单边的t检验，用来检验两个分布的均值有多大概率是相同的
            # 参考https://mathcracker.com/t-test-for-two-means
            summary["z_value"] = (summary["means"] - summary["means_lead"]) / np.sqrt(
                summary["est_std_dev2"] * (1 / summary["nsamples"] + 1 / summary["nsamples_lead"]))

            summary["p_value"] = stats.t.sf(summary["z_value"], summary["est_nsamples"] - 2)
            # 1.如果单箱样本数小于给定值
            # 2.如果单项坏样本数小于给定值
            # 对于上述两种情况均需要合箱操作
            summary["p_value"] = summary.apply(
                lambda row: row["p_value"] + 1 if (row["nsamples"] < self.n_threshold) |
                                                  (row["nsamples_lead"] < self.n_threshold) |
                                                  (row["means"] * row["nsamples"] < self.y_threshold) |
                                                  (row["means_lead"] * row["nsamples_lead"] < self.y_threshold)
                else row["p_value"], axis=1)

            max_p = max(summary["p_value"])
            row_of_maxp = summary['p_value'].idxmax()
            row_delete = row_of_maxp + 1

            if max_p > self.p_threshold:
                summary = summary.drop(summary.index[row_delete])
                summary = summary.reset_index(drop=True)
            else:
                break

            summary["means"] = summary.apply(lambda row: row["est_means"] if row["p_value"] == max_p else row["means"],
                                             axis=1)
            summary["nsamples"] = summary.apply(
                lambda row: row["est_nsamples"] if row["p_value"] == max_p else row["nsamples"], axis=1)
            summary["std_dev"] = summary.apply(
                lambda row: np.sqrt(row["est_std_dev2"]) if row["p_value"] == max_p else row["std_dev"], axis=1)

        self.pvalue_summary = summary.copy()

    def calculate_woe(self):
        woe_summary = self.pvalue_summary[[self.column, "nsamples", "means"]]

        woe_summary["bads"] = woe_summary["means"] * woe_summary["nsamples"]
        woe_summary["goods"] = woe_summary["nsamples"] - woe_summary["bads"]

        total_goods = np.sum(woe_summary["goods"])
        total_bads = np.sum(woe_summary["bads"])

        woe_summary["dist_good"] = woe_summary["goods"] / total_goods
        woe_summary["dist_bad"] = woe_summary["bads"] / total_bads

        woe_summary["WOE_" + self.column] = np.log(woe_summary["dist_bad"] / woe_summary["dist_good"])

        woe_summary["IV_components"] = (woe_summary["dist_bad"] - woe_summary["dist_good"]) * woe_summary[
            "WOE_" + self.column]

        self.total_iv = np.sum(woe_summary["IV_components"])
        self.woe_summary = woe_summary

    def generate_bin_labels(self, row):
        left, right = np.sort([row[self.column], row[self.column + "_shift"]])
        return pd.Interval(left, right, closed=self.closed)

    def generate_final_dataset(self):
        shift_var = -1
        self.woe_summary[self.column + "_shift"] = self.woe_summary[self.column].shift(shift_var)

        if (self.sign and self.bucket) or (not self.sign and self.bucket):
            self.woe_summary.loc[len(self.woe_summary) - 1, self.column + "_shift"] = -np.inf

        else:
            self.woe_summary.loc[len(self.woe_summary) - 1, self.column + "_shift"] = np.inf

        self.bins = np.sort(list(self.woe_summary[self.column]) + [np.Inf, -np.Inf])

        self.woe_summary["labels"] = self.woe_summary.apply(self.generate_bin_labels, axis=1)

        self.dataset["bins"] = pd.cut(self.dataset[self.column], self.bins, right=self.bucket, precision=0)

        self.dataset["bins"] = self.dataset["bins"].astype(str)
        self.dataset['bins'] = self.dataset['bins'].map(lambda x: x.lstrip('[').rstrip(')'))

    def fit(self, dataset):
        self.dataset = dataset
        self.column = self.dataset.columns[self.dataset.columns != self.y][0]

        self.generate_summary()
        self.combine_bins()
        self.calculate_pvalues()
        self.calculate_woe()
        self.generate_final_dataset()

    def transform(self, test_data):
        test_data[self.column + "_bins"] = pd.cut(test_data[self.column], self.bins, right=self.bucket, precision=0)
        return test_data


if __name__ == '__main__':
    # Data available at https://online.stat.psu.edu/stat508/resource/analysis/gcd
    train = pd.read_csv("all_data//Training50.csv")
    test = pd.read_csv("all_data//Test50.csv")

    var = "Age..years."  # variable to be binned
    y_var = "Creditability"  # the target variable

    bin_object = Binning(y_var, n_threshold=train.shape[0] * 0.05, y_threshold=1, p_threshold=1, sign=False,
                         closed='right')
    bin_object.fit(train[[y_var, var]])

    # Print WOE summary
    print(bin_object.woe_summary)

    # The bin cut-points in an array
    print(bin_object.bins)

    test_transformed = bin_object.transform(test)
