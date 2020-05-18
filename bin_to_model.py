import statsmodels.formula.api as smf
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from common_util import *


def logisticscreen(y, datax, p_min, weights):
    """
    :param y:
    :param datax:
    :param p_min:
    :param weights:
    :return:
    """
    print('Start running logistic regression')
    data = pd.concat([y, datax], axis=1)
    glm_fit = smf.logit(formula='{}~{}'.format(y.name, '+'.join(datax.columns[0:-1])), data=data,
                        weights=np.array(weights)).fit()
    coeff = glm_fit.params.iloc[1:, ]
    p = glm_fit.pvalues.iloc[1:, ]
    name = [ele for ele in coeff[coeff > 0].index if ele in p[p <= p_min].index]

    # 逐步回归
    print('Start running stepwise regression')
    print('Screening based on P-value: {}'.format(p_min))
    while (len(coeff[coeff < 0]) > 0 or max(p) > p_min) and datax.shape[1] > 1:
        datax = datax[name]
        glm_fit = smf.logit(formula='{}~{}'.format(y.name, '+'.join(datax.columns)), data=data,
                            weights=np.array(weights)).fit()
        coeff = glm_fit.params.iloc[1:, ]
        p = glm_fit.pvalues.iloc[1:, ]
        name = [ele for ele in coeff[coeff > 0].index if ele in p[p <= p_min].index]
    if (coeff.iloc[0] < 0 or p.iloc[0] > p_min) and datax.shape == 1:
        return print('No residual x-variables after screening.')

    print('After the P-value {} and coefficients screening, remaining {} variables'.format(p_min, datax.shape[1]))

    r = dict()
    r['name'] = name
    r['coeff'] = coeff
    r['model'] = glm_fit
    r['coeff0'] = glm_fit.params.loc['Intercept']
    return r


def standcard(y, train_card, pdo, p0, good_value, bad_value, coeff, coeff0, theta=None):
    """
    :param y:
    :param train_card:
    :param pdo:
    :param p0:
    :param good_value:
    :param bad_value:
    :param coeff:
    :param coeff0:
    :param theta:
    :return:
    """
    if theta is None:
        theta = len(y[y == bad_value]) / len(y[y == good_value])

    b = pdo / np.log(2)
    a = p0 + b * np.log(theta)
    train_card['varname'] = train_card.index.map(lambda x: x.split('.')[0])
    coeff = coeff.to_frame('coeff')
    coeff['varname'] = coeff.index
    train_card = pd.merge(train_card, coeff, on='varname', how='left')
    train_card['score'] = round((a - b * coeff0) / len(coeff) - b * train_card['woe'] * train_card['coeff'])
    train_card = train_card[['varname', 'binning_type', 'label', 'range', 'woe', 'score', 'count.perc']]
    return train_card


def get_one_score(x, one_card):
    """
    :param x:
    :param one_card:
    :return:
    """
    for i in range(len(one_card)):
        if x in one_card['range'].iloc[i]:
            return one_card['score'].iloc[i]


def vartoscore(one_var, one_card):
    """
    :param one_var:
    :param one_card:
    :return:
    """
    one_score = deepcopy(one_var)
    one_score = one_score.apply(lambda x: get_one_score(x, one_card))
    one_score = one_score.to_frame(name=one_var.name)
    return one_score


def dtos(datax, card):
    """
    :param datax:
    :param card:
    :return:
    """
    varname = list(card['varname'].unique())
    score_list = []
    for ele in varname:
        one_var = datax[ele]
        one_card = card[card['varname'] == ele]
        one_score = vartoscore(one_var, one_card)
        score_list.append(one_score)
    scorematrix = pd.concat(score_list, axis=1)
    return scorematrix


# def perfm(target, prediction, F_beta=1):
#     """
#     performance measure
#     :param target:
#     :param prediction:
#     :param F_beta:
#     :return:
#     """
#     if len(target) != len(prediction):
#         return 'the length of target and prediction must be equal.'
#
#     if len(target[target == 0]) == 0 or len(target[target == 1]) == 0:
#         return 'target has no 0 or 1.'
#
#     if len(target[(target != 0) & (target != 1)]) > 0:
#         return 'target has other classify beside 0 and 1.'
#
#     if len(prediction[(prediction < 0) | (prediction > 1)]) > 0:
#         return 'prediction must be probabilities.'
#
#     n = len(target)
#     n0 = len(target[target == 0])
#     n1 = len(target[target == 1])
#     pm = dict()
#     pm['original.target'] = target
#     pm['original.prediction'] = prediction
#     pm['prediction'] = prediction.sort_values(kind='mergesort')
#     pm['target'] = target[prediction.sort_values(kind='mergesort').index]
#     pm['TP'] = list(range(1, n + 1)) - np.cumsum(pm['target'])
#     pm['FP'] = np.cumsum(pm['target'])
#     pm['FN'] = n0 - pm['TP']
#     pm['TN'] = n - pm['TP'] - pm['FP'] - pm['FN']
#     pm['TPR'] = pm['TP'] / n0  # True positive rate
#     pm['FPR'] = pm['FP'] / n1  # False positive rate
#     pm['precision'] = pm['TP'] / list(range(1, n + 1))
#     pm['recall'] = pm['TP'] / n0
#     pm['Fscore'] = (F_beta ** 2 * pm['precision'] + pm['recall']) / (1 + F_beta ** 2) * pm['precision'] * pm['recall']
#     pm['auc'] = 0.5 * sum(np.diff(pm['FPR']) * (pm['TPR'][0: (n - 1)].values + pm['TPR'][1:n].values))
#     pm['micro_P'] = np.mean(pm['TP']) / (np.mean(pm['TP']) + np.mean(pm['FP']))
#     pm['micro_R'] = np.mean(pm['TP']) / (np.mean(pm['TP']) + np.mean(pm['FN']))
#     pm['micro_Fscore'] = (1 + F_beta ** 2) * pm['micro_P'] * pm['micro_R'] / (
#             F_beta ** 2 * pm['micro_P'] + pm['micro_R'])
#     pm['F_beta'] = F_beta
#     return pm


def perfm(y_true, y_score):
    """
    performance measure
    :param y_true:
    :param y_score:
    :return:
    """
    if len(y_true) != len(y_score):
        return 'the length of y_true and y_score must be equal.'

    if len(y_true[y_true == 0]) == 0 or len(y_true[y_true == 1]) == 0:
        return 'y_true has no 0 or 1.'

    if len(y_true[(y_true != 0) & (y_true != 1)]) > 0:
        return 'y_true has other classify beside 0 and 1.'

    if len(y_score[(y_score < 0) | (y_score > 1)]) > 0:
        return 'y_score must be probabilities.'

    pm = dict()
    pm['FPR'], pm['TPR'], _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pm['precision'], pm['recall'] = precision[::-1], recall[::-1]
    pm['auc'] = roc_auc_score(y_true, y_score)
    return pm


def plot_roc(pm, title='AUC-plot', image_path=None):
    fig, ax = plt.subplots()
    plt.plot(pm['FPR'], pm['TPR'], color='black', linewidth=0.5)
    plt.xlabel('FPR(1 - Sensitivity)')
    plt.ylabel('TPR(Specificity)')
    plt.xlim = [0, 1]
    plt.ylim = [0, 1]
    plt.title(title)

    verts = np.array([[-0.1, -0.1],
                      [-0.1, 1.1],
                      [1.1, 1.1],
                      [1.1, -0.1]])
    polygon = Polygon(verts, color='#F5DEB3')
    ax.add_patch(polygon)

    verts = np.array([[0, 0],
                      [0, 1],
                      [1, 1],
                      [1, 0]])
    polygon = Polygon(verts, color='grey')
    ax.add_patch(polygon)

    fpr_array = np.array(list(pm['FPR']) + [1])
    tpr_array = np.array(list(pm['TPR']) + [0])
    verts = np.concatenate((fpr_array.reshape((len(fpr_array), -1)), tpr_array.reshape((len(tpr_array), -1))), axis=1)

    polygon = Polygon(verts, facecolor='lightblue', edgecolor='black')
    ax.add_patch(polygon)

    ax.plot([0, 1], [1, 0], color='grey', linewidth=0.5)
    plt.text(pm['FPR'][abs(pm['TPR'] + pm['FPR'] - 1).argmin()] + 0.15,
             pm['TPR'][abs(pm['TPR'] + pm['FPR'] - 1).argmin()] - 0.15,
             '({},{})'.format(round(pm['FPR'][abs(pm['TPR'] + pm['FPR'] - 1).argmin()], 4),
                              round(pm['TPR'][abs(pm['TPR'] + pm['FPR'] - 1).argmin()], 4)), color='blue')
    plt.text(0.7, 0.2, 'AUC:{}'.format(round(pm['auc'], 4)))
    if image_path:
        plt.savefig(image_path, dpi=500)
    plt.close()


def plot_pr(pm, title="PR-plot", image_path=None):
    fig, ax = plt.subplots()
    plt.plot(pm['recall'], pm['precision'], color='black', linewidth=0.5)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim = [0, 1]
    plt.ylim = [0, 1]
    plt.title(title)

    verts = np.array([[-0.1, -0.1],
                      [-0.1, 1.1],
                      [1.1, 1.1],
                      [1.1, -0.1]])
    polygon = Polygon(verts, color='#F5DEB3')
    ax.add_patch(polygon)

    verts = np.array([[0, 0],
                      [0, 1],
                      [1, 1],
                      [1, 0]])
    polygon = Polygon(verts, color='grey')
    ax.add_patch(polygon)

    recall_array = np.array([0] + list(pm['recall']) + [1])
    precision_array = np.array([0] + list(pm['precision']) + [0])
    verts = np.concatenate(
        (recall_array.reshape((len(recall_array), -1)), precision_array.reshape((len(precision_array), -1))), axis=1)

    polygon = Polygon(verts, facecolor='lightblue', edgecolor='black')
    ax.add_patch(polygon)
    ax.plot([0, 1], [0, 1], color='grey', linewidth=0.5)
    plt.text(pm['recall'][abs(pm['recall'] - pm['precision']).argmin()] - 0.3,
             pm['precision'][abs(pm['recall'] - pm['precision']).argmin()] - 0.15,
             '({},{})'.format(round(pm['recall'][abs(pm['recall'] - pm['precision']).argmin()], 4),
                              round(pm['precision'][abs(pm['recall'] - pm['precision']).argmin()], 4)),
             color='blue')
    if image_path:
        plt.savefig(image_path, dpi=500)
    plt.close()


def interval_to_str(score_info):
    r = []
    for i, v in enumerate(score_info.index.to_list()):
        if v.left == -np.inf:
            left = '-inf'
            right = math.ceil(v.right)
        elif v.right == np.inf:
            left = math.ceil(v.left)
            right = 'inf'
        else:
            left = math.ceil(v.left)
            right = math.ceil(v.right)
        if i == 0:
            r.append('[{},{}]'.format(left, right))
        else:
            r.append('({},{}]'.format(left, right))
    return r


def plot_ks(score_info, goodcol='blue', badcol='darkgreen', kscol='red', title='KS-plot', image_path=None):
    fig, ax = plt.subplots()
    good_plot, = plt.plot(list(range(len(score_info) + 1)), [0] + score_info['good.perc.allgood.acc'].tolist(),
                          color=goodcol, linewidth=0.5, marker='.')
    bad_plot, = plt.plot(list(range(len(score_info) + 1)), [0] + score_info['bad.perc.allbad.acc'].tolist(),
                         color=badcol, linewidth=0.5, marker='.')
    plt.ylabel('acc_perc')
    plt.xlim = [0, len(score_info)]
    plt.ylim = [0, 1]
    plt.title(title)
    ks_plot, = plt.plot([score_info['ks'].argmax() + 1, score_info['ks'].argmax() + 1],
                        [score_info['good.perc.allgood.acc'].iloc[score_info['ks'].argmax()],
                         score_info['bad.perc.allbad.acc'].iloc[score_info['ks'].argmax()]], color=kscol, linewidth=0.5)

    plt.text(score_info['ks'].argmax() + 1.5, score_info['ks'].iloc[score_info['ks'].argmax()],
             'KS = {}'.format(round(score_info['ks'].iloc[score_info['ks'].argmax()], 4)), color=kscol)

    plt.legend([bad_plot, good_plot, ks_plot], ['Bad', 'good', 'KS'])
    ax.set_xticks(np.linspace(0.5, 0.5 + len(score_info) - 1, len(score_info)))
    ax.set_xticklabels(interval_to_str(score_info), rotation=90, fontsize=6)
    plt.axis()
    if image_path:
        plt.savefig(image_path, dpi=500)

    plt.close()


def btom(path, info_data_to_bin, corr_t=0.8, seed=1234, iv_min=0.02, iv_max=100, p0=580, sample_weights=None,
         score_k=10, p_min=0.05, theta=None, pdo=50):
    """
    分箱转模型
    :param path:
    :param info_data_to_bin:
    :param corr_t:
    :param seed:
    :param iv_min:
    :param iv_max:
    :param p0:
    :param sample_weights:
    :param score_k:
    :param p_min:
    :param theta:
    :param pdo:
    :return:
    """
    sample_weights = [1, 1] if sample_weights is None else sample_weights
    # 加载数据
    print('botm start at {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    argms = deepcopy(info_data_to_bin['argms'])
    y_label = argms.loc['y_label'].iloc[0]
    good_value = argms.loc['good_value'].iloc[0]
    bad_value = argms.loc['bad_value'].iloc[0]
    train_index = deepcopy(info_data_to_bin['train_index'])
    test_index = deepcopy(info_data_to_bin['test_index'])
    traindata = deepcopy(info_data_to_bin['traindata'])
    testdata = deepcopy(info_data_to_bin['testdata'])
    traindataother = deepcopy(info_data_to_bin['traindataother'])
    testdataother = deepcopy(info_data_to_bin['testdataother'])

    # 根据IV值筛选
    print('For {} variables start screening by IV'.format(len(train_index)))
    iv_dict = {k: v['varindex']['IV'].iloc[0] for k, v in train_index.items()}
    name = [k for k, v in iv_dict.items() if iv_min <= v <= iv_max]
    train_index = {k: v for k, v in train_index.items() if k in name}
    print('After the IV screening, remaining {} variables'.format(len(train_index)))

    # 根据相关性筛选
    print('For {} variables start screening by correlation'.format(len(train_index)))
    train_woematrix = pd.concat([v['sampleinfo']['woe'].to_frame(name=k) for k, v in train_index.items()], axis=1)
    train_woematrix = filter_var(train_woematrix=train_woematrix, corr_t=corr_t)
    train_index = {k: v for k, v in train_index.items() if k in train_woematrix.columns}
    print('After the correlation screening, remaining {} variables'.format(len(train_index)))

    # 训练logistic回归模型#################
    print('trainning logistic model')
    weights = traindata.iloc[:, 0]
    weights = weights.apply(lambda x: sample_weights[0] if x == good_value else sample_weights[1])

    logitresult = logisticscreen(y=traindata.iloc[:, 0], datax=train_woematrix, p_min=p_min, weights=weights)
    name = logitresult['name']
    glm_fit = logitresult['model']
    coeff = logitresult['coeff']
    coeff0 = logitresult['coeff0']

    # 计算整理训练集信息
    train_woematrix = train_woematrix[name]
    train_index = {k: v for k, v in train_index.items() if k in name}

    # 生成评分卡
    print('building scorecard')
    tmp_card_list = []
    for k, v in train_index.items():
        tmp_card = deepcopy(v['card'])
        tmp_card.reset_index(inplace=True)
        tmp_card.rename(columns={'index': 'range'}, inplace=True)
        tmp_card.index = tmp_card.index.map(lambda x: '{}.{}'.format(k, x + 1))
        tmp_card = tmp_card[[tmp_card.columns[1], tmp_card.columns[0]] + list(tmp_card.columns[2:])]
        tmp_card_list.append(tmp_card)

    train_card = pd.concat(tmp_card_list)

    scorecard = standcard(y=traindata.iloc[:, 0], train_card=train_card, pdo=pdo, p0=p0, good_value=good_value,
                          bad_value=bad_value, coeff=coeff, coeff0=coeff0, theta=theta)

    theta = len(traindata.iloc[:, 0][traindata.iloc[:, 0] == bad_value]) / len(
        traindata.iloc[:, 0][traindata.iloc[:, 0] == good_value])

    print('computing train_samples score')
    train_scorematrix = dtos(traindata, scorecard)
    train_scorematrix['score'] = train_scorematrix.sum(axis=1)
    train_scorematrix = pd.concat([traindataother, train_scorematrix], axis=1)
    print('computing train_samples p_value')

    # 根据训练集预测p值
    train_scorematrix['p'] = glm_fit.predict(train_woematrix)
    train_woematrix = pd.concat([traindataother, train_woematrix], axis=1)

    # 训练集总分分箱信息计算
    train_score_info = binning(y=traindata.iloc[:, 0], var=train_scorematrix['score'], break_type=1, num_bins=score_k)[
        'card']
    train_score_info_breakpoints = binning(y=traindata.iloc[:, 0], var=train_scorematrix['score'], break_type=1,
                                           num_bins=score_k)['breakpoints']
    train_score_info = train_score_info[['label', 'good.count', 'bad.count', 'count', 'count.perc', 'count.perc.acc',
                                         'good.perc.allgood.acc', 'bad.perc.allbad.acc', 'ks']]

    print('computing train_woematrix corr')
    train_corr = train_woematrix.corr()
    train_pm = perfm(traindata[y_label], train_scorematrix['p'])

    # 计算整理测试集信息
    test_index = {k: v for k, v in test_index.items() if k in name}
    test_woematrix = pd.concat([v['sampleinfo']['woe'].to_frame(name=k) for k, v in test_index.items()], axis=1)

    print('computing test_samples score')
    test_scorematrix = dtos(testdata, scorecard)
    test_scorematrix['score'] = test_scorematrix.sum(axis=1)
    test_scorematrix = pd.concat([testdataother, test_scorematrix], axis=1)

    # 根据训练集预测p值
    print('computing test_samples p_value')
    test_scorematrix['p'] = glm_fit.predict(test_woematrix)
    test_woematrix = pd.concat([testdataother, test_woematrix], axis=1)

    # 测试集总分分箱信息计算
    train_score_info_breakpoints[0] = -np.Inf
    train_score_info_breakpoints[-1] = np.Inf
    test_score_info = binning(y=testdata.iloc[:, 0], var=test_scorematrix['score'], break_type=1,
                              breakpoints=train_score_info_breakpoints)['card']

    test_score_info = test_score_info[
        ['label', 'good.count', 'bad.count', 'count', 'count.perc', 'count.perc.acc',
         'good.perc.allgood.acc', 'bad.perc.allbad.acc', 'ks']]

    print('computing test_woematrix corr')
    test_corr = test_woematrix.corr()
    test_pm = perfm(testdata[y_label], test_scorematrix['p'])

    # 保存信息到_btom.xlsx
    info = dict()
    info['scorecard'] = scorecard
    info['model'] = glm_fit.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']]
    info['train_index'] = train_index
    info['train_corr'] = train_corr
    info['train_score_info'] = train_score_info
    info['test_index'] = test_index
    info['test_corr'] = test_corr
    info['test_score_info'] = test_score_info

    argms_dict = dict()
    argms_dict['corr_t'] = corr_t
    argms_dict['seed'] = seed
    argms_dict['iv_min'] = iv_min
    argms_dict['iv_max'] = iv_max
    argms_dict['p0'] = p0
    argms_dict['sample_weights'] = 'goodweights:{},badweights:{}'.format(sample_weights[0], sample_weights[1])
    argms_dict['score_k'] = score_k
    argms_dict['p_min'] = p_min
    argms_dict['theta'] = theta
    argms_dict['pdo'] = pdo
    argms_dict['donetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    argms = pd.DataFrame({k: [v] for k, v in argms_dict.items()})
    argms = argms.T
    argms.columns = ['argmsvalue']
    info['argms'] = argms

    info['train_index'] = {k: v['card'] for k, v in train_index.items()}
    info['test_index'] = {k: v['card'] for k, v in test_index.items()}

    print('save to {}_btom.xlsx'.format(path))
    write_dict_to_excel(path='{}_btom.xlsx'.format(path), info=info)

    print('saving train(and test) AUC-plot and KS-plot')
    # 训练集ROC作图
    title = 'train-ROC'
    image_path = '{}_{}.png'.format(path, title)
    plot_roc(train_pm, title, image_path)
    insert_image_to_excel('{}_btom.xlsx'.format(path), 'train_score_info', image_path, anchor='A15')

    # 训练集PR作图
    title = 'train-PR'
    image_path = '{}_{}.png'.format(path, title)
    plot_pr(train_pm, title, image_path)
    insert_image_to_excel('{}_btom.xlsx'.format(path), 'train_score_info', image_path, anchor='K15')

    # 训试集KS作图
    title = 'train-KS'
    image_path = '{}_{}.png'.format(path, title)
    plot_ks(train_score_info, title=title, image_path=image_path)
    insert_image_to_excel('{}_btom.xlsx'.format(path), 'train_score_info', image_path, anchor='A45')

    # 测试集ROC作图
    title = 'test-ROC'
    image_path = '{}_{}.png'.format(path, title)
    plot_roc(test_pm, title, image_path)
    insert_image_to_excel('{}_btom.xlsx'.format(path), 'test_score_info', image_path, anchor='A15')

    # 测试集PR作图
    title = 'test-PR'
    image_path = '{}_{}.png'.format(path, title)
    plot_pr(test_pm, title, image_path)
    insert_image_to_excel('{}_btom.xlsx'.format(path), 'test_score_info', image_path, anchor='K15')

    # 测试集KS作图
    title = 'test-KS'
    image_path = '{}_{}.png'.format(path, title)
    plot_ks(test_score_info, title=title, image_path=image_path)
    insert_image_to_excel('{}_btom.xlsx'.format(path), 'test_score_info', image_path, anchor='A45')
