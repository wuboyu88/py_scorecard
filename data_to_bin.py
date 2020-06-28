from common_util import *
from datetime import datetime
from collections import OrderedDict
from sklearn.model_selection import train_test_split


def dtob(path, data=None, y_label=None, special_value=None, exclude=None, bestks_k=0, break_type=1, bin_rate_min=0.05,
         train_perc=0.7, sv_perc=0.8, num_bins=10, min_num_bins=3, max_num_bins=10, bad_value=1, closed_on_right=True,
         good_value=0, replace_value=1, comb_type='combinning', woe_stand='monotonous', seed=1234,
         manuel_breakpoints_dict=None):
    """
    数据转分箱
    :param path:
    :param data:
    :param y_label:
    :param special_value:
    :param exclude:
    :param bestks_k:
    :param break_type:
    :param bin_rate_min:
    :param train_perc:
    :param sv_perc:
    :param num_bins:
    :param min_num_bins:
    :param max_num_bins:
    :param bad_value:
    :param closed_on_right:
    :param good_value:
    :param replace_value:
    :param comb_type:
    :param woe_stand:
    :param seed:
    :param manuel_breakpoints_dict:
    :return:
    """
    exclude = [] if exclude is None else exclude
    manuel_breakpoints_dict = dict() if manuel_breakpoints_dict is None else manuel_breakpoints_dict

    # 环境准备
    print('Data to bin start at {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('Preparing the environment.')

    # Y标签识别
    if y_label is not None:
        if y_label not in data.columns:
            return 'No {} in data.'.format(y_label)
        y_index = data.columns.to_list().index(y_label)
        col_index = list(range(data.shape[1]))
        col_index.remove(y_index)
        data = data.iloc[:, [y_index] + col_index]
    else:
        y_label = data.columns[0]

    if len(data.columns.difference([y_label] + exclude)) == 0:
        return 'NO x-variable beside {}.'.format(','.join([y_label] + exclude))

    print('y_label: {}'.format(y_label))

    # 单一值变量筛选#
    if sv_perc != 1:
        print('Start screening by single value')
        sinvar = single_var_name(data[data.columns.difference([y_label] + exclude)], sv_perc=sv_perc)

        if len(sinvar) > 0:
            print('End single value screening: removed {} columns, they are {}'.format(len(sinvar), ','.join(sinvar)))
        data = data[[ele for ele in data.columns if ele not in sinvar]]

        if len(data.columns.difference([y_label] + exclude)) == 0:
            return 'NO x-variable beside {}.'.format(','.join([y_label] + exclude))

        else:
            print("No variable's single value rate more than {}".format(sv_perc))

    # 分训练测试集
    if train_perc > 1 or train_perc <= 0:
        return "percentage of traindata must be '<= 1' or '>0'."
    else:
        data_x = data.drop([y_label], axis=1)
        data_y = data[y_label]
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=train_perc, random_state=seed,
                                                            stratify=data_y)
        traindata = pd.merge(y_train, x_train, left_index=True, right_index=True)
        traindataother = traindata[exclude]
        traindata = traindata[[ele for ele in traindata.columns if ele not in exclude]]

        testdata = pd.merge(y_test, x_test, left_index=True, right_index=True)
        testdataother = testdata[exclude]
        testdata = testdata[[ele for ele in testdata.columns if ele not in exclude]]

        # 如果除了special_value之外只有一个值，则不予考虑
        used_columns = [ele for ele in traindata.columns if f(traindata[ele])]
        traindata = traindata[used_columns]
        testdata = testdata[used_columns]
        print('traindata: {}; testdata: {}'.format(len(traindata), len(testdata)))

    # 训练集分箱计算
    print('Start counting the box information')
    train_index = allvarindex(datax=traindata.iloc[:, 1:], y=traindata.iloc[:, 0], bad_value=bad_value,
                              good_value=good_value, num_bins=num_bins, min_num_bins=min_num_bins,
                              max_num_bins=max_num_bins, bestks_k=bestks_k, special_value=special_value,
                              bin_rate_min=bin_rate_min, replace_value=replace_value, break_type=break_type,
                              comb_type=comb_type, woe_stand=woe_stand, closed_on_right=closed_on_right,
                              manuel_breakpoints_dict=manuel_breakpoints_dict)

    # 整理训练集信息
    print('summary traininfo')
    train_index = OrderedDict(sorted(train_index.items(), key=lambda x: x[1]['varindex']['IV'][0], reverse=True))
    train_iv_ks_numbin = pd.concat([v['varindex'] for v in train_index.values()])

    trainvarinfo = pd.concat([varinfo(traindata[ele], special_value) for ele in traindata.iloc[:, 1:]])
    trainvarinfo = pd.merge(trainvarinfo, train_iv_ks_numbin, left_index=True, right_index=True)
    trainvarinfo.sort_values('IV', ascending=False, inplace=True)

    # train_woematrix = pd.concat([v['sampleinfo']['woe'].to_frame(name=k) for k, v in train_index.items()], axis=1)
    # train_woematrix = pd.merge(traindata.iloc[:, 0], train_woematrix, left_index=True, right_index=True)
    # train_woematrix = pd.merge(traindataother, train_woematrix, left_index=True, right_index=True)

    tmp_card_list = []
    for k, v in train_index.items():
        tmp_card = deepcopy(v['card'])
        tmp_card.reset_index(inplace=True)
        tmp_card.rename(columns={'index': 'range'}, inplace=True)
        tmp_card.index = tmp_card.index.map(lambda x: '{}.{}'.format(k, x + 1))
        tmp_card = tmp_card[[tmp_card.columns[1], tmp_card.columns[0]] + list(tmp_card.columns[2:])]
        tmp_card_list.append(tmp_card)

    train_card = pd.concat(tmp_card_list)

    # 测试集分箱计算
    trainbreakpoints = {k: v['breakpoints'] for k, v in train_index.items()}
    testdata = testdata[[y_label] + list(trainbreakpoints.keys())]

    test_index = allvarindexfortest(testdata.iloc[:, 1:], testdata.iloc[:, 0], trainbreakpoints=trainbreakpoints,
                                    closed_on_right=closed_on_right, break_type=break_type, bad_value=bad_value,
                                    good_value=good_value, special_value=special_value, bin_rate_min=bin_rate_min,
                                    replace_value=replace_value)

    # 测试集信息整理
    test_index = OrderedDict({k: test_index[k] for k in train_index.keys()})
    test_iv_ks_numbin = pd.concat([v['varindex'] for v in test_index.values()])
    testvarinfo = pd.concat([varinfo(testdata[ele], special_value) for ele in testdata.iloc[:, 1:]])
    testvarinfo = pd.merge(testvarinfo, test_iv_ks_numbin, left_index=True, right_index=True)
    testvarinfo = testvarinfo.reindex(trainvarinfo.index)

    # test_woematrix = pd.concat([v['sampleinfo']['woe'].to_frame(name=k) for k, v in test_index.items()], axis=1)
    # test_woematrix = pd.merge(testdata.iloc[:, 0], test_woematrix, left_index=True, right_index=True)
    # test_woematrix = pd.merge(testdataother, test_woematrix, left_index=True, right_index=True)

    tmp_card_list = []
    for k, v in test_index.items():
        tmp_card = deepcopy(v['card'])
        tmp_card.reset_index(inplace=True)
        tmp_card.rename(columns={'index': 'range'}, inplace=True)
        tmp_card.index = tmp_card.index.map(lambda x: '{}.{}'.format(k, x + 1))
        tmp_card = tmp_card[[tmp_card.columns[1], tmp_card.columns[0]] + list(tmp_card.columns[2:])]
        tmp_card_list.append(tmp_card)

    test_card = pd.concat(tmp_card_list)

    # 保存信息到_dtob.xlsx
    info = dict()

    print('summary testinfo')
    test_card = test_card.add_suffix('_TEST')
    card = pd.concat([train_card, test_card.iloc[:, 1], test_card.iloc[:, 3:]], axis=1)
    card['psi'] = card.apply(lambda x: psi(x['count.perc'], x['count.perc_TEST']), axis=1)
    info['testvarinfo'] = testvarinfo
    info['card'] = card

    info['trainvarinfo'] = trainvarinfo
    argms_dict = dict()
    argms_dict['y_label'] = y_label
    argms_dict['special_value'] = ','.join([str(ele) for ele in special_value])
    argms_dict['exclude'] = ','.join([str(ele) for ele in exclude])
    argms_dict['bestks_k'] = bestks_k
    argms_dict['break_type'] = break_type
    argms_dict['bin_rate_min'] = bin_rate_min
    argms_dict['train_perc'] = train_perc
    argms_dict['sv_perc'] = sv_perc
    argms_dict['num_bins'] = num_bins
    argms_dict['min_num_bins'] = min_num_bins
    argms_dict['max_num_bins'] = max_num_bins
    argms_dict['bad_value'] = bad_value
    argms_dict['closed_on_right'] = closed_on_right
    argms_dict['good_value'] = good_value
    argms_dict['replace_value'] = replace_value
    argms_dict['comb_type'] = comb_type
    argms_dict['woe_stand'] = woe_stand
    argms_dict['seed'] = seed
    argms_dict['donetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    argms = pd.DataFrame({k: [v] for k, v in argms_dict.items()})
    argms = argms.T
    argms.columns = ['argmsvalue']
    info['argms'] = argms

    print('saving info to {}_dtob.xlsx'.format(path))
    write_dict_to_excel(path='{}_dtob.xlsx'.format(path), info=info)

    info['train_card'] = train_card
    info['train_index'] = train_index
    info['test_index'] = test_index
    info['traindata'] = traindata
    info['testdata'] = testdata
    info['traindataother'] = traindataother
    info['testdataother'] = testdataother
    info['data'] = data
    return info
