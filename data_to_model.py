from data_to_bin import dtob
from bin_to_model import btom


def dtom(path, data=None, y_label=None, special_value=None, exclude=None, bestks_k=0, break_type=1, bin_rate_min=0.05,
         train_perc=0.7, sv_perc=0.8, num_bins=10, min_num_bins=3, max_num_bins=10, bad_value=1, closed_on_right=True,
         good_value=0, replace_value=1, comb_type='combinning', woe_stand='monotonous', seed=1234, corr_t=0.8,
         iv_min=0.02, iv_max=100, p0=580, sample_weights=None, score_k=10, p_min=0.05, theta=None, pdo=50):
    info_data_to_bin = dtob(path=path, data=data, y_label=y_label, special_value=special_value, exclude=exclude,
                            bestks_k=bestks_k, break_type=break_type, bin_rate_min=bin_rate_min, train_perc=train_perc,
                            sv_perc=sv_perc, num_bins=num_bins, min_num_bins=min_num_bins, max_num_bins=max_num_bins,
                            bad_value=bad_value, closed_on_right=closed_on_right, good_value=good_value,
                            replace_value=replace_value, comb_type=comb_type, woe_stand=woe_stand, seed=seed)

    btom(path=path, info_data_to_bin=info_data_to_bin, corr_t=corr_t, seed=seed, iv_min=iv_min, iv_max=iv_max, p0=p0,
         sample_weights=sample_weights, score_k=score_k, p_min=p_min, theta=theta, pdo=pdo)
