import pandas as pd
import csv
import argparse as ag
import os


_learning_rate = 1
_wts = list()
iteration_num = 100


def check_location(loc_file):
    if not os.path.exists(loc_file):
        raise ag.ArgumentTypeError("{0} does not exist".format(loc_file))
    return loc_file


def check_output(loc_file):
    if not os.path.exists(loc_file):
        file_handler = open(loc_file, 'wt')
        return file_handler
    else:
        handle_file = open(loc_file, 'wt')
        return handle_file


def activation_func(x):
    value = _wts[0]*x[1] + _wts[1] * x[2] + _wts[2] * x[3]
    if value > 0:
        return 1
    else:
        return 0


def error_calc(y):
    return y[0] - y['activation_func_value']


def compare_output_target(z):
    if z[0] == z['activation_func_value']:
        return 0
    else:
        return 1


def calculation_constant_learning_rate(_df):
    i = 0
    _misclassified = list()
    while i <= iteration_num:
        _wt_update = list()
        _df['activation_func_value'] = _df.apply(activation_func, axis=1)
        _df['error'] = _df.apply(error_calc, axis=1)
        _df['compare'] = _df.apply(compare_output_target, axis=1)
        _misclassified.insert(i, int(_df['compare'].sum()))
        # _df['_wt_update'] = _df.apply(_wt_update, axis=1)
        len_of_wts = len(_wts)
        k = 0
        while k < len_of_wts:
            _wt_update.insert(k, sum(_df[k + 1] * _df['error'] * _learning_rate))
            # _wt_update.insert(k, sum(_df[k + 1] * _df['error'] * (_learning_rate/(i+1))))
            _wts[k] = _wts[k] + _wt_update[k]
            k = k + 1
        # print(_wts)
        i = i + 1
    return _misclassified


def calculation_anneal_learning_rate(df):
    _wt_length = len(_wts)
    wt = 0
    while wt < _wt_length:
        _wts[wt] = 0
        wt = wt + 1
    i = 0
    _misclassified_anneal = list()
    while i <= iteration_num:
        _wt_update_anneal = list()
        df['activation_func_value'] = df.apply(activation_func, axis=1)
        df['error'] = df.apply(error_calc, axis=1)
        df['compare'] = df.apply(compare_output_target, axis=1)
        _misclassified_anneal.insert(i, int(df['compare'].sum()))
        # _df['_wt_update'] = _df.apply(_wt_update, axis=1)
        len_of_wts = len(_wts)
        k = 0
        while k < len_of_wts:
            # _wt_update.insert(k, sum(_df[k + 1] * _df['error'] * _learning_rate))
            _wt_update_anneal.insert(k, sum(df[k + 1] * df['error'] * (_learning_rate/(i+1))))
            _wts[k] = _wts[k] + _wt_update_anneal[k]
            k = k + 1
        # print(_wts)
        i = i + 1
    return _misclassified_anneal


def main():
    parser = ag.ArgumentParser()
    parser.add_argument("--data", help="data filename", type=check_location)
    parser.add_argument("--output", help="output file for final decision tree", type=check_output)
    args = parser.parse_args()
    file_name = args.data
    fh = args.output
    _main_df = pd.read_csv(file_name, sep='\t', header=None)
    _main_df = _main_df.dropna(axis=1, how='all')
    _main_df[0] = _main_df[0].map({'A': 1, 'B': 0})
    _main_df[3] = 1
    num_of_cols = _main_df.shape[1]
    i = 0
    while i < num_of_cols - 1:
        _wts.insert(i, 0)
        i = i+1
    _misclassified_constant = calculation_constant_learning_rate(_main_df)
    _misclassified_anneal = calculation_anneal_learning_rate(_main_df)
    # with open('output.tsv', 'wt') as out_file:
    #     tsv_writer = csv.writer(out_file, delimiter='\t')
    #     tsv_writer.writerow(_misclassified_constant)
    #     tsv_writer.writerow(_misclassified_anneal)
    tsv_writer = csv.writer(fh, delimiter='\t')
    tsv_writer.writerow(_misclassified_constant)
    tsv_writer.writerow(_misclassified_anneal)


if __name__ == "__main__":
    main()
