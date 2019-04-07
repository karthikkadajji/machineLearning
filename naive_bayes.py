import pandas as pd
import math
import argparse as ag
import os
import csv

store_val_zero = list()
store_val_one = list()
probability_one = 0.0
probability_zero = 0.0


def check_location(loc_file):
    if not os.path.exists(loc_file):
        raise ag.ArgumentTypeError("{0} does not exist".format(loc_file))
    return loc_file


def check_output(loc_file):
    if not os.path.exists(loc_file):
        file_handler = open(loc_file, 'w', newline='')
        return file_handler
    else:
        handle_file = open(loc_file, 'w', newline='')
        return handle_file


def calculate_estimate_b(x):
    return x['gaussian_estimate_x1_B'] * x['gaussian_estimate_x2_B'] * probability_zero


def calculate_estimate_a(y):
    return y['gaussian_estimate_x1_A'] * y['gaussian_estimate_x2_A'] * probability_one


def calculate_ratio_b(k):
    return k['probability_B']/(k['probability_B'] + k['probability_A'])


def calculate_ratio_a(l):
    return 1 - l['ratio_with_num_B']


def compare_classification(z):
    if z['class'] == z[0]:
        return 0
    else:
        return 1


def compare(m):
    if m['ratio_with_num_B'] > m['ratio_with_num_A']:
        return 0
    else:
        return 1


def total_calculation(_temp_df_zero, _temp_df_one, _main_df):
    # For Class B First Attribute
    mean_x1_zero = _temp_df_zero[1].mean()
    store_val_zero.append(mean_x1_zero)
    _temp_df_zero['for_sd_x1'] = _temp_df_zero[1].apply(lambda x: math.pow((x - mean_x1_zero), 2))
    sd_1_zero = _temp_df_zero['for_sd_x1'].sum() / (_temp_df_zero.shape[0] - 1)
    store_val_zero.append(sd_1_zero)

    # For Class B Second Attribute
    mean_x2_zero = _temp_df_zero[2].mean()
    store_val_zero.append(mean_x2_zero)
    _temp_df_zero['for_sd_x2'] = _temp_df_zero[2].apply(lambda x: math.pow((x - mean_x2_zero), 2))
    sd_2_zero = _temp_df_zero['for_sd_x2'].sum() / (_temp_df_zero.shape[0] - 1)
    store_val_zero.append(sd_2_zero)

    # For Class A first attribute
    mean_x1_one = _temp_df_one[1].mean()
    store_val_one.append(mean_x1_one)
    _temp_df_one['for_sd_x1'] = _temp_df_one[1].apply(lambda x: math.pow((x - mean_x1_one), 2))
    sd_1_one = _temp_df_one['for_sd_x1'].sum() / (_temp_df_one.shape[0] - 1)
    store_val_one.append(sd_1_one)

    # For Second Variable
    mean_x2_one = _temp_df_one[2].mean()
    store_val_one.append(mean_x2_one)
    _temp_df_one['for_sd_x2'] = _temp_df_one[2].apply(lambda x: math.pow((x - mean_x2_one), 2))
    sd_2_one = _temp_df_one['for_sd_x2'].sum() / (_temp_df_one.shape[0] - 1)
    store_val_one.append(sd_2_one)

    # Gaussian Estimate for attributes
    _main_df['gaussian_estimate_x1_B'] = _main_df[1].\
        apply(lambda x: (1/math.sqrt(2*math.pi*sd_1_zero) * math.exp(-(math.pow(x - mean_x1_zero, 2))/(2 * sd_1_zero))))
    _main_df['gaussian_estimate_x2_B'] = _main_df[2].\
        apply(lambda x: (1/math.sqrt(2*math.pi*sd_2_zero) * math.exp(-(math.pow(x - mean_x2_zero, 2))/(2 * sd_2_zero))))

    _main_df['gaussian_estimate_x1_A'] = _main_df[1]. \
        apply(lambda x: (1 / math.sqrt(2 * math.pi * sd_1_one) * math.exp(-(math.pow(x - mean_x1_one, 2)) / (2 * sd_1_one))))
    _main_df['gaussian_estimate_x2_A'] = _main_df[2]. \
        apply(lambda x: (1 / math.sqrt(2 * math.pi * sd_2_one) * math.exp(-(math.pow(x - mean_x2_one, 2)) / (2 * sd_2_one))))

    # print(_main_df['gaussian_estimate_x1'].head(5))

    _main_df['probability_B'] = _main_df.apply(calculate_estimate_b, axis=1)
    _main_df['probability_A'] = _main_df.apply(calculate_estimate_a, axis=1)

    _main_df['ratio_with_num_B'] = _main_df.apply(calculate_ratio_b, axis=1)
    _main_df['ratio_with_num_A'] = _main_df.apply(calculate_ratio_a, axis=1)

    _main_df['class'] = _main_df.apply(compare, axis=1)
    _main_df['misclassified'] = _main_df.apply(compare_classification, axis=1)

    misclassified = int(_main_df['misclassified'].sum())

    return misclassified


def main():
    global probability_one
    global probability_zero
    parser = ag.ArgumentParser()
    parser.add_argument("--data", help="data filename", type=check_location)
    parser.add_argument("--output", help="output file for final decision tree", type=check_output)
    args = parser.parse_args()
    file_name = args.data
    fh = args.output
    _df = pd.read_csv(file_name, delimiter='\t', header=None)
    _df = _df.dropna(axis=1, how='all')
    _df[0] = _df[0].map({'A': 1, 'B': 0})
    _df_one = _df[_df[0] == 1].reset_index()
    _df_zero = _df[_df[0] == 0].reset_index()
    probability_one = _df_one.shape[0] / _df.shape[0]
    probability_zero = _df_zero.shape[0] / _df.shape[0]
    misclassified_num = total_calculation(_df_zero, _df_one, _df)
    store_val_one.append(probability_one)
    store_val_zero.append(probability_zero)
    misclassified = list()
    misclassified.append(misclassified_num)
    print("Writing to the specified output file.")
    tsv_writer = csv.writer(fh, delimiter='\t')
    tsv_writer.writerow(store_val_one)
    tsv_writer.writerow(store_val_zero)
    tsv_writer.writerow(misclassified)
    fh.close()
    print("Output file has been created.")


if __name__ == "__main__":
    main()
