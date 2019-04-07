# import sys
import csv
import argparse
import os


def check_location(x):
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


parser = argparse.ArgumentParser()
parser.add_argument("--data", help="data filename", type=check_location)
parser.add_argument("--learningRate", help="learning rate required for gradient descent", type=float)
parser.add_argument("--threshold", help="threshold for error calculation", type=float)
args = parser.parse_args()
filename = args.data
learningRate = args.learningRate
threshold = args.threshold
csv_file = open(filename, "r")
csv_reader = csv.reader(csv_file, delimiter=',')
first = next(csv_reader)
n_weights = len(first)
csv_file.seek(0)
data = []
for reader in csv_reader:  
    data.append(reader)
    # becz the first element was already read so we put the stuff to last.
w = list(range(0, n_weights))
len_data = len(data)
x0 = [1]*len_data
# SSE = 0
for i in range(0, n_weights):
    w[i] = 0


def training_output(iteration_num, data, w):
    sse = 0
    # make this cleaner
    func = [0] * len_data
    error = [0] * len_data
    gradient = [0]*n_weights
    
    for i in range(0, len_data):
        for j in range(0, n_weights):
            if j == n_weights-1:
                
                func[i] += w[j] * x0[i]
            else:
                func[i] += w[j] * float(data[i][j])
        error[i] = float(data[i][n_weights-1]) - func[i]
        sse += error[i] * error[i]
        # gradient = [0] * n_weights
    for j in range(0, n_weights):
        for k in range(0, len(data)):
            if j == n_weights-1:
                gradient[j] += x0[k] * error[k]
            else:
                gradient[j] += float(data[k][j]) * error[k]
    remaining_weights = ', '.join(map(str, w[:n_weights - 1]))
    print(str(iteration_num) + "," + str(w[n_weights - 1]) + "," + remaining_weights + "," + str(sse))
    for l in range(0, n_weights):
        w[l] = w[l] + float(learningRate) * float(gradient[l])
    
    # print(str(w[n_weights-1]) + "," + w[:n_weights-1], sse)
    return sse


def main():
    itr = 0
    old_sse = training_output(itr, data, w)
    diff = old_sse
    
    while diff > float(threshold):
        itr += 1
        # print(itr)
        new_sse = training_output(itr, data, w)
        
        diff = old_sse - new_sse
        # print(new_sse,old_sse)
        old_sse = new_sse
        
        # print(diff)


main()
        
    
        
   


         




#def gradient_descent(w,)
