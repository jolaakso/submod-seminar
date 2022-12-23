import numpy as np
import sys

def print_data(words, sentences, rouge1rec, rouge2rec, rouge1prec, rouge2prec):
    print("{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(words, sentences, rouge1rec, rouge2rec, rouge1prec, rouge2prec))

def statistics(filename='results/res.csv'):
    rows = []
    with open(filename, 'r') as f:
        for line in f:
            if line[0] != '#':
                cells = line.split(',')
                summary_ixs = cells[1].split(' ')
                rows.append([int(cells[2]), len(summary_ixs), float(cells[3]), float(cells[4]), float(cells[5]), float(cells[6])])
    array = np.array(rows, dtype='float')

    maxs = np.amax(array, axis=0)
    medians = np.median(array, axis=0)
    variance = np.var(array, axis=0)
    mins = np.amin(array, axis=0)
    means = np.mean(array, axis=0)

    print_data(means[0], means[1], means[2], means[3], means[4], means[5])
    print_data(medians[0], medians[1], medians[2], medians[3], medians[4], medians[5])
    print_data(variance[0], variance[1], variance[2], variance[3], variance[4], variance[5])
    print_data(mins[0], mins[1], mins[2], mins[3], mins[4], mins[5])
    print_data(maxs[0], maxs[1], maxs[2], maxs[3], maxs[4], maxs[5])

if __name__ == "__main__":
    target_file = 'results/res.csv'
    if len(sys.argv) >= 2:
        target_file = sys.argv[1]
    statistics(target_file)
