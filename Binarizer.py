import csv
import numpy as np


def binarize(path, bin_count):
    with open(path+".csv", 'r', newline='') as file_in:
        with open(path +"_" + str(bin_count) + "_bins.csv", 'w', newline='') as file_out:
            reader = csv.reader(file_in)
            writer = csv.writer(file_out)

            # save rows, get min and max com
            rows = []
            in_header = True
            in_first = True
            min_com = 0
            max_com = 0
            base_acc = 0
            for row in reader:
                if in_header:
                    in_header = False
                    writer.writerow(row)
                    continue
                rows += [row]
                com = float(row[0])
                if in_first:
                    in_first = False
                    min_com = com
                    max_com = com
                    base_acc = float(row[1])
                    continue
                if com < min_com:
                    min_com = com
                if com > max_com:
                    max_com = com

            # fill bins
            bin_width = (max_com - min_com) / bin_count
            bins = []
            for i in range(bin_count):
                bins += [[]]
            for row in rows:
                com = float(row[0])
                bin_idx = int((com - min_com) / bin_width)
                if bin_idx == bin_count:
                    bin_idx = bin_count - 1
                bin = bins[bin_idx]
                bin += [(float(row[2]))]

            # write file
            for i in range(len(bins)):
                if len(bins[i]) > 0:
                    writer.writerow([str(min_com + (i + 0.5) * bin_width), str(base_acc), str(np.mean(bins[i]))])


binarize("C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\results\\creditRiskAssessment", 60)
