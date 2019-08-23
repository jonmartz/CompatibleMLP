import csv


def encode(path, cols_to_encode, make_balanced=False, zeroes_fraction=None, ones_fraction=None):
    cols_to_encode.sort()
    header = []
    dict_cols = []
    with open(path+".csv", 'r', newline='') as file_in:
        with open(path +"_encoded.csv", 'w', newline='') as file_out:
            reader = csv.reader(file_in)
            writer = csv.writer(file_out)
            in_header = True
            balance = [0, 0]
            for row in reader:

                if in_header:
                    in_header = False
                    header = row
                    writer.writerow(header)
                    # add dict for each col
                    for i in header:
                        dict_cols += [{}]
                    continue

                if make_balanced:
                    this_label = int(row[0])
                    other_label = (int(row[0]) + 1) % 2
                    this_fraction = ones_fraction
                    other_fraction = zeroes_fraction
                    if this_label == 0:
                        this_fraction = zeroes_fraction
                        other_fraction = ones_fraction
                    if balance[this_label]*this_fraction > balance[other_label]*other_fraction:
                        continue
                    balance[int(row[0])] += 1
                row_out = []

                for col in range(len(header)):
                    if not col in cols_to_encode:
                        row_out += [row[col]]
                        continue
                    value = row[col]
                    dict = dict_cols[col]
                    if not value in dict:
                        dict[value] = str(len(dict))
                    row_out += [dict[value]]
                writer.writerow(row_out)

    with open(path + "_dictionary.csv", 'w', newline='') as file_dict:
        writer = csv.writer(file_dict)
        for col in range(len(header)):
            writer.writerow([header[col], 'CODES'])
            dict = dict_cols[col]
            for value, code in dict.items():
                writer.writerow([value, str(code)])


# cols_to_encode = [1,3,9,10]
# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\recividismPrediction\\compas-scores-two-years'

cols_to_encode = [2,3]
dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\fraudDetection\\train_transaction_short'

encode(dataset_path, cols_to_encode)
