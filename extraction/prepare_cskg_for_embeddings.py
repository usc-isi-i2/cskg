import csv
import os
import config
import random

VERSION=config.VERSION
cskg_edges_file='../output_v%s/cskg_merged/edges_v%s.csv' % (VERSION, VERSION)
output_dir='../output_v%s/emb_data' % VERSION

def write_data(l, f):
    with open(f, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        for line in l:
            spamwriter.writerow(line)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

rows=[]
with open(cskg_edges_file, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    spamreader.__next__()
    for row in spamreader:
        my_row=row[:3]
        rows.append(my_row)

random.shuffle(rows)  # randomly shuffles the ordering of filenames

split_1 = int(0.8 * len(rows))
#split_2 = int(0.9 * len(rows))
train_data = rows[:split_1]
dev_data = rows[split_1:]
#test_data = rows[split_2:]

write_data(train_data, '%s/train.csv' % output_dir)
write_data(dev_data, '%s/dev.csv' % output_dir)
#write_data(test_data, '%s/test.csv' % output_dir)



