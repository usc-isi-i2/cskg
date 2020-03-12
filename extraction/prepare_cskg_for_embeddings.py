import csv
import os
import config
import random

VERSION=config.VERSION

def write_data(l, f):
    with open(f, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        for line in l:
            spamwriter.writerow(line)

for dataset in ['cskg_merged', 'conceptnet', 'wikidata']:

    edges_file='../output_v%s/%s/edges_v%s.csv' % (VERSION, dataset, VERSION)
    output_dir='../output_v%s/emb_data/%s' % (VERSION, dataset)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rows=[]
    with open(edges_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        spamreader.__next__()
        for row in spamreader:
            my_row=row[:3]
            rows.append(my_row)

    random.shuffle(rows)  # randomly shuffles the ordering of filenames

    write_data(rows, '%s/train.csv' % output_dir)

    split_1 = int(0.1 * len(rows))
    split_2 = int(0.2 * len(rows))
    dev_data = rows[:split_1]
    test_data = rows[split_1:split_2]

    write_data(dev_data, '%s/dev.csv' % output_dir)
    write_data(test_data, '%s/test.csv' % output_dir)



