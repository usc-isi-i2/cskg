

def get_labels(data):
    if '|' in data:
        return data.split('|')
    else:
        return [data]
            

def analyze_lengths(filename):
    longest_label=''
    longest_len=-1
    with open(filename, 'r') as f:
        header=next(f)
        for line in f:
            line_data=line.split('\t')
            node1_labels=get_labels(line_data[4])
            node2_labels=get_labels(line_data[5])
            #print(node1_labels, node2_labels)
            for label in node1_labels+node2_labels:
                if not len(label):
                    print(line)
                if len(label)>longest_len:
                    longest_len=len(label)
                    longest_label=label
                #if len(label)>100:
                #    print(label, len(label), line_data[9])
            #input('c')


    print(longest_label, longest_len)



if __name__=="__main__":
    filename="../output/cskg_compact.tsv"
    analyze_lengths(filename)
