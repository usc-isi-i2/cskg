import sys
import csv
csv.field_size_limit(sys.maxsize)

def extract_node_data(node_file):
	node2label={}
	node2pos={}
	with open(node_file, 'r') as f:
		reader = csv.reader(f, delimiter='\t')
		header=next(reader, None)
		for fs in reader:
			if fs and len(fs)==6:
				node_id=fs[0].strip()
				if not fs[1].strip(): continue
				node2label[node_id]=fs[1].strip()
				pos=fs[3].strip()
				if ',' in pos:
					pos=''
				node2pos[node_id]=pos
	print('node index ready')
	return node2label, node2pos

def normalize_relation(r):
	if r.startswith('/r/'):
		return r[3:]
	else:
		return ':'.join(r.split(':')[1:])
