import numpy as np


def n_gram_train(label_file, n, example_nums=0):
	label_file = open(label_file)
	if example_nums == 0:
		dic_all = {}
		lines = label_file.readline()
		while lines:
			code = lines.split(',')[1]
			token = ['818'] + code.split(' ')[:-1] + ['819']
			target_key = ''
			if len(token) > n-1:  # sentence is long enough
				for i in range(n):
					target_key = target_key + token[i] + ' '
				target_key = target_key[:-1]
				for i in range(len(token)-n):
					if target_key not in dic_all:
						dic_all[target_key] = 1
					else:
						dic_all[target_key] += 1
					target_key = target_key[target_key.index(' ')+1:]+' ' + token[i+n]
				if target_key not in dic_all:
					dic_all[target_key] = 1
				else:
					dic_all[target_key] += 1
				lines = label_file.readline()
		label_file.close()
		return dic_all
	else:
		return n_gram_train_helper(label_file, n, example_nums)


def n_gram_train_helper(label_file, n, example_nums):
	dic_all = {}
	lines = label_file.readline()
	for trys in range(example_nums):
		code = lines.split(',')[1]
		token = ['818'] + code.split(' ')[:-1] + ['819']
		target_key = ''
		if len(token) > n-1:
			for i in range(n):
				target_key = target_key + token[i] + ' '
			target_key = target_key[:-1]
			for i in range(len(token)-n):
				if target_key not in dic_all:
					dic_all[target_key] = 1
				else:
					dic_all[target_key] +=1
				target_key = target_key[target_key.index(' ')+1:]+' ' + token[i+n]
			if target_key not in dic_all:
				dic_all[target_key] = 1
			else:
				dic_all[target_key] +=1
			lines = label_file.readline()
		label_file.close()
		return dic_all


def n_gram_infer(n_gram, qry):
	# infer p(x_n | x_1 ... x_n-1)
	# n_gram: dict, key: '111 222 333 444' -> val: 1
	# qry: a numpy array of size(1,n-1)
	# output p: a numpy array of size(w,1)

	# erase zero padding
	qry = qry[np.where(qry != 0)]

	# merge qry to qry_str
	qry_str = ""
	for i in range(qry.size):
		qry_str += str(qry[i])
		qry_str += " "

	# count occurrence
	p = np.zeros(819)
	for i in range(819):
		cnt = n_gram.get(qry_str + str(i+1), 0)
		p[i] = cnt
		if cnt != 0: print(qry_str + str(i+1) + " / " + str(cnt))

	# normalize
	if p.sum(0) != 0:
		p = p/p.sum(0)

	# smooth
	p[np.where(p == 0)] = 1e-6
	return p


# examples
print(n_gram_train(label_file="train_label", n=3, example_nums=1))
print(n_gram_train(label_file="train_label", n=4, example_nums=1))
print(n_gram_train(label_file="train_label", n=6, example_nums=30))

qry = np.array([519, 662, 749])
n_gram = n_gram_train(label_file="train_label", n=4)
p = n_gram_infer(n_gram, qry)
print(p)
