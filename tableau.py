'''
This code takes a regular MaxEntGrammarTool input files and convert it to the format that can be fed to the Power Learner.
'''
import numpy as np

def file_open (filepath, div='\t'):
	with open(filepath, 'r', encoding='utf-8') as f:
	    readobj=f.readlines()
	    del readobj[0] # remove one row of constraint names
	    readobj = [x.strip('\n').split(div) for x in readobj]
	return readobj


def cons_extractor (readobj): # extracts constraint names and save in a list
	constraints = [x for x in readobj[0] if len(x)>0]
	weights = [1.0 for i in range(len(constraints))]
	powers = [1.0 for i in range(len(constraints))]
	return constraints, np.array(powers), np.array(weights)


def fv_extractor (tableau):
	all_frequency = []
	all_violations = []

	for line in tableau[1:]:
		# print(line)
		if len(line[0]) > 0:
			# start a new sublist
			try:
				all_frequency.append(freq)
				# print(all_frequency)
				all_violations.append(violations)
				# print(all_violations)
			except NameError: freq, violations = None, None
			freq = [float(line[2])] #[80]
			# print(freq)
			violations = [[int(x) if x!='' else int(0) for x in line[3:]]] #[[0, 0]]
			# print(violations)
		else:
			freq.append(float(line[2])) # [80, 20]
			# print(freq)
			violations.extend([[int(x) if x!='' else int(0) for x in line[3:]]]) #[[0, 0], [1, 0]]
			# print(violations)
	all_frequency.append(freq)
	all_probability = []
	for pair in all_frequency:
		s = sum(pair)
		prob = []
		for i in range(len(pair)):
			prob.append(pair[i]/sum(pair))
		all_probability.append(prob)
	all_violations.append(violations)
	print(all_probability)
	return np.array(all_probability), np.array(all_violations)


