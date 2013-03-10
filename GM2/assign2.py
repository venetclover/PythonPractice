import numpy as np
from scipy.optimize import fmin_bfgs

#out = open('/home/venet/Dropbox/Graphical Models/Assignment2A/output.txt', 'w')
out = open('output.txt', 'w')
char = {'e':0, 't':1, 'a':2, 'i':3, 'n':4, 'o':5, 's':6, 'h':7, 'r':8, 'd':9}
invchar = dict([(v, k) for (k, v) in char.items()])

def findAns(filename, i):
	"""filename contain ans"""
	"""for verifying the algorithm"""
	with open(filename, 'r') as fr:
		lines = fr.readlines()
		return lines[i].strip()

def log_partition(table):
	Z = 0
	for row in table:
		Z += row[1]

	return np.log(Z)

def joint_table(matr, t_matr):
	"""docstring for logPartition"""
	size = np.shape(matr)
	nums = [[x] for x in range(10)]
	
	#loop over the column(characters)
	combine = nums
	for i in range(size[1]-1):
		combine = [x+y for x in combine for y in nums]

	table = []
	for row in combine:
		t_sum = 0
		for j, label_p in enumerate(row):
			t_sum += matr[label_p, j]
		
		for i in range(0, len(row)-1):
			t_sum += t_matr[row[i],row[i+1]]

		table.append([row, np.exp(t_sum)])

	return table

def marginal(table, Z):
	"""docstring for marginal"""
	size = len(table[0][0])
	#i is the position of the word
	#j is the character
	pair = [(i, j) for i in range(size) for j in range(10)]
	prob_table = np.zeros((10, size))
	marg_table = []
	for p in pair:
		t_sum = 0
		i = p[0]
		j = p[1]
		for row in table:
			com_r = row[0]
			factor = row[1]
			if com_r[i] == j:
				t_sum += factor
#		print 'p', p, 'factor', t_sum/Z
		marg_table.append([p, np.exp(np.log(t_sum)-Z)])
		prob_table[j][i] = np.exp(np.log(t_sum)-Z)
	return prob_table


def q1():
	"""read data from 2 files. multiply them"""
	data = np.matrix(np.genfromtxt("data/test_img1.txt"))
	para = np.matrix(np.genfromtxt("model/feature-params.txt"))
	t_para = np.matrix(np.genfromtxt("model/transition-params.txt"))

	#q1_1
	potential = para*data.transpose()
	print >> out, '1.1\n', potential

	#q1_2
	word = findAns('data/test_words.txt', 0)

	sumPot = 0
	for i, c in enumerate(word):
		sumPot += potential[char[c], i]
	print >> out, '1.2\n', sumPot

	#q1_3, q1_4, q1_5
	print >> out, '1.3'
	for i in range(3):
		filename = "data/test_img"+str(i+1)+".txt"
		data = np.matrix(np.genfromtxt(filename))
		potential = para*data.transpose()
		
		table = joint_table(potential, t_para)
		max_p = 0
		for row in table:
			if row[1] > max_p:
				max_p = row[1]
				max_word = row[0]
		
		word = [ invchar[ic] for ic in max_word ]

		Z = log_partition(table)
		print >> out, i+1, 'Z=', Z
		print >> out, ''.join(word), np.exp(np.log(max_p)-Z)
	
		if i == 0:
			words_marg = marginal(table, Z)
			print >> out, '1.5\n', words_marg

def obj_deriv(x):
	sign = 1.0
	dfdx0 = sign*(2 - 2*x[0] + 400*x[0]*x[1]-400*x[0]**3)
	dfdx1 = sign*(-200*x[1]+200*x[0]**2)
	return np.array([dfdx0, dfdx1])

def obj_func(x):
	sign = 1.0
	return sign * ((1-x[0])**2 + 100*(x[1]-x[0]**2) ** 2)

def q4():
	"""docstring for q4"""
	x = [10,10]
	xopt = fmin_bfgs(obj_func, x, fprime=obj_deriv)
	print xopt

def sumProductMPLS(cliqs, senders, lams):
	"""docstring for SumProductMPLS"""
	#lams is in log space
	dels_f = []
	for i in range(len(cliqs)-1):
		if  i > 0 and (x in senders for x in cliqs[i-1]):
			tt = np.exp(lams[i] + dels_f[i-1].transpose())
		else:
			tt = np.exp(lams[i])
		sum_t = np.sum(tt, axis=0)
		delta = np.log(sum_t)
		dels_f.append(delta)
	
	dels_b = []
	for i in range(len(cliqs)-1, 0, -1):
		if  i < len(cliqs)-1 and (x in senders for x in cliqs[i+1]):
			tt = np.exp(lams[i] + dels_b[0].transpose())
		else:
			tt = np.exp(lams[i])
		sum_t = np.sum(tt, axis=1)
		delta = np.log(sum_t)
#		print 'delta', i, delta
		dels_b.insert(0, delta)

	betas = []
	for i in range(len(cliqs)):
		add = lams[i]
		if i > 0:
			add += dels_f[i-1].transpose()
		if i < len(cliqs)-1:
			add += dels_b[i].transpose()
		betas.append(add)
	
#	print 'beta', betas[0]
	return betas

def logSumExp(table):
	#table is in log space
	logZ = []
	for tp in table:
		c = tp.max()
		size = np.shape(tp)
		temp_sum = 0
		for i in range(size[0]):
			for j in range(size[1]):
				temp_sum += np.exp(tp[i,j]-c)
		logZ.append(c + np.log(temp_sum))

	return logZ
	
def phi(t_para, n_potent, cond):
	#condition squence 
	#      must follow the order of position and contigious
	phi = t_para + n_potent[:,cond[0]]

	if len(cond) == 2:
		phi = phi + n_potent[:,cond[1]].transpose()
	
	return phi

def buildTable(combs, lambs, logZ):
	"""docstring for buildTable"""
	size = len(combs[0])
	table = []
	for row in combs:
		t_sum = 0
		for ind in range(size-1):	
			i = row[ind]
			j = row[ind+1]
			t_lamb = lambs[ind]
			t_sum += t_lamb[i,j]

		ans = np.exp(t_sum-logZ)
		table.append([row, ans])
	
	return table

def predWord(betas, wind):
	"""docstring for predWord"""
	wSet = []
	for beta in betas:
		Z = np.sum(beta)
		t_sum = np.sum(beta, axis=1)
		pos = np.argmax(t_sum/Z)
		wSet.append(pos)

	t_sum = np.sum(betas[len(betas)-1], axis=0)
	pos = np.argmax(t_sum/Z)
	wSet.append(pos)

	word = ''
	word = [word+invchar[x] for x in wSet]
	return ''.join(word)

def predWord1(betas):
	"""docstring for predWord"""
	print betas

	wSet = []
	
	t_sum = np.sum(betas[0], axis=1)
	pos = np.argmax(t_sum)
	wSet.append(pos)
	
	for beta in betas:
		t_sum = np.sum(beta, axis=0)
		pos = np.argmax(t_sum)
		wSet.append(pos)

	word = ''
	word = [word+invchar[x] for x in wSet]
	return ''.join(word)

def q2():
	"""docstring for q2"""
	data = np.matrix(np.genfromtxt("data/test_img1.txt"))
	para = np.matrix(np.genfromtxt("model/feature-params.txt"))
	t_para = np.matrix(np.genfromtxt("model/transition-params.txt"))
	n_potent = para*data.transpose()
	
	#q2_1 
	phi1 = phi(t_para, n_potent, [0])
	print >>out, 'phi1', phi1
	phi2 = phi(t_para, n_potent, [1])
	print >>out, 'phi2', phi2
	phi3 = phi(t_para, n_potent, [2, 3])
	print >>out, 'phi3', phi3
	
	#q2_2
	#\sigma1->2(Y_2)
	#sum over the columns(Y_1)
	delta1 = np.log(np.sum(np.exp(phi1), axis=0))
	print >>out, 'delta1', delta1

	#\sigma3->2(Y_3)
	#pass Y_1, so sum over the columns(Y_2)
	delta4 = np.log(np.sum(np.exp(phi3), axis=1))
	print >>out, 'delta4', delta4

	#\sigma2->1(Y_2) 
	#add 10*10 matrix and 1*10 matrix
	t_phi2 = phi2 + delta4.transpose()
#	print 'total', np.log(phi2[0,1]), delta4[1], t_phi2[0,1]
	#pass Y_2, so sum over the colums(Y_3)
	delta2 = np.log(np.sum(np.exp(t_phi2), axis=1))
	print >>out, 'delta2', delta2
	 
	#\sigma2->3(Y_3)
	#add 10*10 matrix and 10*1 matrix
	t_phi3 = phi2 + delta1.transpose()
	#pass Y_3, so sum over the row(Y_2)
	delta3 = np.log(np.sum(np.exp(t_phi3), axis=0))
	print >>out, 'delta3', delta3

	#q2_3
	#\beta(Y_1, Y_2)
	beta1 = phi1+delta2.transpose()
	print >>out, 'beta1', beta1

	#\beta(Y_2, Y_3)
	beta2 = phi2+delta1.transpose()+delta4.transpose()
	print >>out, 'beta2', beta2

	#\beta(Y_3, Y_4)
	beta3 = phi3+delta3.transpose()
	print >>out, 'beta3', beta3

	#q2_4
	beta1_t = beta1[0:2,0:2]
	Z = np.sum(beta1)
	print >>out, 'q2_4', beta1_t/Z

	beta2_t = beta2[0:2,0:2]
	Z = np.sum(beta2)
	print >>out, 'q2_4', beta2_t/Z

	beta3_t = beta3[0:2,0:2]
	Z = np.sum(beta3)
	print >>out, 'q2_4', beta3_t/Z

	#q2_5
	predChars = ''
	trueChars = ''
	correctChar = 0
	for i in range(200):
		data = np.matrix(np.genfromtxt("data/test_img"+str(i+1)+".txt"))
		n_potent = para*data.transpose()
		size = np.shape(data)
		w_len = size[0]
		
		#find phi
		phi_list = []
		for cn in range(w_len-1):
			if cn == w_len-2:
				phi_list.append(phi(t_para, n_potent, [cn, cn+1]))
			else:
				phi_list.append(phi(t_para, n_potent, [cn]))
		
		#find clique
		cliq_list = []
		for cn in range(w_len-1):
			cliq_list.append([cn, cn+1])
		
		#find sender
		s_list = []
		for cn in range(1, w_len-1):
			s_list.append(cn)

		beta_t = sumProductMPLS(cliq_list, s_list, phi_list)
#		print >>out, '----', beta_t
#		logZ_list = logSumExp(beta_t)
#		print 'logZ', logZ		

#		print logZ_list[0]
#		print 'pairwise: ', beta_t[0]/logZ_list[0]

		pred_word = predWord(beta_t, i)
		true_word = findAns('data/test_words.txt', i)
		print >>out, pred_word, '<->', true_word
		predChars += pred_word
		trueChars += true_word
	
	for cind in range(len(predChars)):
		predC = predChars[cind]
		trueC = trueChars[cind]
		if predC == trueC:
			correctChar += 1

	print >>out, 'accuracy', round(float(correctChar)/len(predChars), 3)
#	print 'accuracy', round(float(correctChar)/len(predChars), 3)

def main():
#	q1()
	q2()
#	q4()

if __name__ == '__main__':
	main()
