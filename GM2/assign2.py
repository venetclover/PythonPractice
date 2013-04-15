import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy.optimize import fmin_cg
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin

#out = open('/home/venet/Dropbox/Graphical Models/Assignment2A/output.txt', 'w')
out = open('output.txt', 'w')
t_para_ini = np.matrix(np.genfromtxt("model/transition-params.txt"))
s_para_ini = np.matrix(np.genfromtxt("model/feature-params.txt"))
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
	print 'deriv=', np.array([dfdx0, dfdx1])
	return np.array([dfdx0, dfdx1])

def obj_func(x):
	sign = 1.0
	x = np.reshape(x, (2, 1))
	x0= x[0,0]
	x1= x[1,0]
	print x
	y = ((1-x0)**2 + 100*(x1-x0**2) ** 2)
	return sign * y

def marginal_w(beta, logZ, i, t):
	"""docstring for marginal_w"""
	if t == 0:
		t_beta = np.sum(np.exp(beta), axis=1)
	else:
		t_beta = np.sum(np.exp(beta), axis=0)
	t_beta = np.reshape(t_beta, (1, 10))
	return t_beta[0,i]/np.exp(logZ)

ansFile = 'data/train_words.txt'

def obj_ws_deriv(x, *args):
	sign = 1.0
	N = args[0]
	#feature-params: 3310, transition-params:100
	x_s = x[0:3210]
	x_t = x[3210:3310]
	x_s = np.reshape(x_s, (10, 321))
	x_t = np.reshape(x_t, (10,10))

	x_deriv = np.zeros_like(x)
	for i in range(N):
		#find value based on real answer
		m_word = np.matrix(np.genfromtxt("data/train_img"+str(i+1)+".txt"))
		real_word = findAns(ansFile, i)
		w_len = len(real_word)
			
		#find phi
		potent = x_s * m_word.transpose()
		phi_list = []
		for cn in range(w_len-1):
			if cn == w_len-2:
				phi_list.append(phi(x_t, potent, [cn, cn+1]))
			else:
				phi_list.append(phi(x_t, potent, [cn]))
		
		#find clique
		cliq_list = []
		for cn in range(w_len-1):
			cliq_list.append([cn, cn+1])

		#find_sender
		s_list = []
		for cn in range(1, w_len-1):
			s_list.append(cn)

		beta_t = sumProductMPLS(cliq_list, s_list, phi_list)
		logZ = logSumExp(beta_t[0])
		
		tmp_x_s = np.zeros(shape=(10,321))
		tmp_x_t = np.zeros(shape=(10,10))
		for j, c in enumerate(real_word):
			if j == w_len-1:
				pc = marginal_w(beta_t[j-1], logZ, char[c], 1)
			else:
				pc = marginal_w(beta_t[j], logZ, char[c], 0)
			tmp_x_s[char[c],:] = 1 - pc*m_word[j,:]

			if j < w_len-1:
				c1 = real_word[j+1]
				p1 = char[c]
				p2 = char[c1]
				t_beta = beta_t[j]
				pw = np.exp(t_beta[p1,p2])/np.exp(logZ)
				tmp_x_t[p1, p2] = 1 - pw

		tmp_x_s = np.reshape(tmp_x_s, (3210))
		tmp_x_t = np.reshape(tmp_x_t, (100))
		x_deriv += np.array(np.concatenate((tmp_x_s, tmp_x_t), axis=1))
	y1 = (1/float(N)) * x_deriv
#	print 'y1', [k for k in y1 if k>1]
	return (-1) * y1

def obj_ws_func(x, *args):
	"""docstring for obj_ws_func"""
	sign = 1.0
	N = args[0]
	#feature-params: 3310, transition-params:100
	x_s = x[0:3210]
	x_t = x[3210:3310]
	x_s = np.reshape(x_s, (10, 321))
	x_t = np.reshape(x_t, (10,10))

	t_diff = 0
	for i in range(N):
		tmp_v = 0
		logZ = 0
		#find value based on real answer
		m_t_word = np.matrix(np.genfromtxt("data/train_img"+str(i+1)+".txt")).transpose()
		real_word = findAns(ansFile, i)
		w_len = len(real_word)
		for j, c in enumerate(real_word):
			v = x_s[char[c],:] * m_t_word[:,j]
			tmp_v += v[0,0]
		
		for j in range(w_len-1):
			c1 = real_word[j]
			c2 = real_word[j+1]
			tmp_v += x_t[char[c1],char[c2]]
			
		#find phi
		potent = x_s * m_t_word
		phi_list = []
		for cn in range(w_len-1):
			if cn == w_len-2:
				phi_list.append(phi(x_t, potent, [cn, cn+1]))
			else:
				phi_list.append(phi(x_t, potent, [cn]))
		
		#find clique
		cliq_list = []
		for cn in range(w_len-1):
			cliq_list.append([cn, cn+1])

		#find_sender
		s_list = []
		for cn in range(1, w_len-1):
			s_list.append(cn)

		beta_t = sumProductMPLS(cliq_list, s_list, phi_list)
		logZ = logSumExp(beta_t[0])
		t_diff += (tmp_v - logZ)

#	print tmp_v, logZ,  (1/float(N))*(t_diff)
	y = (1/float(N))*(t_diff)
	return  (-1) * y

def cal_accuracy(s_para, t_para):
	"""docstring for obj_ws_func"""
	predChars = ''
	trueChars = ''
	for i in range(200):
		#find value based on real answer
		m_t_word = np.matrix(np.genfromtxt("data/test_img"+str(i+1)+".txt")).transpose()
		r_word = findAns('data/test_words.txt', i)
		w_len = len(r_word)

		#find phi
		potent = s_para * m_t_word
		phi_list = []
		for cn in range(w_len-1):
			if cn == w_len-2:
				phi_list.append(phi(t_para, potent, [cn, cn+1]))
			else:
				phi_list.append(phi(t_para, potent, [cn]))
		
		#find clique
		cliq_list = []
		for cn in range(w_len-1):
			cliq_list.append([cn, cn+1])

		#find_sender
		s_list = []
		for cn in range(1, w_len-1):
			s_list.append(cn)

		beta_t = sumProductMPLS(cliq_list, s_list, phi_list)
		pred_word = predWord(beta_t, i)
	#	print >>out, pred_word, '<->', true_word
		predChars += pred_word
		trueChars += r_word

	correctChar = 0
	for cind in range(len(predChars)):
		predC = predChars[cind]
		trueC = trueChars[cind]
		if predC == trueC:
			correctChar += 1

	print >>out, 'accuracy', round(float(correctChar)/len(predChars), 5)
	return round(float(correctChar)/len(predChars), 5)
#	print 'accuracy', round(float(correctChar)/len(predChars), 3)

out_s_params = open('train_s_params.txt', 'w')
out_t_params = open('train_t_params.txt', 'w')

def opt():
	"""docstring for opt"""
	s_para_ini = np.genfromtxt("model/feature-params.txt")
	t_para_ini = np.genfromtxt("model/transition-params.txt")

	x1 = np.reshape(s_para_ini, (1, 3210))
	x2 = np.reshape(t_para_ini, (1, 100))
	x = np.array(np.concatenate((x1, x2), axis=1))
	x0 = x[0]
#	x0 = [0] * 3310
	s_para = []
	t_para = []
	for i in [80]:
#	for i in [50, 100, 150, 200, 250, 300, 350, 400]:
		N = i
#		obj_ws_deriv(x0, (N,))
		opt = fmin_l_bfgs_b(obj_ws_func, x0, fprime=obj_ws_deriv, args=(N,))
#		opt = fmin_l_bfgs_b(obj_ws_func, x0, args=(N,), approx_grad=True)
		xopt = opt[0]
		print opt
		print xopt
		s_para = np.reshape(xopt[0:3210], (10, 321))
		t_para = np.reshape(xopt[3210:3310], (10, 10))
		print >>out_s_params, [s_para[x, y] for x in range(10) for y in range(321)]
		print >>out_s_params, '\n', '\n', '\n'
		print >>out_t_params, [t_para[x, y] for x in range(10) for y in range(10)]
		print >>out_t_params, '\n', '\n', '\n'

		print 'training case = ', i, 'accuracy = ', cal_accuracy(s_para, t_para)

	#print transit params
	img_pix = t_para
	charA = np.asarray(['e', 't', 'a', 'i', 'n', 'o', 's', 'h', 'r', 'd'])
	imgplot = plt.imshow(img_pix, interpolation='nearest')
	plt.xticks(range(10), charA)
	plt.yticks(range(10), charA)
	plt.colorbar()
	plt.savefig('t_params.png')
#	plt.show()

	#print single params
	for ind, row in enumerate(s_para):
		plt.clf()
		img_pix = np.reshape(row[1:321], (20, 16))
		imgplot = plt.imshow(img_pix, cmap=cm.Greys_r, interpolation='nearest')
		img_n = 's_params'+str(ind)+'.png'
		plt.savefig(img_n)
#		plt.show()

def q4():
	"""docstring for q4"""
	x = [10,10]
	xopt = fmin_l_bfgs_b(obj_func, x, fprime=obj_deriv)
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
	lst1 = []
	for i in range(10):
		m = np.max(table[i,:])
		t_m = 0
		for j in range(10):
			t_m += np.exp(table[i,j]-m)
		lst1.append(m+np.log(t_m))

	m = np.max(lst1)
	t_m = 0
	for e in lst1:
		t_m += np.exp(e-m)

	return m+np.log(t_m)
	
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
	for i in range(1):
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
#		print '----', [np.sum(b) for b in beta_t]
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

	print >>out, 'accuracy', round(float(correctChar)/len(predChars), 5)
#	print 'accuracy', round(float(correctChar)/len(predChars), 3)


def main():
#	q1()
#	q2()
#	q4()
	opt()

if __name__ == '__main__':
	main()
