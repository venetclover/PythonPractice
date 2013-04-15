import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

wp = 11.0
wl = 11500.0

def get_data(filename):	
	return np.loadtxt(filename)

def find_neighbors(posi, posj):
	"""docstring for find_neighbors"""
	neighbors = []
	pivot = [-1, 1]
	for i in pivot:
		if posi+i>=0 and posi+i<100:
			neighbors.append((posi+i, posj))	
	for j in pivot:
		if posj+j>=0 and posj+j<100:
			neighbors.append((posi, posj+j))
	return neighbors

def gibbs_b_iter(mn, w, h):
	y_matrix = mn
	for i in range(w):
		for j in range(h):
			neighbors = find_neighbors(i, j)
			#find numerator and t_denom of probability
			#t_denom records [y=0]
			#therefore denominator = t_denom+numerator
			numerator = 0.0
			t_denom = 0.0
			for (k,l) in neighbors:
				if y_matrix[k,l] == 1:
					numerator += wp
				elif y_matrix[k,l] == 0:
					t_denom += wp

			if mn[i,j] == 1:
				numerator += wl
			elif mn[i,j] == 0:
				t_denom += wl
			
			prob_yij = np.exp(numerator)/(np.exp(numerator)+np.exp(t_denom))
			alpha = np.random.random_sample()
			if alpha < prob_yij:
				y_matrix[i,j] = 1
			else:
				y_matrix[i,j] = 0
	
	return y_matrix

def gibbs_c_iter(mn, w, h):
	"""docstring for gibbs_c_iter"""
	y_t = mn
	for i in range(w):
		for j in range(h):
			neighbors = find_neighbors(i, j)
			nij = len(neighbors)

			t_wp_ykl = 0
			for (k,l) in neighbors:
				t_wp_ykl += wp*y_t[k,l]
			
			var = 1/(2*(wp*nij+wl))
			mu = 1/(wp*nij+wl)*(wl*mn[i,j]+t_wp_ykl)
			
			y_t[i,j] = np.random.normal(mu, np.sqrt(var))

	return y_t

def gibbs_c_iter_p(mn, w, h):
	"""docstring for gibbs_c_iter"""
	y_t = mn
	wp_p = np.zeros((100,100), dtype=float)
	for i in range(w):
		for j in range(h):
			neighbors = find_neighbors(i, j)
			nij = len(neighbors)

			s_wp_ijkl = 0
			t_wp_ykl = 0
			for (k,l) in neighbors:
				wp_ijkl = wp/(0.01+(mn[i,j]-mn[k,l])**2)
				s_wp_ijkl += wp_ijkl
				t_wp_ykl += wp_ijkl*y_t[k,l]
			
			var = 1/(2*(s_wp_ijkl+wl))
			mu = 1/(s_wp_ijkl+wl)*(wl*mn[i,j]+t_wp_ykl)
			
			y_t[i,j] = np.random.normal(mu, np.sqrt(var))

	return y_t

def gibbs_b(m, mn):
	w, h = m.shape
	y = mn
	y_matrix = np.zeros((w,h), dtype=float)
	t_list = [i*5 for i in range(1,20)]
	mae = []
	for t in range(1,101):
		y = gibbs_b_iter(y, w, h)
		y_matrix += y
		print y_matrix
		if t in t_list:
			y1 = y_matrix/float(t)
			t_mae = 0.0
			for i in range(w):
				for j in range(h):
					t_mae += np.absolute(y1[i,j]-m[i,j])
			mae.append((1.0/float(w*h))*t_mae)
			print 't=', t, 'MAE', mae

	fig = plt.figure()
	ax = fig.add_subplot(111, xlabel='T', ylabel='MAE')
#	ax.set_xlim(8, 1088)
#	ax.set_ylim(0.03, 0.04)
	ax.plot(t_list, mae, 'o-')
	plt.show()
	plt.clf()
			
	img_pix = np.round(y1)
	imgplot = plt.imshow(img_pix, interpolation='nearest', cmap=cm.Greys_r)
	img_name = 'p'+str(wp)+'_'+str(wl)+'_'+str(mae[len(mae)-1])+'.png'
	plt.savefig(img_name)
	plt.show()

def gibbs_c(mn):
	t = 100
	w, h = mn.shape
	y = mn
	y_matrix = np.zeros((w,h), dtype=float)
	for t1 in range(t):
#		y = gibbs_c_iter(y, w, h)
		y = gibbs_c_iter_p(y, w, h)
		y_matrix += y
	
	print y_matrix/float(t)
	return (1.0/float(t))*y_matrix

def q1():
	"""docstring for q1"""
	m1 = get_data('Data/stripes.txt')
	m1n = get_data('Data/stripes-noise.txt')

	gibbs_b(m1, m1n)

def q2():
	m2 = get_data('Data/swirl.txt')
	m2n = get_data('Data/swirl-noise.txt')
	y = gibbs_c(m2n)
	w, h = y.shape
	t_mae = 0.0
	for i in range(w):
		for j in range(h):
			t_mae += np.absolute(y[i,j]-m2[i,j])
	
	print t_mae
	mae = (1.0/float(w*h))*t_mae
	print 'MAE', mae

	img_pix = y
	imgplot = plt.imshow(img_pix, interpolation='nearest', cmap=cm.Greys_r)
	img_name = 'qp'+str(wp)+'_'+str(wl)+'_'+str(mae)+'.png'
	plt.savefig(img_name)

def main():
	"""docstring for main"""
#	q1()
	q2()

if __name__ == '__main__':
	main()
