import random

def qsort(a):
	"""docstring for qsort"""
	if len(a) <= 1:
		return a
	else:
		pivot = a[len(a)-1]
		less = []
		greater = []
		print a[0:len(a)-1]
		for e in a[0:len(a)-1]:
			if e < pivot:
				less.append(e)
			else:
				greater.append(e)
		return qsort(less)+[pivot]+qsort(greater)

x = [ int(x) for x in range(100)]
y = random.sample(x, 10)
print y
print qsort(y)
