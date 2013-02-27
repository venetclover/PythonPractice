import random as rand

nPig = 8
portList = []

def initialize():
	"""docstring for initialize"""
#	try: 
#		nPig = int(raw_input("Enter number of pig: "))
#		print nPig
#	except TypeError:
#		initialize()
     
	ports = [x+3000 for x in range(100)]
	portList = rand.sample(ports, nPig)
	print portList
	for i in range(nPig):
		pig = new pigInst(i+1, port)

	#start the connection
	for i in range(nPig):	
	#### In formal method, the connection should be random
	#	randConnect = rand.sample(portList, 3)

	##### Build a chain structured network for test
		

def main():
	"""docstring for main"""
	initialize()

if __name__ == '__main__':
	main()
