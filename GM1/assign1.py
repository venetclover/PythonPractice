import numpy
import csv
import itertools
from copy import copy, deepcopy
from node import node
from bnet import bnet

def main():
	net = constructNet()	#construct the given bayes net
#	net = constructNewNet()	#construct our own net
#	mat = readFiles('data-train-1.txt')	
#	likeliEst(mat, net)	#find the distribution(CPT)
#	printCPTTex(net)
#	ans = query(net, net.nodes[0], [-1,1,4,1,1,1,1,1,1])
#	ans = query(net, net.nodes[3], [-1,1,1,-1,2,1,2,2,1])
#	print ans
#	trainingNet(net)
	learningNet(net)	#train the Bayes net and estimate accuracy

def printCPT(name, pos, parents, cpt, nodes):
	header = ['P('+name+')', name]
	layout = ['c','c']
	for p in parents:
		header.append(p.name)
		layout.append('c')
	
	print '===========latex start=========='
	print '\\begin{tabular}{|'+'|'.join(layout)+'|}\\hline'
	
	print ' & '.join(header)+'\\\\ \\hline \\hline'
	for row in cpt:
		tRow = []
		tRow.append(str(row[1]))
		for ind, i in enumerate(row[0]):
			if i != -1 and ind != pos:
				tRow.append(nodes[ind].valueMap[i])
			elif ind == pos:
				tRow.insert(1, nodes[ind].valueMap[i])
		print ' & '.join(tRow) + '\\\\ \\hline'
	print '\end{tabular}'

def printCPTTex(net):
	nodeList = [1, 7, 3, 8]
	for i in nodeList:
		name = net.nodes[i].name
		parents = net.nodes[i].parent
		printCPT(name, i, parents, net.nodes[i].cpt, net.nodes)

#find P(A|pa(A))
#-1 in CPT means this element is not a parent of A, 
#so we keep it
#there will be only one row match filterList
def lookupCPT(cpt, filterList):
	for row in cpt:
		for i, val in enumerate(row[0]):
			get = 1
			if val != -1 and val != filterList[i]:
				get = 0
				break
		if get == 1:
			return row[1]

def margin(targValue, targPos, table):
	#pos is a list of magined element
	probList = [0] * len(targValue)
	for row in table:
		for i, val in enumerate(targValue):
			if targPos != -1 and row[0][targPos] == val:
				probList[i] += row[1]
			elif targPos == -1:
				probList[i] += row[1]
	return probList

def query(net, target, given):
	#Find all possible combinations, including target
	#Same func. as findComs, but different implementation
	allGivenList = [given[:]]
	targPos = net.dataMap[target]

	for i in range(9):
		tList = []
		for row in allGivenList[:]:
			for k, val in enumerate(row):
				if val == -1:
					valList = net.nodes[k].value
					for var in valList:
						row[k] = var
						copyR = deepcopy(row)
						tList.append(copyR)
		if len(tList) > 0:
			allGivenList = deepcopy(tList)
	
	#Loop over the list
	lookupResults = []
	for row in allGivenList:
		prob = 1
		#find all prob. value for all combination
		for i, var in enumerate(row):
			print i, lookupCPT(net.nodes[i].cpt, row)
			prob *= lookupCPT(net.nodes[i].cpt, row)
		lookupResults.append([row, prob])
		print row, prob

	#Find the factor that need marginalize
	margList = []
	for i, val in enumerate(given):
		if i != targPos and val == -1:
			margList.append(i)

	#Marginal list of numerator
	num = margin(target.value, targPos, lookupResults)

	#Marginal list of denumerator, add variable
	denum = margin([-1], -1, lookupResults)

	qResult = []
	for i, val in enumerate(target.value):
		if denum[0] != 0:
			qResult.append([val, round(num[i]/denum[0], 6)])
		else: 
			qResult.append([val, 0])
#		print target.name, val, round(num[i]/denum[0], 4)
		
	return qResult

def likeliEst(mat, net):
	for n in net.nodes:
		constructCPT(mat, net, n)
#		print 'node: ', n.name
#		print n.cpt

def findCom(parents):
	coms = []
	if len(parents) > 1:
		tNode = parents.pop(0)
		tempMat = findCom(parents)
		coms = [[x]+y for x in tNode.value for y in tempMat]
		return coms
	else:
		for paNode in parents:
			for val in paNode.value:
				coms.append([val])
		return coms

def count(mat, filt):
	total = 0
	for d in mat:
		for i in range(len(filt)):
			take = 1
			if d[i] != filt[i] and filt[i] != -1:
				take = 0
				break
		total = total+take
		
	return total

def constructCPT(mat, net, targ):
	coms = findCom(targ.parent[:])	
	
	if len(coms) == 0 and len(targ.parent) == 0:
		coms.append([-1]*len(net.nodes))

	cpt = []
	for row in coms:
		filterlist = [-1]*len(net.nodes)
		Tpa = targ.parent[:]
		if len(Tpa) > 0:
			for i, e in enumerate(row):
				filterlist[net.dataMap[Tpa[i]]] = e	
		
		givenT = count(mat, filterlist)

		for val in targ.value:
			filterlist[net.dataMap[targ]] = val
			n = count(mat, filterlist)
			if givenT != 0:
				p = round(float(n)/float(givenT), 6)
			else:
				p = 0
#			print 'n=', n, 'num', givenT
#			print 'v=', val, 'p=', p, '||', filterlist
			cpt.append([filterlist[:], p])

	targ.cpt = cpt

def classify(net, target, dataRow):	
	#change data's format to fit query()
	pos = net.dataMap[target]
	trueClass = dataRow[pos]
	cData = deepcopy(dataRow)
	cData[pos] = -1
	resultVals = query(net, target, cData)
	
	prob = 0
	predClass = -1
	for option in resultVals[:]:
		if option[1] > prob:
			prob = option[1]
			predClass = option[0]
#	print 'result1: ', result1, 'result2: ', result2
#	print 'pred: ', result, 'true val: ', data[targPos]
	return True if (trueClass == predClass) else False

#for distribution of 6.b, but there are too many values
def trainingNet(net):
	mat = []	
	for i in range(5):
		tmat = readFiles('data-train-'+str(i+1)+'.txt')
		mat.extend(tmat)

	likeliEst(mat, net)
	allGivenList = [[-1,-1,-1,-1,-1,-1,-1,-1,-1]]
	
	for i in range(9):
		tList = []
		for row in allGivenList[:]:
			if row[i] == -1 and i != 8 :
				valLst = net.nodes[i].value
				for nVal in valLst:
					row[i] = nVal
					tList.append(row[:])
			else:
				tList.append(row[:])
		allGivenList = tList[:]
	
	for row in allGivenList:
		query(net, net.nodes[8], row)


def learningNet(net):
	probList1 = [0.0] * 5	
	for i in range(5):
		mat = readFiles('data-train-'+str(i+1)+'.txt')

		#construct all CPTs
		likeliEst(mat, net)
		probList2 = [0.0] * 5
			
		data = readFiles('data-test-'+str(i+1)+'.txt')
		accClassify = 0
		for row in data:
			if(classify(net, net.nodes[8], row) == True):
				accClassify += 1
		accClassifyRate = round(float(accClassify)/float(len(data)),6)
		
		probList1[i] = accClassifyRate
	
	b = numpy.array(probList1, float)
	print 'total', probList1, 'mean=', b.mean(), 'standard dev=', b.std()
	return 0

def readFiles(filename):
	fileDir = 'Data/'
	dMat=numpy.loadtxt(open(fileDir+filename,"rb"),delimiter=',',skiprows=0)
	return dMat

def constructNewNet():

	net = bnet()

	a = node("a")
	ch = node("ch")
	g = node("g")
	bp = node("bp")
	hd = node("hd")
	cp = node("cp")
	eia = node("eia")
	ecg = node("ecg")
	hr = node("hr")

	a.setVal([1,2,3])
	g.setVal([1,2])
	cp.setVal([1,2,3,4])
	bp.setVal([1,2])
	ch.setVal([1,2])
	ecg.setVal([1,2])
	hr.setVal([1,2])
	eia.setVal([1,2])
	hd.setVal([1,2])

	ch.addParent(a)
	bp.addParent(a)
	hd.addParent(a)
	hd.addParent(g)
	hd.addParent(bp)
	hd.addParent(ch)
	hd.addParent(hr)
	cp.addParent(hd)
	eia.addParent(cp)
	eia.addParent(ecg)
	eia.addParent(hd)
	ecg.addParent(cp)
	ecg.addParent(hd)
	hr.addParent(a)

	net.addNode(a)
	net.addNode(g)
	net.addNode(cp)
	net.addNode(bp)
	net.addNode(ch)
	net.addNode(ecg)
	net.addNode(hr)
	net.addNode(eia)
	net.addNode(hd)

	#{attribute:order}
	net.dataMap = {
		a:0,
		g:1,
		cp:2,
		bp:3,
		ch:4,
		ecg:5,
		hr:6,
		eia:7,
		hd:8
		}

	return net

def constructNet():

	net = bnet()

	a = node("a")
	ch = node("ch")
	g = node("g")
	bp = node("bp")
	hd = node("hd")
	cp = node("cp")
	eia = node("eia")
	ecg = node("ecg")
	hr = node("hr")

	a.setVal([1,2,3])
	g.setVal([1,2])
	cp.setVal([1,2,3,4])
	bp.setVal([1,2])
	ch.setVal([1,2])
	ecg.setVal([1,2])
	hr.setVal([1,2])
	eia.setVal([1,2])
	hd.setVal([1,2])

	valMap1 = {1:'<45', 2:'45-55', 3:'>=55'}
	a.setValMap(valMap1)
	valMap2 = {1:'Female', 2:'Male'}
	g.setValMap(valMap2)
	valMap3 = {1:'Typical', 2:'Atypical', 3:'Non-Anginal', 4:'None'}
	cp.setValMap(valMap3)
	valMap4 = {1:'Normal', 2:'Abnormal'}
	ecg.setValMap(valMap4)
	valMap5 = {1:'Low', 2:'High'}
	bp.setValMap(valMap5)
	ch.setValMap(valMap5)
	hr.setValMap(valMap5)
	valMap6 = {1:'No', 2:'Yes'}
	eia.setValMap(valMap6)
	hd.setValMap(valMap6)

	ch.addParent(a)
	bp.addParent(a)
	bp.addParent(g)
	hd.addParent(ch)
	hd.addParent(bp)
	hd.addParent(g)
	cp.addParent(hd)
	eia.addParent(hd)
	ecg.addParent(hd)
	hr.addParent(hd)

	net.addNode(a)
	net.addNode(g)
	net.addNode(cp)
	net.addNode(bp)
	net.addNode(ch)
	net.addNode(ecg)
	net.addNode(hr)
	net.addNode(eia)
	net.addNode(hd)

	#{attribute:order}
	net.dataMap = {
		a:0,
		g:1,
		cp:2,
		bp:3,
		ch:4,
		ecg:5,
		hr:6,
		eia:7,
		hd:8
		}
	return net

if __name__=="__main__":
	main();
