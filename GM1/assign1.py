import numpy
import csv
import itertools
from node import node
from bnet import bnet

def main():
	net = constructNet()
	mat = readFiles()
	likeliEst(mat, net)
'''	func = raw_input('solve #? ')
	func = {
		4:likeliEst,
		6:rate
		}
'''
	
def likeliEst(mat, net):
	for n in net.nodes:
		cpt = constructCPT(mat, net, n)


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

	for row in coms:
		filterlist = [-1]*len(net.nodes)
		Tpa = targ.parent[:]
		if len(Tpa) > 0:
			for i, e in enumerate(row):
				filterlist[net.dataMap[Tpa[i]]] = e	
		
		givenT = count(mat, filterlist)

		print targ.name
		for val in targ.value:
			filterlist[net.dataMap[targ]] = val
			n = count(mat, filterlist)
			p = float(n)/float(givenT)
			print 'n=', n, 'num', givenT
			print 'v=', val, 'p=', p, '||', filterlist 
	return 0

def learningNet(mat):
	return 0

def readFiles():
	fileDir = 'Data/'
	dMat=numpy.loadtxt(open(fileDir+'data-train-1.txt',"rb"),delimiter=',',skiprows=0)
	return dMat

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
