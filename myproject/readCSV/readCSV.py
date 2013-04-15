import csv
import itertools
import numpy
import networkx as nx
import matplotlib.pyplot as plt

tags = []
files = []
edges = []
g = nx.Graph()

def getEdges(mapping):
	coms = list(itertools.combinations(files, 2))
	for row in coms:
		ia = files.index(row[0])
		ib = files.index(row[1])
		ra = mapping[ia][:]
		rb = mapping[ib][:]
		x = [ra[i]&rb[i] for i in range(len(ra))].count(1)
		if x != 0:
			g.add_edge(row[0], row[1])
			edges.append(x)

def drawing(mapping):
	getEdges(mapping)

	pos = nx.spring_layout(g)
	nx.draw_networkx_nodes(g, pos, node_size=700, node_color='b', linewidths=None)
	print list(g.edges())
	print edges
	nx.draw_networkx_edges(g, pos, width=edges)
	nx.draw_networkx_labels(g, pos, font_size=12, font_family='sans-serif')

	plt.axis('off')
	plt.show()

def initRelation():

	with open('tag.csv','rb') as csvfile:
		tagreader = csv.reader(csvfile, delimiter=',')
		for row in tagreader:				
			files.append(row[0])
			for i in range(1, len(row)):
				if (row[i] in tags) == False:
					tags.append(row[i])

	mapping = [ [ 0 for i in range(len(tags))] for j in range(len(files))]
	with open('tag.csv','rb') as csvfile:
		tagreader = csv.reader(csvfile, delimiter=',')
		for nFile, row in enumerate(tagreader):
			for i in range(1, len(row)):
				tagInd = tags.index(row[i])
				mapping[nFile][tagInd] = 1	
	return mapping

def main():
	mapping = initRelation();
	print mapping
	drawing(mapping)

if __name__ == "__main__":
	main();
