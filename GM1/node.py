class node:
	def __init__(self, nodename):
		self.parent = []
		self.value = []
		self.cpt = []
		self.valueMap = {}
		self.name = nodename

	def addParent(self, parent):
		self.parent.append(parent)

	def setVal(self, value):
		for v in value:
			self.value.append(v)
	
	def setValMap(self, dic):
		self.valueMap = dic

