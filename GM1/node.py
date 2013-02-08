class node:
	def __init__(self, nodename):
		self.parent = []
		self.value = []
		self.name = nodename


	def addParent(self, parent):
		self.parent.append(parent)

	def setVal(self, value):
		for v in value:
			self.value.append(v)

