import random as rand
import threading
import time
import Queue
import socket
import json
#from client import client

nPig = 6
pigList = []
attackingPig = -1
attackingStone = -1
msgRcvList = [0]*6
current_bird = -1
objLocList = []
time_ini = 0

class client(threading.Thread):
	
	def __init__(self, port):
		threading.Thread.__init__(self)
		self.c_port = port
		self.s = None
		self.conn = None
		
	def run(self):
		"""docstring for run"""
		global msg_q
		global pigList
		global msgRcvList
		print 'start receiving status report'
		status_all()
		msgRcvList = [0]*6
		#while loop the keep receiving response from pig
		#receive data
		data = "{\"msg_id\":\"123\"}"
		receive_msg(data)

	def stop(self):
		print 'stop receiving status report'
		global attackingPig
		#when the admin not receiving attackingPig(target)'s 
		#message, directly order it die.
		if attackingPig != -1 and msgRcvList[attackingPig] == 0:
			killPig()
			msg = directOrder()
			#send message
			jmsg = json.dump(msg)

		#when the attacking position if a stone, ask the pig
		#behind it(if any) get hurt
		if attackingStone != -1:
			if objLocList[attackingStone+1] in range(1, 7) :
				vals = {}
				vals['status'] = 1
				vals['dest' = objLocList[attckingStone+1]
				msg = constructMsg(5, vals)
				#send message
				jmsg = json.dump(msg)

class pigInfo:
	#a record of a pig
	def __init__(self, ID, loc, port):
		self.iden = ID
		self.loc = loc
		self.port = port
		self.status = 2
	
	def addNeigh(self, nlist):
		self.neighbors = nlist

	def getNeigh(self):
		return self.neighbors

def killPig():
	global attackingPig
	global objLocList
	pig = pigList[attackingPig]
	pig.status = 0
	objLocList[pig.loc] = 0

def arrangeNeighbors():
	"""docstring for arrangeNeighbors"""
	global objLocList

	for p in pigList:
		objLocList[p.loc] = p.iden
	
	for p in pigList:
		nlst = []
		if p.loc == 1:
			nlst.append(-1)
		else:
			nlst.append(objLocList[p.loc-1])
		if p.loc == 20:
			nlst.append(-1)
		else:
			nlst.append(objLocList[p.loc+1])
		p.addNeigh(nlst)

def initialize():
	"""docstring for initialize"""
    	global objLocList 
	global pigList
	ports = [x+3000 for x in range(100)]
	locations = [x for x in range(1, 21)]
	objLocList = [0] * 21

	portList = rand.sample(ports, nPig)
	locList = rand.sample(locations, nPig+7)
	for i in range(nPig):
		pig = Pig("127.0.0.1", portList[i], i+1, locList[i], [i-1, i+1])
		pigD = pigInfo(i+1, locList[i], portList[i])
                pigList.append(pigD)
		objLocList[locList[i]] = i+1

	for i in range(nPig, nPig+7):
		objLocList[locList[i]] = 20
	
	#set neighbor list in pig info based on their position
	arrangeNeighbors()

	#start the connection
#	for i in range(nPig):	
	#### In formal method, the connection should be random
	#	randConnect = rand.sample(portList, 3)

	##### Build a chain structured network for test

def receive_msg(data):
	"""docstring for receive_msg"""
	print 'start receive'
	print data
	global current_bird
	global msgRcvList

	msg_json = json.loads(data)
	msg_type = msg_json['type']
	if msg_type == 2:
		msg_id = msg_json['msg_id']
		pig_id = msg_json['pig_id']
		pig_status = msg_json['pig_status']
		if current_bird == msg_id and msgRcvList[int(pig_id)-1] != 2:
			msgRcvList[int(pig_id)-1] = 1
			pig = pigList[int(pig_id-1)]
			pig.status = pig_status
	
	#status report from pigs
	elif msg_type == 4:
		msg_id = msg_json['msg_id']
		pig_id = msg_json['pig_id']
		pig_status = msg_json['pig_status']
		pig_loc = msg_json['pig_loc']
		if current_bird == msg_id:
			msgRcvList[int(pig_id)-1] = 2
			pig = pigList[int(pig_id)-1]
			pig.status = pig_status
			pig.loc = pig_loc

def directOrder():
	"""docstring for directOrder"""
	global attackingPig
	vals = {"dist":attackingPig, "status":0}
	return constructMsg(5, vals)

def constructMsg(msg_type, values):
	msg = {}
	t = time.time()
	global current_bird

	#bird attack
	if msg_type == 1:
		msg["type"] = 1
		current_bird = t
		msg["msg_id"] = str(t)
		msg["dest"] = values["dest"]
		msg["targ_nei"] = values["targ_nei"]
		msg["arr_time"] = values["arr_time"]
		msg["time_pass"] = 0
		msg["hubC"] = 8
#		print msg

	#direct order
	elif msg_type == 5:
		msg["type"] = 5
		msg["msg_id"] = current_bird
		msg["path"] = [0]
		msg["dest"] = values["dest"]
		msg["status"] = values["status"]
		msg["hubC"] = 8
#		print msg

	return msg

def status_all():
	for pig in pigList:
		vals = {}
		vals["dest"] = pig.iden
		vals["status"] = -1
		msg = constructMsg(5, vals)
		
		#send message
		jmsg = json.dump(msg)

def report_to_user():
	status_dict = {0:'killed', 1:'hurt', 2:'great'}
	for pig in pigList:
		print 'Pig '+pig.iden+'is'+status_dict[pig.status]
		
def setTimer1(cc):
	print 'Bird lands.'
	cc.start()

def setTimer2(cc):
	print 'End of report time.'
	cc.stop()
	#set up neighbor list in pig info 
	#after receiving status report from pigs
	arrangeNeighbors()
	#tell user the status of all objects
	report_to_user()
	#start over from initialize bird attack
	start_game()

def start_game():
	global attackingPig
	global attackingStone
	attackingPig = -1
	attackingStone = -1
	
	str_pigs_loc = ''
	stonesLoc = []
	for ind, v in enumerate(objLocList):
		if v == 20:
			stonesLoc.append(ind)
		elif v < 7 and v > 0:
			str_pigs_loc += 'pig# ' + str(v) + ', loc: ' + str(ind) + '\n'
	print 'The pigs are:\n'+str_pigs_loc
	print 'And the stones are on:', stonesLoc
	
	attackingPos = int(raw_input('Where do you want to attack? '))
	if attackingPos not in range(21):
		start_game()
	print 'What is the arrival time?'
	arr_time = int(raw_input())
	print 'You want to attack '+ str(attackingPos) + '.'

	if objLocList[attackingPos] in range(1, 7):
		attackingPig = objLocList[attackingPos]

	if objLocList[attackingPos] == 20:
		attackingStone = attackingPos

	#construct the attacking message
	attr = {}
	attr["dest"] = attackingPos
	if attackingPig != -1:
		attr["targ_nei"] = pigList[attackingPig].getNeigh()
	else:
		nlst = []
		if attackingPos == 1:
			nlst.append(-1)
		else:
			nlst.append(objLocList[attackingPos-1])
		
		if attackingPos == 20:
			nlst.append(-1)
		else:
			nlst.append(objLocList[attackingPos+1])
		attr["targ_nei"] = nlst
		
	attr["arr_time"] = arr_time
	msg = constructMsg(1, attr)

	#send message
	jmsg = json.dump(msg)

	#set timer for 2 sec.
	#send bird attack message
	cc = client(pigList[0].port)
	t = threading.Timer(2.0, setTimer1, args=[cc])
	t.start()

	#wait for bird attack message spreading in P2P
	time.sleep(2)

	#send status_all() message to P2P to ask status
	#wait for response message
	t = threading.Timer(2.0, setTimer2, args=[cc])
	t.start()

def main():
	"""docstring for main"""
	initialize()
	start_game()

if __name__ == '__main__':
	main()
