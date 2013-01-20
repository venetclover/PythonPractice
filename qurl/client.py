import re
import httplib2

#Download the text from the webpage
#
h = httplib2.Http("cache")
url = "http://www.33md.cc/html/9193/"
resp, content = h.request(url,"GET")
#resp, content = h.request("http://www.33md.cc/","GET")
str_content = content.decode(encoding='gb2312', errors='ignore')
#print str_content

#Extract the text
m = re.compile('<TITLE>(.*?)/').search(str_content)
title = m.group(1)
print "title: "+title

m = re.compile('<DIV class=play id="play_1">(.*?)</DIV>').search(str_content)
longVideoList = m.group(1)
#print longVideoList

m = re.compile('href="/(.*?)">').findall(longVideoList)
#print m
#print "http://www.33md.cc/"+m
qUrlList = []
for s in m:
	newUrl = "http://www.33md.cc/"+s
	resp, content = h.request(newUrl, "GET")
	str_content = content.decode(encoding='gb2312', errors='ignore')
	m = re.compile('<param name=\'URL\' value=\'(.*?)\'>').search(str_content)
	qUrlList.append(m.group(1))
#	print m.group(1)

for s in qUrlList:
	print s
