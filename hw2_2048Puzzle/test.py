from random import randint

count0 = 0
count1 = 0
for x in xrange(100000):
	number = randint(0,1)
	if number == 0:
		count0 = count0 + 1
	else:
		count1 = count1 + 1
print "0: " + str(float(count0))
print "1: " + str(float(count1))

