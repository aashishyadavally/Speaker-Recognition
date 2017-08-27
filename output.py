def trainy_main():
	output1 = []
	for i in range(36):
		temp = []
		for j in range(36):
			if j==i:
				temp.append(1)
			else:
				temp.append(0)
		output1.append(temp)
	return output1

def testy_main():
	output2 = []
	for i in range(36):
		temp = []
		for j in range(36):
			if j==i:
				temp.append(1)
			else:
				temp.append(0)
		output2.append(temp)
	return output2

