import matplotlib.pyplot as plt
import math
import itertools as it
import operator as oper
import PySimpleGUI as sg
import cmath

def display_message(s1, s2 = 'Ok'):
	layout = [[sg.Text(x)],[sg.Button(s2)]]
	window = sg.Window('Task', layout)
	window.read()
	window.close()

def read(reqs):
	layout = []
	for req in reqs:
		layout.append([sg.Text(req),sg.InputText()])
	
	layout.append([sg.Button('Submit')])
	window = sg.Window('Task1', layout)
	event, value = window.read()
	window.close()
	return value

def choose(options):
	layout = []
	for option in options:
		layout.append([sg.Button(option)])
	window = sg.Window('Task1', layout)
	event, value = window.read()
	window.close()
	return event	

def read_sample():
	layout = [[sg.InputText(key = 'name'),sg.FileBrowse()],
	[sg.Button("OK")]]
	window = sg.Window('task1', layout)
	nvm, val = window.read()
	window.close()
	name = val['name']

	f = open(name,'r')
	f.readline()
	f.readline()
	n = int(f.readline())
	x = []
	y = []
	for i in range(n):
		l = f.readline()
		x0,y0 = l.split(' ')[:2]
		if y0[-1]=='\n':
			y0 = y0[:-1]
		if x0[-1]=='f':
			x0 = x0[:-1]
		if y0[-1] == 'f':
			y0 = y0[:-1]
		x.append(float(x0))
		y.append(float(y0))
	return x,y


def ops(x,y):
	op = choose(['Add', 'Subtract' ,'Multiply', 'Square', 'Shift', 'Normalize', 'Accumulate','FFT','IFFT','Mod','Quant','DCT'
		,'-DC'])
	if op in ['Add', 'Subtract' ,'Multiply']:
		x2, y2 = read_sample()
		if(x2!=x):
			display_message('Invalid operation')
			return ops(x,y)
		for i in range(len(y)):
			if op == 'Add':
				y[i] +=y2[i]
			elif op=='Subtract':
				y[i] -=y2[i]
			elif op=='Multiply':
				y[i] *=y2[i]
		return x,y

	if op == 'Square':
		return x,[i*i for i in y]
	elif op == 'Shift':
		c = float(read(['Value'])[0])
		return x,[i+c for i in y]
	elif op == 'Accumulate':
		return x, list(it.accumulate(y,oper.Add))
	elif op == 'Normalize':
		rng = read(['Upper bound','Lower bound'])
		ub, lb = float(rng[0]), float(rng[1])
		if ub<lb:
			ub,lb = lb,ub
		mn = min(y)
		mx = max(y)
		old_rng = mx-mn
		new_rng = ub - lb
		return x,[(i-mn)/old_rng*new_rng for i in y]
	if op == 'FFT':
		return fft(y)
	if op == 'IFFT':
		return ifft(x,y)
	if op == 'Mod': 
		return mod(x,y)
	if op == 'Quant':
		return x,quantize(y)
	if op == 'DCT':
		return dct(x,y)
	if op == '-DC':
		return unDC(x,y)

def round(y, levels):
	Y = []
	for i in y:
		mn = 10**9
		val = -1
		for j in levels:
			if(abs(j-i)<mn):
				mn = abs(j-i)
				val = j
		Y.append(val)
	return Y

def quantize(y):
	inp = read(['Levels','Bits'])
	levels = 0
	try:
		levels = int(inp[0])
	except:
		try:
			bits = int(inp[1])
			levels = int(2**bits)
		except:
			print("Bad input\n")
	mn,mx = min(y), max(y)
	rng = mx-mn
	lvl_rng = rng/levels
	vals = [mn+lvl_rng*(.5+i) for i in range(levels)]
	return round(y,vals)
	

def mod(x,y):
	freq = int(choose([i+1 for i in range(len(y))]))
	new_coords = read(['Amp','Phi'])
	amp = float(new_coords[0])
	phase = float(new_coords[1])
	x[freq-1] = amp
	y[freq-1] = phase
	return x,y

def fft(X,s=-1):
	Y = []
	n = len(X)
	for i in range(n):
		arg = 0
		tot = 0
		for j in range(n):
			delta = cmath.rect(X[j].real,arg)
			tot+= delta
			arg+=2*s*math.pi*i/n
		Y.append(tot)
	X = [(i.real**2+i.imag**2)**.5 for i in Y]
	Y = [math.atan(i.imag/(i.real+10**-9)) for i in Y]
	return X,Y

def ifft(X,Y):
	n = len(X)
	X = [cmath.rect(X[i],Y[i])/n for i in range(n)]
	Y,X = fft(X,s=1)
	X = range(len(X))
	return X,Y

def dct(x,y):
	N = len(y)
	xf = range(N)
	yf = []
	for i in xf:
		yf.append(0)
		for j in xf:
			yf[i]+=((2/N)**.5)*y[j]*math.cos(math.pi*(2*i-1)*(2*j-1)/(4*N))
	print(yf)
	n = read(['Samples #'])
	n = int(n[0])
	if n>0:
		sigtofile(xf[:n],yf[:n],'DCT')
	return xf,yf

def unDC(x,y):
	avg = sum(y)/len(y)
	print(x)
	y = [i-avg for i in y]
	print(y)
	return x,y

def graph(x,y):
	rep = choose(['Digital','Analog']) 

	if rep =='Digital':
		plt.stem(x,y)

	if rep =='Analog':	
		plt.plot(x,y)
	plt.show()
	
def sigtofile(X,Y,name = 'output'):
	f = open(name+'.txt','w')
	f.write('0\n')
	f.write('1\n')
	f.write(str(len(X))+'\n')
	for i in range(len(X)):
		f.write(str(X[i])+' ')
		f.write(str(Y[i])+'\n')
	f.close()	