import matplotlib.pyplot as plt
import math
import itertools as it
import operator as oper
import PySimpleGUI as sg
import cmath
import os

def display_message(s1, s2 = 'Ok'):
	layout = [[sg.Text(s1)],[sg.Button(s2)]]
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

def read_folder(cls = 1):
	layout = [[sg.InputText(key = 'name'),sg.FolderBrowse()],
	[sg.Button("OK")]]
	window = sg.Window('task1', layout)
	nvm, val = window.read()
	window.close()
	name = val['name']
	
	y = []

	for file in os.listdir(name):
		f = os.path.join(name, file)
		xh,yh = read_file(f)
		y.append(yh)

	if cls == 0:
		return y

	Y = []
	for i in range(len(y[0])):
		v = 0
		for j in y:			
			v+=j[i]
		v/=len(y)
		Y.append(v)
	return Y
	

def read_file(name):
	f = open(name,'r')
	y = [int(i) for i in f.read().splitlines()]
	return range(len(y)),y

def ops(x,y):
	op = choose(['Add', 'Subtract' ,'Multiply', 'Square', 'Shift', 'Normalize', 'Accumulate','FFT','IFFT','Mod','Quant','DCT'
		,'-DC','Smooth','Sharpen','Delay/Advance','Fold','-DC(FD)','Convolve','Correlation'])
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

	if op == 'Correlation':
		return cor(x,y)

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
	if op == 'Smooth':
		return smooth (x,y)
	if op == 'Sharpen':
		return sharpen(x,y)
	if op == 'Delay':
		k = read(['Delay (enter -ve value to advance)'])
		return [i+k for i in x], y
	if op == 'Fold':
		x =[-i for i in x], y
		XY = []
		for i in range(len(x,y)):
			XY.append((x,y))
		XY.sort()
		x = [i[0] for i in XY]
		y = [i[1] for i in XY]
		return x,y
	if op == '-DC(FD)':
		return undcfc(x,y)
	if op == 'Convolve':
		return conv(x,y)

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
			delta = X[j]*cmath.rect(1,arg)
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

def undcfc(x,y):
	x,y = fft(y)
	for i in range(len(x)):
		if x[i] == 0:
			y[i] = 0
	return ifft(x,y)

def smooth(x,y):
	k = read("# of points")
	n = len(x)
	X = []
	Y = []
	if i + k - k//2 - 1 == n:
		return X,Y
	for i in range(k//2,n):
		v = 0
		X.append(x[i])
		for j in range (i-k//2,i+k-k//2-1):
			v+=y[i]
		v/=k
		Y.append(v)

def sharpen(x,y):
	n = read('nth derivate')
	for i in range(n):
		x,y = dydx(x,y)
	return x,y

def dydx(x,y):
	x = x[1:]
	y = [y[i] - y[i-1] for i in range(1,len(y))]
	return x,y

def to_sig(x,y):
	s = {}
	for i in range(len(x)):
		s[x[i]] = y[i]
	return s

def convolution(x,y,x2,y2):
	N = int(x[-1]+x2[-1])
	B = int(min(x[0],x2[0]))
	s1 = to_sig(x,y)
	s2 = to_sig(x2,y2)
	for k in s1:
		if not k in s2:
			s2[k] = 0
	for k in s2:
		if not k in s1:
			s1[k] = 0
	S = {}
	for i in range(B,N+1):
		S[i] = Cnth(s1,s2,i)
	return [k for k in S],[S[k] for k in S]

def Cnth(s1,s2,n):
	ans = 0

	for k in s1:
		if n-k in s2:
			ans+=s1[k]*s2[n-k]
	return ans	
def conv(x,y):
	x2,y2 = read_sample()
	
	return convolution(x,y,x2,y2)
	

def cor(x,y):
	x2,y2 = read_sample()
	s1 = {}
	s2 = {}
	for i in range(len(x)):
		s1[x[i]] = y[i]
		s2[x[i]] = 0
	for i in range(len(x2)):
		s2[x2[i]] = y2[i]
		if not x2[i] in s1:
			s1[x2[i]] = 0

	x = []
	x2 = []
	y = []
	y2 = []
	for i in  s1:
		x.append(i)
		x2.append(i)
		y.append(s1[i])
		y2.append(s2[i])

	op = choose(['Correlation', 'Time-analysis', 'Temp matching'])
	if op == 'Correlation':
		x,y = fft(y)
		x2,y2 = fft(y2)
		X,Y = [x[i]*x2[i] for i in range(len(x))],[y[i]-y2[i] for i in range(len(y))]
		x,y = ifft(X,Y)
		sum = 0
		for i in y:
			sum+=i
		for i in range(len(y)):
			y[i]/=sum
		return x,y



	if op == 'Time-analysis':
		Fs = float(read(['Fs'])[0])
		Ts = 1/Fs
		ans, p = time_analyze(y,y2)
		display_message('Shift = '+str(ans*Ts)+'\nCorrelation = '+str(p))
		return x,y
	if op == 'Temp matching':
		c1 = read_folder()
		c2 = read_folder()
		tst = read_folder(0)
		classes = []
		for t in tst:
			v1 = time_analyze(t,c1)[1]
			v2 = time_analyze(t,c2)[1]
			if v1>v2:
				classes.append(1)
			else:
				classes.append(2)	
		sigtofile(range(1,len(classes)+1),classes,'classification')
	return x,y

def time_analyze(y1,y2):	
	ans, p = 0,0
	for i in range(len(y1)):
		v = p12(y1,y2,i)
		if v>p:
			ans,p = i,v
	return ans,p


def p12(y1,y2,n):
	N = len(y2)
	nrm1 = 0
	for i in y1:
		nrm1 += i**2
	nrm2 = 0
	for i in y2:
		nrm2 += i**2
	ans = 0
	for i in range(N):
		ans+= y1[i]*y2[(i+n)%N]

	ans/=(nrm1*nrm2)**.5
	return ans

def filt(att):
	filters = [[.9,.7416,13,21], [3.1,.0546,31,44], [3.3,.0194,41,53],[5.5,.0017,57,74]]
	for filter in filters:
		if filter[3] >= att:
			return filter

def clc_fil(filter,x,N):
	if filter[0] == .9:
		return 1
	if filter[0] == 3.1:
		return .5 + .5*math.cos(2*math.pi*x/N)
	if filter[0] == 3.3:
		return .54 + .46*math.cos(2*math.pi*x/N)
	if filter[0] == 5.5:
		return .42 + .5*math.cos(2*math.pi*x/(N-1)) + .08*math.cos(4*math.pi*x/(N-1))

def LPF(n,Fc):
	if n == 0:
		return 2*Fc
	return math.sin(2*math.pi*Fc*n)/math.pi/n

def HPF(n,Fc):
	if n == 0:
		return 1 - LPF(n,Fc)
	return -LPF(n,Fc)

def BPF(n,f1,f2):
	if n == 0:
		return 2*(f2-f1)
	w1 = f1*2*math.pi
	w2 = f2*2*math.pi
	return 2*(f2*math.sin(w2*n)/w2 - f1*math.sin(w1*n)/w1)/n

def BSF(n,f1,f2):
	if n == 0:
		return 1 - BPF(n,f1,f2)
	return -BPF(n,f1,f2)

def fil_apply(filter,wid,type,fc =0, f2 = 0):
	N = math.ceil(filter[0]/wid)
	if N%2==0:
		N = N+1
	
	X = []
	FY = []
	IY = []
	for i in range(-(N//2),N//2+1):
		X.append(i)
		FY.append(clc_fil(filter,i,N))
		if type == 'low':
			IY.append(LPF(i,fc+wid/2))
		if type == 'high':
			IY.append(HPF(i,fc-wid/2))
		if type == 'band pass':
			IY.append(BPF(i,fc-wid/2,f2+wid/2))
		if type == 'band stop':
			IY.append(BSF(i,fc+wid/2,f2-wid/2))

	Y = [FY[i]*IY[i] for i in range(len(FY))]
	return X,Y
	
	
def filter(x,y,type,specs):
	Fs = float(specs[0])
	att = float(specs[1])
	width = float(specs[2])/Fs
	filter = filt(att)
	if type in ['low','high']:
		fc = float(specs[3])/Fs
		print(fc)
		x2,y2 = fil_apply(filter,width,type,fc)
	else:
		f1 = float(specs[3])/Fs
		f2 = float(specs[4])/Fs
		x2,y2 = fil_apply(filter,width,type,f1,f2)
	print(x)
	print(x2)
	ans =  convolution(x,y,x2,y2)
	return ans

def FIR(x,y):
	type = choose(['low','high','band pass','band stop'])
	specs= []
	if type in ['low','high']:
		specs = read(['Fs','Attentuation','Transition band','Fc'])
	elif type in ['band pass','band stop']:
		specs = read(['Fs','Attentuation','Transition band','F1','F2'])
	
	return filter(x,y,type,specs)
			

def up_sample(x,y,L):
	N = len(x)
	X = range(L*N)
	Y = []
	for i in range(L*N):
		if i%L == 0:
			Y.append(y[i//L])
		else:
			Y.append(0)
	return X,Y

def down_sample(x,y,M):
	X = [i+x[0] for i in range(len(x)//M)]
	Y = [y[i] for i in range(0,len(y),M)]
	return X,Y

def resample(x,y):
	X = []
	Y = []
	specs = read(['Fs','Attentuation','Transition band','Fc'])
	LM = read(['L','M'])
	L = int(LM[0])
	M = int(LM[1])
	if L==0 and M==0:
		display_message('Invalid')


	if L!=0:
		x,y = up_sample(x,y,L)
	x,y = filter(x,y,'low',specs)
	if M!=0:
		x,y = down_sample(x,y,M)
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