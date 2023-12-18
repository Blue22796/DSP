import numpy as np
import funcs as fn
import math

event = fn.choose(['Generator','Read Sample','Skip'])
x,y = [],[]
while True:
	if event == 'Read Sample':
		x,y = fn.read_sample()

	elif event =='Generator':
		value = fn.read(['Amplitude','Frequency', 'Sampling frequency', 'Phase shift'])

		amp = float(value[0])
		freq = float(value[1])
		sf = float(value[2])
		ph = float(value[3])
	
		function = fn.choose(['Sin','Cos'])

		if function == 'Sin':
			ph-=math.pi/2
		x = np.arange(0,10,1/sf)
		y = [amp*math.cos(2*math.pi*freq*i + ph) for i in x]
	while True:
		event = fn.choose(['Apply operation','Graph','Print','Filter','Resample','Distinguish','Exit'])
		if event == 'Apply operation':
			x,y = fn.ops(x,y)
		else:
			break

	if event == 'Graph':
		fn.graph(x,y)
	if event == 'Print':
		fn.sigtofile(x,y)
	if event == 'Filter':
		x,y = fn.FIR(x,y)
	if event == 'Resample':
		x,y = fn.resample(x,y)
	if event == 'Distinguish':
		x,y = fn.dist()
	if event == 'Exit':
		break