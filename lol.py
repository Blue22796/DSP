import signalcompare as sc
import funcs as fn

X,Y = fn.read_sample()
X2,Y2 = fn.read_sample()
s1 = sc.SignalComapreAmplitude(X,X2) 
s2 = sc.SignalComaprePhaseShift(Y,Y2)
if s1 and s2:
	print("Congratulations\n")
elif s1 or s2:
	print("Well done\n")
else:
	print("Good job")