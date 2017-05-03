import numpy as np


def CausalConvInt(val, time, kernel):
	"""
	Computing convolution of time varying data with given kernel function.
	"""
	ntime = time.size
	dt_temp = np.diff(time)
	dt = np.r_[time[0], dt_temp]
	out = np.zeros_like(val)
	for i in range(1, ntime):
		temp = 0.
		if i==0:
			temp += val[:,0]*kernel(time[i]-time[0])*dt[0]*0.5

		for k in range(1,i+1):
			temp += val[:,k-1]*kernel(time[i]-time[k-1])*dt[k]*0.5
			temp += val[:,k]*kernel(time[i]-time[k])*dt[k]*0.5
		out[:,i] = temp

	return  out

def CausalConvIntSingle(val, time, kernel):
	"""
	Computing convolution of time varying data with given kernel function.
	"""
	ntime = time.size
	dt_temp = np.diff(time)
	dt = np.r_[time[0], dt_temp]
	out = np.zeros_like(val)
	for i in range(1, ntime):
		temp = 0.
		if i==0:
			temp += val[0]*kernel(time[i]-time[0])*dt[0]*0.5

		for k in range(1,i+1):
			temp += val[k-1]*kernel(time[i]-time[k-1])*dt[k]*0.5
			temp += val[k]*kernel(time[i]-time[k])*dt[k]*0.5
		out[i] = temp

	return  out
