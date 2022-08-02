import numpy as np


class Traces:
	timestamps = []
	signals = {}
	def __init__(self,timestamp,signal = None):
		self.timestamps = timestamp
		if signal is not None:
			self.signals = signal
		
	def Add(self,name,signal):
		temp = {name: signal}
		self.signals.update(temp)

	def Get(self,name):
		return self.signals[name]

class Signal:
	def __init__(self,name,var_range):
		self.nom = "Signal"
		self.name = name
		self.var_range = var_range
		self.variables = [self.name]
		self.horizon = 0

	def find_id(self,traces,time):
		indice = -1
		for i in range(len(traces.timestamps)):
			if (traces.timestamps[i] == time):
				indice = i
		if indice == -1:
			raise Exception("time error") 
		else:
			return indice

	def eval(self, traces, time):
		return traces.signals[self.name][self.find_id(traces,time)]

class Const:
	def __init__(self,val):
		self.nom = "Const"
		self.val = val
		self.var_range = [val,val]
		self.variables = []
		self.horizon = 0

	def eval(self, traces, time):
		return self.val

class Substract:
	def __init__(self, left_formula, right_formula):
		self.nom = "Substract"
		self.left_formula = left_formula
		self.right_formula = right_formula
		A = left_formula.var_range[0] - right_formula.var_range[1]
		B = left_formula.var_range[1] - right_formula.var_range[0]
		if (A > B):
			temp = A
			A = B
			B = temp
		self.var_range = [A, B]
		self.variables = list(set(self.left_formula.variables + self.right_formula.variables))
		self.arity = 2

	def eval(self, traces, time):
		return self.left_formula.eval(traces, time) - self.right_formula.eval(traces, time)

class GreaterThan:
	# left_formula > right_formula ?
	def __init__(self, left_formula, right_formula):
		self.nom = "GreaterThan"
		self.left_formula = left_formula
		self.right_formula = right_formula
		A = left_formula.var_range[0] - right_formula.var_range[0]
		B = left_formula.var_range[1] - right_formula.var_range[1]
		if (A > B):
			temp = A
			A = B
			B = temp
		self.var_range = [A, B]
		self.horizon = 0
		self.variables = list(set(self.left_formula.variables + self.right_formula.variables))
		self.arity = 2

	def eval(self, traces, time):
		return Substract(self.left_formula,self.right_formula).eval(traces, time)

class LessThan:
	# left_formula < right_formula ?
	def __init__(self, left_formula, right_formula):
		self.nom = "LessThan"
		self.left_formula = left_formula
		self.right_formula = right_formula
		A = right_formula.var_range[0] - left_formula.var_range[0]
		B = right_formula.var_range[1] - left_formula.var_range[1]
		if (A > B):
			temp = A
			A = B
			B = temp
		self.var_range = [A, B]
		self.horizon = 0
		self.variables = list(set(self.left_formula.variables + self.right_formula.variables))
		self.arity = 2

	def eval(self, traces, time):
		return Substract(self.right_formula,self.left_formula).eval(traces, time)


class Abs:
	def __init__(self, formula):
		self.nom = "Abs"
		self.formula = formula
		A = np.abs(formula.var_range[0])
		B = np.abs(formula.var_range[1])
		if (A > B):
			temp = A
			A = B
			B = temp
		self.var_range = [A, B]
		self.variables = self.formula.variables
		self.horizon = 0
		self.arity = 1

	def eval(self, traces, time):
		temp = self.formula.eval(traces, time)
		if temp < 0 :
			return -1*temp
		else :
			return temp

class Sum:
	def __init__(self, left_formula, right_formula):
		self.nom = "Sum"
		self.left_formula = left_formula
		self.right_formula = right_formula
		A = left_formula.var_range[0] + right_formula.var_range[0]
		B = left_formula.var_range[1] + right_formula.var_range[1]
		self.var_range = [A, B]
		self.variables = list(set(self.left_formula.variables + self.right_formula.variables))
		self.arity = 2

	def eval(self, traces, time):
		return self.left_formula.eval(traces, time) + self.right_formula.eval(traces, time)

class Implication:
	def __init__(self, left_formula, right_formula):
		self.nom = "Implication"
		self.left_formula = left_formula
		self.right_formula = right_formula
		A = max(-1*self.left_formula.var_range[1], self.right_formula.var_range[0])
		B = max(-1*self.left_formula.var_range[0], self.right_formula.var_range[1])
		self.var_range = [A, B]
		self.horizon = max(self.left_formula.horizon, self.right_formula.horizon)
		self.variables = list(set(self.left_formula.variables + self.right_formula.variables))
		self.arity = 2

	def eval(self, traces, time):
		return Or(Not(self.left_formula),self.right_formula).eval(traces, time)

class Equals:
	def __init__(self, left_formula, right_formula):
		self.nom = "Equals"
		self.left_formula = left_formula
		self.right_formula = right_formula
		A = -1*np.abs(right_formula.var_range[0] - left_formula.var_range[0])
		B = -1*np.abs(right_formula.var_range[1] - left_formula.var_range[1])
		if A > B :
			temp = A
			A = B
			B = temp 
		self.var_range = [A, B]
		self.horizon = 0
		self.variables = list(set(self.left_formula.variables + self.right_formula.variables))
		self.arity = 2

	def eval(self, traces, time):
		temp = Substract(self.left_formula,self.right_formula).eval(traces, time)
		if temp == 0 :
			return 1
		else :
			return Not(Abs(Substract(self.left_formula,self.right_formula))).eval(traces, time)

class Not:
	def __init__(self, formula):
		self.nom = "Not"
		self.formula = formula
		A = -1*formula.var_range[0] 
		B = -1*formula.var_range[1] 
		if A > B :
			temp = A
			A = B
			B = temp
		self.var_range = [A, B]
		self.horizon = self.formula.horizon
		self.variables = self.formula.variables
		self.arity = 1


	def eval(self, traces, time):
		return -1*self.formula.eval(traces, time)

class Next:
	def __init__(self, formula):
		self.nom = "Next"
		self.formula = formula
		self.var_range = formula.var_range
		self.horizon = 1 + self.formula.horizon
		self.variables = self.formula.variables
		self.arity = 1

	def find_next(self,traces,time):
		indice = -1
		for i in range(len(traces.timestamps)):
			if (traces.timestamps[i] == time):
				indice = i
		if indice == -1:
			raise Exception("time error") 
		else:
			return traces.timestamps[indice+1]
		

	def eval(self, traces, time):
		time_temp = self.find_next(traces,time)
		return self.formula.eval(traces, time_temp)

class Global:
	def __init__(self, lower_time_bound, upper_time_bound, formula):
		self.nom = "Global"
		self.upper_time_bound = upper_time_bound
		self.lower_time_bound = lower_time_bound
		self.formula = formula
		self.var_range = formula.var_range
		self.horizon = self.upper_time_bound + self.formula.horizon
		self.variables = self.formula.variables

	def eval(self, traces, time):
		i = 0
		while  traces.timestamps[i] < self.lower_time_bound:
			i += 1
		min_temp = self.formula.eval(traces, traces.timestamps[i])
		while  traces.timestamps[i] < self.upper_time_bound:
			if min_temp > self.formula.eval(traces, traces.timestamps[i]):
				min_temp = self.formula.eval(traces, traces.timestamps[i])
			i += 1
		return min_temp

class Finally:
	def __init__(self, lower_time_bound, upper_time_bound, formula):
		self.nom = "Finally"
		self.upper_time_bound = upper_time_bound
		self.lower_time_bound = lower_time_bound
		self.formula = formula
		self.var_range = formula.var_range
		self.horizon = self.upper_time_bound + self.formula.horizon
		self.variables = self.formula.variables

	def eval(self, traces, time):
		temp = Not(Global(self.lower_time_bound,self.upper_time_bound,Not(self.formula)))
		return temp.eval(traces, time)

class Or:
	formulas = []
	def __init__(self, *args):
		self.nom = "Or"
		def Maxhorizon(formulas):
			temp = formulas[0].horizon
			for i in range(len(formulas)):
				if formulas[i].horizon > temp:
					temp = formulas[i].horizon
			return temp

		def MaxRange(formulas,bound):
			temp = formulas[0].var_range[bound]
			for i in range(len(formulas)):
				if formulas[i].var_range[bound] < temp:
					temp = formulas[i].var_range[bound]
			return temp

		self.formulas = []
		for parameter in args:
			self.formulas.append(Not(parameter))

		A = MaxRange(self.formulas,0)
		B = MaxRange(self.formulas,1)
		self.var_range = [A, B]
		self.horizon = Maxhorizon(self.formulas)

		temp = []
		for i in range(len(self.formulas)):
			temp += self.formulas[i].variables
		self.variables = list(set(temp))

	def eval(self, traces,time):
		temp = Not(And(formula = self.formulas))
		return temp.eval(traces, time)

class And:
	formulas = []
	def __init__(self, *args, formula = None):
		self.nom = "And"
		def Maxhorizon(formulas):
			temp = formulas[0].horizon
			for i in range(len(formulas)):
				if formulas[i].horizon > temp:
					temp = formulas[i].horizon
			return temp

		def MinRange(formulas,bound):
			temp = formulas[0].var_range[bound]
			for i in range(len(formulas)):
				if formulas[i].var_range[bound] < temp:
					temp = formulas[i].var_range[bound]
			return temp

		self.formulas = []
		if formula is not None:
			self.formulas = formula
		else:
			for parameter in args:
				self.formulas.append(parameter)

		A = MinRange(self.formulas,0)
		B = MinRange(self.formulas,1)
		self.var_range = [A, B]
		self.horizon = Maxhorizon(self.formulas)

		temp_ = []
		for i in range(len(self.formulas)):
			temp_ += self.formulas[i].variables
		self.variables = list(set(temp_))

	def eval(self, traces,time):

		def P_Prime(p,p_min):
			tab = []
			for i in range(len(p)):
				temp = (p[i]-p_min)/p_min
				tab.append(temp)
			return tab

		mu = 1
		p = []
		for i in self.formulas:
			p.append(i.eval(traces, time))
		p_min = min(p)
		numerator = 0
		if p_min < 0:
			denominator = 0
			p_prime = P_Prime(p,p_min)
			for i in range(len(p)):
				numerator += p_min*(np.e**p_prime[i])*(np.e**(mu*p_prime[i]))
				denominator += np.e**(mu*p_prime[i])
			return numerator/denominator
		else:
			if p_min > 0:
				denominator = 0
				p_prime = P_Prime(p,p_min)
				for i in range(len(p)):
					numerator += p[i]*np.e**(-mu*p_prime[i])
					denominator += np.e**(-mu*p_prime[i])
				return numerator/denominator
			else:
				return numerator

class Until:

	def __init__(self,lower_time_bound,upper_time_bound,left_formula,right_formula):
		self.nom = "Until"
		self.left_formula = left_formula
		self.right_formula = right_formula
		self.upper_time_bound = upper_time_bound
		self.lower_time_bound = lower_time_bound
		A = min(self.left_formula.var_range[0], self.right_formula.var_range[0])
		B = min(self.left_formula.var_range[1], self.right_formula.var_range[1])
		self.var_range = [A, B]
		self.horizon = self.upper_time_bound +  max(self.left_formula.horizon, self.right_formula.horizon)
		self.variables = list(set(self.left_formula.variables + self.right_formula.variables))
	
	def eval(self, traces,time):
		i = 0
		while  traces.timestamps[i] < self.lower_time_bound:
			i += 1
		min_left_formula = self.left_formula.eval(traces, traces.timestamps[i])

		while  traces.timestamps[i] < self.upper_time_bound & self.right_formula.eval(traces, traces.timestamps[i]):
			if min_left_formula > self.left_formula.eval(traces, traces.timestamps[i]):
				min_left_formula = self.left_formula.eval(traces, traces.timestamps[i])
			i += 1
		return min_left_formula

class Release:
	def __init__(self,lower_time_bound,upper_time_bound,left_formula,right_formula):
		self.nom = "Release"
		self.left_formula = left_formula
		self.right_formula = right_formula
		self.upper_time_bound = upper_time_bound
		self.lower_time_bound = lower_time_bound
		A = min(self.left_formula.var_range[0], self.right_formula.var_range[0])
		B = min(self.left_formula.var_range[1], self.right_formula.var_range[1])
		self.var_range = [A, B]
		self.horizon = self.upper_time_bound +  max(self.left_formula.horizon, self.right_formula.horizon)
		self.variables = list(set(self.left_formula.variables + self.right_formula.variables))
	
	def eval(self, traces,time):
		i = 0
		while  traces.timestamps[i] < self.lower_time_bound:
			i += 1
		min_left_formula = self.left_formula.eval(traces, traces.timestamps[i])

		while  traces.timestamps[i] < self.upper_time_bound & self.right_formula.eval(traces, traces.timestamps[i-1]):
			if min_left_formula > self.left_formula.eval(traces, traces.timestamps[i]):
				min_left_formula = self.left_formula.eval(traces, traces.timestamps[i])
			i += 1
		return min_left_formula