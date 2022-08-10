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
		return np.array(self.signals[name])

class Signal:
	def __init__(self,name,var_range):
		self.nom = "Signal"
		self.name = name
		self.var_range = var_range
		self.variables = [self.name]
		self.horizon = 0
	"""
	def find_id(self,traces): 
	# from a timestamps time, return the correspondant index of the table 
		indice = -1
		for i in range(len(traces.timestamps)):
			if (traces.timestamps[i] == time):
				indice = i
		if indice == -1:
			raise Exception("time error") 
		else:
			return indice
	"""
	def eval(self, traces):
		return np.array(traces.signals[self.name])
		

class Const:
# for a constant signal
	def __init__(self,val):
		self.nom = "Const"
		self.val = val
		self.var_range = [val,val]
		self.variables = []
		self.horizon = 0

	def eval(self, traces):
		return np.full(len(traces.timestamps), self.val)


#for the following classes there are several parameters in the initilisation :
#nom = name of the function 
#formula, right_formula, left_formula = the formulas that we use in the function
#var_range = the range of the return value
#variables = the names of the signals used in the function
#horizon = ?
#arity = number of formula that are used in the function

class Subtract:
#left_formula - right_formula
	def __init__(self, left_formula, right_formula):
		self.nom = "Subtract"
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

	def eval(self, traces):
		res = []
		left_formula_robustness = self.left_formula.eval(traces)
		right_formula_robustness = self.right_formula.eval(traces)
		return np.subtract(left_formula_robustness,right_formula_robustness)

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

	def eval(self, traces):
	#return the difference between the left and the right formula : if it is positive the lessThan is true, otherwise it is false
		return Subtract(self.left_formula,self.right_formula).eval(traces)

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

	def eval(self, traces):
	#return the difference between the right and the left formula : if it is positive the lessThan is true, otherwise it is false
		return Subtract(self.right_formula,self.left_formula).eval(traces)


class Abs:
# absolute value of the formula at a given time
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

	def eval(self, traces):
		formula_robustness = self.formula.eval(traces)
		return np.absolute(formula_robustness)

class Sum:
# left_formula + right_formula at a given time
	def __init__(self, left_formula, right_formula):
		self.nom = "Sum"
		self.left_formula = left_formula
		self.right_formula = right_formula
		A = left_formula.var_range[0] + right_formula.var_range[0]
		B = left_formula.var_range[1] + right_formula.var_range[1]
		self.var_range = [A, B]
		self.variables = list(set(self.left_formula.variables + self.right_formula.variables))
		self.arity = 2

	def eval(self, traces):
		res = []
		left_formula_robustness = self.left_formula.eval(traces)
		right_formula_robustness = self.right_formula.eval(traces)
		return np.add(left_formula_robustness,right_formula_robustness)

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

	def eval(self, traces):
		return Or(Not(self.left_formula),self.right_formula).eval(traces)

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

	def eval(self, traces):
	# Compute -|left_formula - right_formula|for each time. If this is nonzero, return as is.
	# Otherwise return 1.
		formula_robustness = Not(Abs(Subtract(self.left_formula,self.right_formula))).eval(traces)
		return np.where(formula_robustness == 0, 1, formula_robustness)	

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


	def eval(self, traces):
		formula_robustness = self.formula.eval(traces)
		return formula_robustness*-1


class Next:
#Return the value at the next state
	def __init__(self, formula):
		self.nom = "Next"
		self.formula = formula
		self.var_range = formula.var_range
		self.horizon = 1 + self.formula.horizon
		self.variables = self.formula.variables
		self.arity = 1

	def find_next(self,traces):
		indice = -1
		for i in range(len(traces.timestamps)):
			if (traces.timestamps[i] == time):
				indice = i
		if indice == -1:
			raise Exception("time error") 
		else:
			if len(traces.timestamps) == indice+1:
				return traces.timestamps[indice]
			else:
				#Add +1 to return the index of the next state
				return traces.timestamps[indice+1]
		

	def eval(self, traces):
		formula_robustness = self.formula.eval(traces)
		res = np.roll(formula_robustness, -1)
		res[len(res) - 1] = res[len(res) - 2]
		return res


class Global:
#formula has to be True on the entire subsequent path
	def __init__(self, lower_time_bound, upper_time_bound, formula):
		self.nom = "Global"
		self.upper_time_bound = upper_time_bound
		self.lower_time_bound = lower_time_bound
		self.formula = formula
		self.var_range = formula.var_range
		self.horizon = self.upper_time_bound + self.formula.horizon
		self.variables = self.formula.variables
		
		

	def eval(self, traces):
		i = 0
		#finding the first index
		while  traces.timestamps[i] < self.lower_time_bound :  
			i += 1

		#initialistion with the higest value
		min_temp = self.var_range[1]
		formula_robustness = self.formula.eval(traces)

		#if the time bound isn't exceed
		while  i < len(traces.timestamps) and traces.timestamps[i] <= self.upper_time_bound :
			#finding the minimum
			if min_temp > formula_robustness[i]:
				min_temp = formula_robustness[i]
			i += 1
		return np.full(len(formula_robustness),min_temp)


#formula eventually has to be True (somewhere on the subsequent path)
class Finally:
	def __init__(self, lower_time_bound, upper_time_bound, formula):
		self.nom = "Finally"
		self.upper_time_bound = upper_time_bound
		self.lower_time_bound = lower_time_bound
		self.formula = formula
		self.var_range = formula.var_range
		self.horizon = self.upper_time_bound + self.formula.horizon
		self.variables = self.formula.variables

	def eval(self, traces):
		formula_robustness = Not(Global(self.lower_time_bound,self.upper_time_bound,Not(self.formula)))
		return formula_robustness.eval(traces)

class Or:
	formulas = []
	def __init__(self, *args):

		self.nom = "Or"
		# Calculate the horizon
		def Maxhorizon(formulas):
			#Max of all the horizon
			temp = formulas[0].horizon
			for i in range(len(formulas)):
				if formulas[i].horizon > temp:
					temp = formulas[i].horizon
			return temp

		# Calculate the Range
		def MaxRange(formulas,bound):
			#Min of all the horizon
			temp = formulas[0].var_range[bound]
			for i in range(len(formulas)):
				if formulas[i].var_range[bound] > temp:
					temp = formulas[i].var_range[bound]
			return temp

		#saving all the NOT(formulas)
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

	def eval(self, traces):
		formula_robustness = Not(And(formula = self.formulas))
		return formula_robustness.eval(traces)

class And:
	formulas = []
	def __init__(self, *args, formula = None):
		self.nom = "And"
		# Calculate the horizon
		def Maxhorizon(formulas):
			#Max of all the horizon
			temp = formulas[0].horizon
			for i in range(len(formulas)):
				if formulas[i].horizon > temp:
					temp = formulas[i].horizon
			return temp

		# Calculate the Range
		def MinRange(formulas,bound):
			#Min of all the horizon
			temp = formulas[0].var_range[bound]
			for i in range(len(formulas)):
				if formulas[i].var_range[bound] < temp:
					temp = formulas[i].var_range[bound]
			return temp

		#saving all the formulas
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

	def eval(self, traces):

		#mu can be change
		mu = 1
		p = []
		#evaluate all the formulas
		for i in self.formulas:
			p.append(i.eval(traces))
		#finding the min
		p_min = np.array(p).min(axis = 0)

		temp_min = np.tile(p_min, (len(self.formulas), 1))
		p_prime = np.array(p)- temp_min
		p_prime = np.divide(p_prime,temp_min)

		res = []
		for i in range(len(p[0])):
			numerator = 0
			if p_min[i] < 0:
				denominator = 0
				for j in range(len(p)):
					numerator += p_min[i]*(np.e**p_prime[j][i])*(np.e**(mu*p_prime[j][i]))
					denominator += np.e**(mu*p_prime[j][i])
				res.append(numerator/denominator)
			else:
				if p_min[i] > 0:
					denominator = 0
					for j in range(len(p)):
						numerator += p[j][i]*np.e**(-mu*p_prime[j][i])
						denominator += np.e**(-mu*p_prime[j][i])
					res.append(numerator/denominator)
				else:
					# if p_min == 0:
					res.append(numerator) # equal to 0
		return np.array(res)


# left_formula has to hold at least until right_formula; if right_formula never becomes true, left_formula must remain true forever. 
class Weak_Until:
	def __init__(self,lower_time_bound,upper_time_bound,left_formula,right_formula):
		self.nom = "Until"
		self.left_formula = left_formula
		self.right_formula = right_formula
		self.upper_time_bound = upper_time_bound
		self.lower_time_bound = lower_time_bound
		self.var_range = left_formula.var_range
		self.horizon = self.upper_time_bound +  max(self.left_formula.horizon, self.right_formula.horizon)
		self.variables = list(set(self.left_formula.variables + self.right_formula.variables))
	
	def eval(self, traces,time):
		i = 0
		#finding the first index
		while  traces.timestamps[i] < self.lower_time_bound:  
			i += 1

		#initialistion with the higest value
		min_left_formula = self.var_range[1]

		#evaluate the left formulas
		left_formula_robustness = self.left_formula.eval(traces)
		right_formula_robustness = self.right_formula.eval(traces)
		
		#if the time bound is exceed or if the right_formula is True
		while  i < len(traces.timestamps) and traces.timestamps[i] <= self.upper_time_bound and right_formula_robustness[i] < 0:
			

			#finding the minimum of the left formula
			if min_left_formula > left_formula_robustness[i]:
				min_left_formula = left_formula_robustness[i]
			i += 1
		#return the minimum of the left formula
		return np.full(len(left_formula_robustness),min_left_formula)

class Predicate(object):
	def __init__(self, variable_name, A_Matrix, bound, robustness=None, param_name=None, process_type='cpu',
				 thread_pool=False):
		self.variable_name = variable_name
		self.arity = len(self.variable_name)
		self.value = None
		self.robustness_array = None
		self.param_name = param_name
		self.predicate_robustness = None
		if robustness != None:
			self.predicate_robustness = np.array(robustness, dtype=np.float32)
		if type(bound) == list:
			self.bound = np.array(bound, dtype=np.float64)
		else:
			self.bound = bound
		self.robustness = 0
		if type(A_Matrix) == list:
			self.A_Matrix = np.array(A_Matrix, dtype=np.float64)
		else:
			self.A_Matrix = A_Matrix
		self.horizon = 0
		if type(self.variable_name) != list:
			self.variables = [self.variable_name]
		else:
			self.variables = self.variable_name
		self.thread_pool = thread_pool
		self.process_type = process_type

		self.var_range = None
		self.evaluated_range = None

	def eval_interval(self, traces, time_stamps, param_names=None):
		if type(self.predicate_robustness) != type(None):
			self.robustness = self.predicate_robustness[0]
			return self.predicate_robustness

		# If we have one variable, then trace is just the trace correspoding to
		# this variable. If there are multiple traces, we zip them to create a
		# single vector-valued trace.
		if type(self.variable_name) != list:
			trace = traces[self.variable_name]
		else:
			iterts = [traces[name] for name in self.variable_name]
			trace = np.array(list(zip(*iterts)), dtype=np.float64)

		predicate_robustness = []
		np_A_Matrix = np.array(self.A_Matrix)

		if type(self.robustness_array) != type(None):
			return self.robustness_array

		if type(trace) == list:
			trace = np.array(trace, dtype=np.float64)

		if self.thread_pool == False:
			if ((len(trace.shape) == 1) and (type(self.A_Matrix) == int or type(self.A_Matrix) == float)):
				trace = np.array(trace, dtype=np.float32)
				#                if self.process_type == 'cpu':
				predicate_robustness = py_one_dim_pred_numpy(trace, self.A_Matrix, self.bound)
			# predicate_robustness = py_one_dim_pred(list(trace), self.A_Matrix, self.bound)
			#                elif self.process_type == 'cpu_threaded':
			#                    predicate_robustness = py_one_dim_pred_threaded_numpy(trace, self.A_Matrix, self.bound)
			#                else:
			#                    predicate_robustness = gpubackend.py_one_dim_pred_numpy_gpu(list(trace), self.A_Matrix, self.bound)
			else:

				# traces = np.transpose(np.array(traces)).tolist()
				# predicate_robustness = py_higher_dim(trace_size, n, m, q, l, u, init_A, init_P, traces, length, results)
				# if self.process_type == 'cpu':
				predicate_robustness = solve_polyhedron_test(self.A_Matrix, self.bound, trace)
		# else:
		# predicate_robustness = quadprog_polyhedron.solve_polyhedron_threaded(self.A_Matrix,self.bound,trace)

		# for value in trace:
		# np_value = np.array(value)
		# # if np_value.size == 1 and np_A_Matrix.size == 1:
		# # predicate_robustness.append(value * self.A_Matrix - self.bound)
		# if (self.A_Matrix*np_value <= self.bound).all():            #calculate depth if np_value is in A*x <= b
		# x = cp.Variable(np_value.size)
		# objective = cp.Minimize(cp.norm(x - np_value))
		# constraints = [self.A_Matrix*x >= self.bound]
		# prob = cp.Problem(objective, constraints)
		# predicate_robustness.append(-prob.solve(solver=cp.ECOS)) #ECOS has fastest time of ones tested. Going to add this as an argument eventually
		# else:                                               #calculate distance if np_value is not in A*x <= b
		# x = cp.Variable(np_value.size)
		# objective = cp.Minimize(cp.norm(x - np_value))
		# constraints = [self.A_Matrix*x <= self.bound]
		# prob = cp.Problem(objective, constraints)
		# predicate_robustness.append(prob.solve(solver=cp.ECOS))
		# else:
		# #I wanted to pass the Pool() as an argument but that weirdly causes an error. I then tryied making the thread pool in the object init but the same error happens. Not sure how to make the thread pool creation only happen once if the object is used more than once
		# with Pool(cpu_count()) as p:
		# t0 = time()
		# predicate_robustness = p.map(self.optimize_polyhedron, trace)
		# t1 = time()
		# print('Predicate time: ', t1 - t0)
		# p.close()
		# p.join()
		self.robustness = predicate_robustness[0]
		self.robustness_array = predicate_robustness
		return predicate_robustness

	def reset(self):
		self.robustness_array = None

	pass

class Until(object):
    # no doc
    def eval_interval(self, *args, **kwargs): # real signature unknown
        pass

    def reset(self, *args, **kwargs): # real signature unknown
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

"""
#left_formula has to hold at least until right_formula becomes true, which must hold at the current or a future position. 
class Until:

	def __init__(self,lower_time_bound,upper_time_bound,left_formula,right_formula):
		self.nom = "Until"
		self.left_formula = left_formula
		self.right_formula = right_formula
		self.upper_time_bound = upper_time_bound
		self.lower_time_bound = lower_time_bound
		A = min(self.left_formula.var_range[0], self.right_formula.var_range[0])
		B = max(self.left_formula.var_range[1], self.right_formula.var_range[1])
		self.var_range = [A, B]
		self.horizon = self.upper_time_bound +  max(self.left_formula.horizon, self.right_formula.horizon)
		self.variables = list(set(self.left_formula.variables + self.right_formula.variables))

	
	def eval(self, traces,time):
		i = 0
		#finding the first index
		while  traces.timestamps[i] < self.lower_time_bound:
			i += 1

		min_left_formula = self.var_range[1]
		min_right_formula = self.var_range[1]
		max_right_formula = self.var_range[0]
		#if the time bound is exceed or if the right_formula is True when left_formula is need to be True
		while  i < len(traces.timestamps) and traces.timestamps[i] <= self.upper_time_bound & self.right_formula.eval(traces, traces.timestamps[i]) < 0:
			left_formula_evaluation = self.left_formula.eval(traces, traces.timestamps[i])
			right_formula_evaluation = self.right_formula.eval(traces, traces.timestamps[i])
			#Finding the min of the left_formula
			if min_left_formula > left_formula_evaluation:
				min_left_formula = left_formula_evaluation
			#Finding the max of the right_formula
			if max_right_formula < right_formula_evaluation:
				max_right_formula = right_formula_evaluation
			i += 1

		#if the time bound is exceed
		if traces.timestamps[i] > self.upper_time_bound:
			#return Negative Value
			return max_right_formula
		elif self.right_formula.eval(traces, traces.timestamps[i]) > 0:
			# right_formula need to stay True
			while  traces.timestamps[i] <= self.upper_time_bound:
				#Finding the min of the right_formula
				right_formula_evaluation = self.right_formula.eval(traces, traces.timestamps[i])
				if min_right_formula > right_formula_evaluation:
					min_right_formula = right_formula_evaluation
				i += 1
				return min_right_formula

		return min_left_formula

# left_formula has to be true until and including the point where right_formula first becomes true; if right_formula never becomes true, left_formula must remain true forever.
class Release:
	def __init__(self,lower_time_bound,upper_time_bound,left_formula,right_formula):
		self.nom = "Release"
		self.left_formula = left_formula
		self.right_formula = right_formula
		self.upper_time_bound = upper_time_bound
		self.lower_time_bound = lower_time_bound
		self.var_range = left_formula.var_range
		self.horizon = self.upper_time_bound +  max(self.left_formula.horizon, self.right_formula.horizon)
		self.variables = list(set(self.left_formula.variables + self.right_formula.variables))
	
	def eval(self, traces,time):
		i = 0
		#finding the first index
		while  traces.timestamps[i] < self.lower_time_bound:  
			i += 1

		#initialistion with the higest value
		min_left_formula = self.var_range[1]

		#if the time bound is exceed or if the right_formula is True when left_formula is need to be True
		while  i < len(traces.timestamps) and traces.timestamps[i] <= self.upper_time_bound & self.right_formula.eval(traces, traces.timestamps[i-1]) < 0:
			#evaluate the left formula
			left_formula_evaluation = self.left_formula.eval(traces, traces.timestamps[i])
			#finding the minimum of the left formula
			if min_left_formula > left_formula_evaluation:
				min_left_formula = left_formula_evaluation
			i += 1
		#return the minimum of the left formula
		return min_left_formula
		"""