#!/usr/bin/python

#import pyedlut.pyedlut as pyedlut
import pyedlut as pyedlut

import matplotlib.pyplot as plt

# Declare the simulation object
simulation = pyedlut.PySimulation_API()

# Print all the available Neuron Models in EDLUT
simulation.PrintAvailableNeuronModels()

#print information about the neuron model that we want to use
simulation.PrintNeuronModelInfo("AdExTimeDrivenModelVector")


# Create the input neuron layers (three input fibers for spikes and, input fiber for current and a sinusoiidal current generator)
input_spike_layer = simulation.AddNeuronLayer(
        num_neurons=3,
        model_name='InputSpikeNeuronModel',
        param_dict={},
        log_activity=False,
        output_activity=True)

poisson_generator_params = {
        'frequency': 50.0,
}

input_poisson_generator_layer = simulation.AddNeuronLayer(
        num_neurons=3,
        model_name='PoissonGeneratorDeviceVector',
        param_dict=poisson_generator_params,
        log_activity=False,
        output_activity=True)

input_current_layer = simulation.AddNeuronLayer(
        num_neurons=1,
        model_name='InputCurrentNeuronModel',
        param_dict={},
        log_activity=False,
        output_activity=False)

sin_params = {
        'frequency': 2.0,
        'amplitude': 100.0,
        'offset': 300.0,
        'phase': 0.0,
}

sin_current_layer = simulation.AddNeuronLayer(
        num_neurons=1,
        model_name='SinCurrentDeviceVector',
        param_dict=sin_params,
        log_activity=False,
        output_activity=False)


info_AdEx_vector = simulation.GetNeuronModelInfo('AdExTimeDrivenModelVector')
simulation.PrintNeuronModelInfo('AdExTimeDrivenModelVector')
AdEx_vector_parameters1 = simulation.GetVectorizableParameters('AdExTimeDrivenModelVector')
simulation.PrintVectorizableParameters('AdExTimeDrivenModelVector')
AdEx_vector_parameters2 = simulation.GetVectorizableParameters('AdExTimeDrivenModel')
simulation.PrintVectorizableParameters('AdExTimeDrivenModel')


# Get the default parameter values of the neuron model
default_params = simulation.GetNeuronModelDefParams('AdExTimeDrivenModelVector');
print('Default parameters for LIF Time-driven model:')
for key, value in zip (default_params.keys(), default_params.values()):
    print(key, value)

# Create the output neuron layer
#default_output_layer = simulation.AddNeuronLayer(
#	num_neurons= 2,
#	model_name= 'LIFTimeDrivenModel',
#	param_dict= default_params,
#	log_activity = False,
#	output_activity = True)

# Get the default parameter values of the integration method
default_im_param = simulation.GetIntegrationMethodDefParams('Euler')
print('Default parameters for Euler integration method:')
for key in default_im_param.keys():
    print(key)

# Define the neuron model parameters for the output layer
integration_method = pyedlut.PyModelDescription(model_name='Euler', params_dict={'step': 0.0001})
output_params = {
        'a': 1.0, #VECTOR
		'b': 9.0, #VECTOR
		'thr_slo_fac': 2.0, #VECTOR
		'v_thr': -50.0, #VECTOR
		'tau_w': 50.0, #VECTOR
		'e_exc': 0.0, #FIXED
		'e_inh': -80.0, #FIXED
		'e_reset': -80.0, #VECTOR
		'e_leak': -65.0, #VECTOR
		'g_leak': 10.0, #VECTOR
		'c_m': 110.0, #VECTOR
		'tau_exc': 5.0, #FIXED
		'tau_inh': 10.0, #FIXED
		'tau_nmda': 20.0, #FIXED
        'int_meth': integration_method
}



# Create the output layer
output_layer = simulation.AddNeuronLayer(
        num_neurons = 8,
        model_name = 'AdExTimeDrivenModelVector',
        param_dict = output_params,
        log_activity = True,
        output_activity = True)

# Get the default parameter values of the neuron model
default_param_lrule = simulation.GetLearningRuleDefParams('STDP');
print('Default parameters for STDP learning rule:')
for key in default_param_lrule.keys():
    print(key)

# Define the learning rule parameters
lrule_params = {
        'max_LTP': 0.016,
        'tau_LTP': 0.010,
        'max_LTD': 0.033,
        'tau_LTD': 0.005
    }

# Create the learning rule
STDP_rule = simulation.AddLearningRule('STDP', lrule_params)

# Define the synaptic parameters
#source_neuron = input_layer[:]
#target_neuron = [2] * output_layer[0]


source_neurons = []
for i in range (3):
	source_neurons.append(input_spike_layer[i])
for i in range (3):
	source_neurons.append(input_poisson_generator_layer[i])
source_neurons.append(input_current_layer[0])
source_neurons.append(sin_current_layer[0])

for i in range (3):
	source_neurons.append(input_spike_layer[i])
for i in range (3):
	source_neurons.append(input_poisson_generator_layer[i])
source_neurons.append(input_current_layer[0])
source_neurons.append(sin_current_layer[0])


target_neurons = []
for i in range (3):
	target_neurons.append(output_layer[0])
for i in range (3):
	target_neurons.append(output_layer[2])
target_neurons.append(output_layer[4])
target_neurons.append(output_layer[6])

for i in range (3):
	target_neurons.append(output_layer[1])
for i in range (3):
	target_neurons.append(output_layer[3])
target_neurons.append(output_layer[5])
target_neurons.append(output_layer[7])


connection_type = [0, 1, 2, 0, 1, 2, 3, 3, 0, 1, 2, 0, 1, 2, 3, 3]
wchange = [STDP_rule, -1, STDP_rule, -1, -1, -1, -1, -1, STDP_rule, -1, STDP_rule, -1, -1, -1, -1, -1]
synaptic_params = {
        'weight': 8.0,
        'max_weight': 100.0,
        'type': connection_type,
        'delay': 0.001,
        'wchange': wchange,
        'trigger_wchange': -1
    }

# Create the list of synapses
synaptic_layer = simulation.AddSynapticLayer(source_neurons, target_neurons, synaptic_params);
print('Synaptic layer:', synaptic_layer)


#DEFINE INPUT AND OUTPUT FILE DRIVERS
simulation.AddFileInputSpikeActivityDriver('Input_spikes.cfg')
simulation.AddFileInputCurrentActivityDriver('Input_currents.cfg')
simulation.AddFileOutputSpikeActivityDriver('Output_spikes.txt')
simulation.AddFileOutputMonitorDriver('Output_state.txt', True)
simulation.AddFileOutputWeightDriver('Output_weights.txt', 0.5)


# Initialize the network
simulation.Initialize()

################ SET FINAL PARAMETER IN AdExTimeDrivenModelVector #####################
output_params1 = {
        'a': 1.5, #VECTOR
		'b': 9.5, #VECTOR
		'thr_slo_fac': 2.5, #VECTOR
		'v_thr': -45.0, #VECTOR
		'tau_w': 40.0, #VECTOR
		'e_reset': -78.0, #VECTOR
		'e_leak': -65.0, #VECTOR
		'g_leak': 12.0, #VECTOR
		'c_m': 120.0, #VECTOR
}
simulation.SetSpecificNeuronParams(output_layer[0],output_params1)
simulation.SetSpecificNeuronParams(output_layer[2],output_params1)
simulation.SetSpecificNeuronParams(output_layer[4],output_params1)
simulation.SetSpecificNeuronParams(output_layer[6],output_params1)

output_params2 = {
        'a': 0.9, #VECTOR
		'b': 8.0, #VECTOR
		'thr_slo_fac': 1.8, #VECTOR
		'v_thr': -50.0, #VECTOR
		'tau_w': 55.0, #VECTOR
		#'e_reset': -80.0, #VECTOR
		#'e_leak': -65.0, #VECTOR
		'g_leak': 9.0, #VECTOR
		#'c_m': 110.0, #VECTOR
}
simulation.SetSpecificNeuronParams(output_layer[1],output_params2)
simulation.SetSpecificNeuronParams(output_layer[3],output_params2)
simulation.SetSpecificNeuronParams(output_layer[5],output_params2)
simulation.SetSpecificNeuronParams(output_layer[7],output_params2)

final_params_vector = simulation.GetSpecificNeuronParams(output_layer)
for i in range (len(output_layer)):
    print('- Neuron ', output_layer[i])
    print (final_params_vector[i])


################ SET FINAL PARAMETER IN PoissonGeneratorDeviceVector #####################
poisson_params_exc ={
        'frequency': 100.0,
}
simulation.SetSpecificNeuronParams(input_poisson_generator_layer[0], poisson_params_exc)

poisson_params_inh ={
        'frequency': 30.0,
}
simulation.SetSpecificNeuronParams(input_poisson_generator_layer[1], poisson_params_inh)

poisson_params_NMDA ={
        'frequency': 50.0,
}
simulation.SetSpecificNeuronParams(input_poisson_generator_layer[2], poisson_params_NMDA)


# Inject input spikes to the network
spikes = { 'times':[], 'neurons':[] }

time = 0.001
while time < 1.0:
    spikes['times'].append(time)
    spikes['neurons'].append(input_spike_layer[0])
    time += 0.005
time = 0.004
while time < 1.0:
    spikes['times'].append(time)
    spikes['neurons'].append(input_spike_layer[1])
    time += 0.050
time = 0.002
while time < 1.0:
    spikes['times'].append(time)
    spikes['neurons'].append(input_spike_layer[2])
    time += 0.010
simulation.AddExternalSpikeActivity(spikes['times'], spikes['neurons'])


# Inject input current to the network
currents = { 'times':[], 'neurons':[], 'currents':[]}
time = 0.010
while time < 1.0:
    currents['times'].append(time)
    currents['neurons'].append(input_current_layer[0])
    currents['currents'].append(300.0)
    time += 0.100
    currents['times'].append(time)
    currents['neurons'].append(input_current_layer[0])
    currents['currents'].append(200.0)
    time += 0.100
simulation.AddExternalCurrentActivity(currents['times'], currents['neurons'], currents['currents'])


# Run the simulation step-by-step
total_simulation_time = 1.0
simulation_step = 0.01
sim_time = 0.0
while sim_time < total_simulation_time:
    simulation.RunSimulation(sim_time + simulation_step)
    sim_time += simulation_step


# Retrieve output spike activity
output_times, output_index = simulation.GetSpikeActivity()


# Print the output spike activity
print('Output activity: ')
for t, i in zip(output_times, output_index):
    print("{:.7}".format(t), i)


# Print the synaptics in a compressed format
N_equal_weights, equal_weights = simulation.GetCompressedWeights()
print('Compressed synaptic weights (N_synapses, weigth): ')
for t, i in zip(N_equal_weights, equal_weights):
    print(t, i)

# Print the synaptics in a extended format
weights = simulation.GetWeights()
print('Extended synaptic weights (index, weigth): ')
for i in range(len(weights)):
    print(i, weights[i])

# Print a selection of synapses in a extended format
indexes = [0, 2, 4]
weights = simulation.GetSelectedWeights(indexes)
print('Extended synaptic weights (index, weigth): ')
for i in range(len(weights)):
    print(indexes[i], weights[i])


print('Simulation finished')

#PLOT RASTERPLOT
rasterplot = plt.scatter(output_times, output_index, alpha=0.5)
plt.xlim(0, total_simulation_time)
plt.xlabel('time (s)')
plt.ylabel('5, 6 = spikes (AMPA, GABA and NMDA);   7, 8 = square current (5 Hz);   9, 10 = sin current (2 Hz)')
plt.show()
