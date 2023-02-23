#!/usr/bin/python

#import pyedlut.pyedlut as pyedlut
import pyedlut as pyedlut

import matplotlib.pyplot as plt

# Declare the simulation object
simulation = pyedlut.PySimulation_API()

# Get and print all the available Neuron Models in EDLUT
NeuronModelList = simulation.GetAvailableNeuronModels()
simulation.PrintAvailableNeuronModels()

#Get and print information about all the Neuron Model in EDLUT
for i in NeuronModelList:
    print('-------------------')
    simulation.PrintNeuronModelInfo(i)
    #print('-------------------')
    #NeuronModelInfo = simulation.GetNeuronModelInfo(i)
    #for name,value in zip(NeuronModelInfo.keys(),NeuronModelInfo.values()):
    #    print(name,'->',value)


# Get and print all the available Integration Methods in CPU for EDLUT
IntegrationMethodList = simulation.GetAvailableIntegrationMethods()
simulation.PrintAvailableIntegrationMethods()

# Get and print all the available Learning Rules in EDLUT
LearningRuleList = simulation.GetAvailableLearningRules()
simulation.PrintAvailableLearningRules()

# Create the input neuron layers (three input fibers for spikes and, input fiber for current and a sinusoiidal current generator)
input_spike_layer = simulation.AddNeuronLayer(
        num_neurons=3,
        model_name='InputSpikeNeuronModel',
        param_dict={},
        log_activity=False,
        output_activity=False)

input_current_layer = simulation.AddNeuronLayer(
        num_neurons=1,
        model_name='InputCurrentNeuronModel',
        param_dict={},
        log_activity=False,
        output_activity=False)

sin_params = {
        'frequency': 2.0,
        'amplitude': 2.0,
        'offset': 7.0,
        'phase': 0.0,
}

sin_current_layer = simulation.AddNeuronLayer(
        num_neurons=1,
        model_name='SinCurrentDeviceVector',
        param_dict=sin_params,
        log_activity=False,
        output_activity=False)


# Get the default parameter values of the neuron model
default_params = simulation.GetNeuronModelDefParams('LIFTimeDrivenModel');
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
        'tau_ref': 1.0,
        'v_thr': -40.0,
        'e_exc': 0.0,
        'e_inh': -80.0,
        'e_leak': -65.0,
        'g_leak': 0.2,
        'c_m': 2.0,
        'tau_exc': 0.5,
        'tau_inh': 10.0,
        'tau_nmda': 15.0,
        'int_meth': integration_method
}



# Create the output layer
output_layer = simulation.AddNeuronLayer(
        num_neurons = 3,
        model_name = 'LIFTimeDrivenModel',
        param_dict = output_params,
        log_activity = False,
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
source_neurons.append(input_current_layer[0])
source_neurons.append(sin_current_layer[0])

target_neurons = []
for i in range (3):
	target_neurons.append(output_layer[0])
target_neurons.append(output_layer[1])
target_neurons.append(output_layer[2])



connection_type = [0, 1, 2, 3, 3]
wchange = [STDP_rule, -1, STDP_rule, -1, -1, ]
synaptic_params = {
        'weight': 1.0,
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
simulation.AddFileOutputSpikeActivityDriver('Output_spikes.txt')

# Initialize the network
simulation.Initialize()


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
    currents['currents'].append(8.0)
    time += 0.100
    currents['times'].append(time)
    currents['neurons'].append(input_current_layer[0])
    currents['currents'].append(5.0)
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
plt.ylabel('5 = spikes (AMPA, GABA and NMDA);   6 = square current (5 Hz);   7 = sin current (2 Hz)')
plt.show()
