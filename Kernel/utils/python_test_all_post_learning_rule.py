#!/usr/bin/python

#import pyedlut.pyedlut as pyedlut
import pyedlut as pyedlut

import matplotlib.pyplot as plt

# Declare the simulation object
simulation = pyedlut.PySimulation_API()

# # Get and print all the available Neuron Models in EDLUT
# NeuronModelList = simulation.GetAvailableNeuronModels()
# simulation.PrintAvailableNeuronModels()
#
# #Get and print information about all the Neuron Model in EDLUT
# for i in NeuronModelList:
#     print('-------------------')
#     simulation.PrintNeuronModelInfo(i)
#     #print('-------------------')
#     #NeuronModelInfo = simulation.GetNeuronModelInfo(i)
#     #for name,value in zip(NeuronModelInfo.keys(),NeuronModelInfo.values()):
#     #    print(name,'->',value)
#
#
# # Get and print all the available Integration Methods in CPU for EDLUT
# IntegrationMethodList = simulation.GetAvailableIntegrationMethods()
# simulation.PrintAvailableIntegrationMethods()
#


# Get and print all the available Learning Rules in EDLUT
LearningRuleList = simulation.GetAvailableLearningRules()
simulation.PrintAvailableLearningRules()
print('#############################################################')

#Get and print information about all the Learning Rules in EDLUT
for i in LearningRuleList:
	# Get the default parameter values of the each learning rule
	default_param_lrule = simulation.GetLearningRuleDefParams(i);
	print('Default parameters for:', i)
	for key, value in zip(default_param_lrule.keys(),default_param_lrule.values()):
		print(key, value)
	print('##########################')



N_synapses=1000
Initial_weight=0.0051



# Create the input neuron layers (three input fibers for spikes and, input fiber for current and a sinusoiidal current generator)
input_spike_layer_1 = simulation.AddNeuronLayer(
        num_neurons=N_synapses,
        model_name='InputSpikeNeuronModel',
        param_dict={},
        log_activity=False,
        output_activity=False)

input_spike_layer_2 = simulation.AddNeuronLayer(
        num_neurons=1,
        model_name='InputSpikeNeuronModel',
        param_dict={},
        log_activity=False,
        output_activity=False)

# # Get the default parameter values of the neuron model
# default_params = simulation.GetNeuronModelDefParams('LIFTimeDrivenModel');
# print('Default parameters for LIF Time-driven model:')
# for key, value in zip (default_params.keys(), default_params.values()):
#     print(key, value)
#
# # Create the output neuron layer
# #default_output_layer = simulation.AddNeuronLayer(
# #	num_neurons= 2,
# #	model_name= 'LIFTimeDrivenModel',
# #	param_dict= default_params,
# #	log_activity = False,
# #	output_activity = True)
#
# # Get the default parameter values of the integration method
# default_im_param = simulation.GetIntegrationMethodDefParams('Euler')
# print('Default parameters for Euler integration method:')
# for key in default_im_param.keys():
#     print(key)

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
        num_neurons = 1,
        model_name = 'LIFTimeDrivenModel',
        param_dict = output_params,
        log_activity = False,
        output_activity = True)

# Get the default parameter values of the neuron model
default_param_lrule = simulation.GetLearningRuleDefParams('ExpAdditiveKernel');
print('Default parameters for ExpAdditiveKernel learning rule:')
for key in default_param_lrule.keys():
    print(key)


#Learning rules without teaching signals
SimetricCosSinSTDPAdditiveKernel_rule = simulation.AddLearningRule('SimetricCosSinSTDPAdditiveKernel', simulation.GetLearningRuleDefParams('SimetricCosSinSTDPAdditiveKernel'))
SimetricCosSTDPAdditiveKernel_rule = simulation.AddLearningRule('SimetricCosSTDPAdditiveKernel', simulation.GetLearningRuleDefParams('SimetricCosSTDPAdditiveKernel'))
STDPLS_rule = simulation.AddLearningRule('STDPLS', simulation.GetLearningRuleDefParams('STDPLS'))
STDP_rule = simulation.AddLearningRule('STDP', simulation.GetLearningRuleDefParams('STDP'))






# Define the synaptic parameters
source_neurons_1 = input_spike_layer_1
target_neurons_1 = []
for i in range(N_synapses):
    target_neurons_1.append(output_layer[0])

source_neurons_2 = []
source_neurons_2.append(input_spike_layer_2[0])
target_neurons_2 = output_layer


synaptic_params_0 = {
        'weight': 10.0,
        'max_weight': 100.0,
        'type': 0,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': -1
    }

synaptic_params_9 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': SimetricCosSinSTDPAdditiveKernel_rule,
        'trigger_wchange': -1
    }

synaptic_params_10 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': SimetricCosSTDPAdditiveKernel_rule,
        'trigger_wchange': -1
    }

synaptic_params_11 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': STDPLS_rule,
        'trigger_wchange': -1
    }

synaptic_params_12 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': STDP_rule,
        'trigger_wchange': -1
    }







# Create the list of synapses
synaptic_layer_0 = simulation.AddSynapticLayer(source_neurons_2, target_neurons_2, synaptic_params_0);
synaptic_layer_9 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, synaptic_params_9);
synaptic_layer_10 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, synaptic_params_10);
synaptic_layer_11 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, synaptic_params_11);
synaptic_layer_12 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, synaptic_params_12);






# Initialize the network
simulation.Initialize()


# Inject input spikes to the network
spikes = { 'times':[], 'neurons':[] }
times = []
reference = []

step = 0.001
for i in range(N_synapses):
    times.append(i+step)
    reference.append(Initial_weight)
    spikes['times'].append(i*step)
    spikes['neurons'].append(input_spike_layer_1[i])
#first output spike
spikes['times'].append(N_synapses*step*0.25)
spikes['neurons'].append(input_spike_layer_2[0])
#second output spike
spikes['times'].append(N_synapses*step*0.75)
spikes['neurons'].append(input_spike_layer_2[0])


simulation.AddExternalSpikeActivity(spikes['times'], spikes['neurons'])

# Run the simulation step-by-step
simulation.RunSimulation(N_synapses*step*1.1)

weights_9 = simulation.GetSelectedWeights(synaptic_layer_9)
weights_10 = simulation.GetSelectedWeights(synaptic_layer_10)
weights_11 = simulation.GetSelectedWeights(synaptic_layer_11)
weights_12 = simulation.GetSelectedWeights(synaptic_layer_12)

plt.figure()
plt.plot(times, reference, times, weights_9, times, weights_10, times, weights_11, times, weights_12)
plt.xlabel('time (ms)')
plt.ylabel('Synaptic weight (nS)')
plt.title('All post learning rules')
plt.show()




print('Simulation finished')
#PLOT RASTERPLOT
# Retrieve output spike activity
output_times, output_index = simulation.GetSpikeActivity()
rasterplot = plt.scatter(output_times, output_index, alpha=0.5)
#plt.xlim(0, total_simulation_time)
plt.xlabel('time (s)')
#plt.ylabel('5 = spikes (AMPA, GABA and NMDA);   6 = square current (5 Hz);   7 = sin current (2 Hz)')
plt.show()
