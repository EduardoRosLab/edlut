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
contant_lr=-0.00001
kernel_lr=0.0001


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


#Learning rules with teaching signals
ExpBufferedAdditiveKernel_rule = simulation.AddLearningRule('ExpBufferedAdditiveKernel', simulation.GetLearningRuleDefParams('ExpBufferedAdditiveKernel'))
ExpAdditiveKernel_rule = simulation.AddLearningRule('ExpAdditiveKernel', simulation.GetLearningRuleDefParams('ExpAdditiveKernel'))
CosAdditiveKernel_rule = simulation.AddLearningRule('CosAdditiveKernel', simulation.GetLearningRuleDefParams('CosAdditiveKernel'))
SimetricCosBufferedAdditiveKernel_rule = simulation.AddLearningRule('SimetricCosBufferedAdditiveKernel', simulation.GetLearningRuleDefParams('SimetricCosBufferedAdditiveKernel')) ##THIS MODEL FAIL
SimetricCosSinAdditiveKernel_rule = simulation.AddLearningRule('SimetricCosSinAdditiveKernel', simulation.GetLearningRuleDefParams('SimetricCosSinAdditiveKernel'))
SimetricCosAdditiveKernel_rule = simulation.AddLearningRule('SimetricCosAdditiveKernel', simulation.GetLearningRuleDefParams('SimetricCosAdditiveKernel'))
SinBufferedAdditiveKernel_rule = simulation.AddLearningRule('SinBufferedAdditiveKernel', simulation.GetLearningRuleDefParams('SinBufferedAdditiveKernel'))
SinAdditiveKernel_rule = simulation.AddLearningRule('SinAdditiveKernel', simulation.GetLearningRuleDefParams('SinAdditiveKernel'))






# Define the synaptic parameters
source_neurons_1 = input_spike_layer_1
target_neurons_1 = []
for i in range(N_synapses):
    target_neurons_1.append(output_layer[0])

source_neurons_2 = []
source_neurons_2.append(input_spike_layer_2[0])
target_neurons_2 = output_layer


synaptic_params_0 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 0,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': -1
    }

synaptic_params_1 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': ExpBufferedAdditiveKernel_rule,
        'trigger_wchange': -1
    }

synaptic_params_1t = {
        'weight': 0.0,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': ExpBufferedAdditiveKernel_rule
    }

synaptic_params_2 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': ExpAdditiveKernel_rule,
        'trigger_wchange': -1
    }

synaptic_params_2t = {
        'weight': 0.0,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': ExpAdditiveKernel_rule
    }

synaptic_params_3 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': CosAdditiveKernel_rule,
        'trigger_wchange': -1
    }

synaptic_params_3t = {
        'weight': 0.0,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': CosAdditiveKernel_rule
    }

synaptic_params_4 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': SimetricCosBufferedAdditiveKernel_rule,
        'trigger_wchange': -1
    }

synaptic_params_4t = {
        'weight': 0.0,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': SimetricCosBufferedAdditiveKernel_rule
    }

synaptic_params_5 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': SimetricCosSinAdditiveKernel_rule,
        'trigger_wchange': -1
    }

synaptic_params_5t = {
        'weight': 0.0,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': SimetricCosSinAdditiveKernel_rule
    }

synaptic_params_6 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': SimetricCosAdditiveKernel_rule,
        'trigger_wchange': -1
    }

synaptic_params_6t = {
        'weight': 0.0,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': SimetricCosAdditiveKernel_rule
    }

synaptic_params_7 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': SinBufferedAdditiveKernel_rule,
        'trigger_wchange': -1
    }

synaptic_params_7t = {
        'weight': 0.0,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': SinBufferedAdditiveKernel_rule
    }

synaptic_params_8 = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': SinAdditiveKernel_rule,
        'trigger_wchange': -1
    }

synaptic_params_8t = {
        'weight': 0.0,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': SinAdditiveKernel_rule
    }





# Create the list of synapses
synaptic_layer_0 = simulation.AddSynapticLayer(source_neurons_2, target_neurons_2, synaptic_params_0);
synaptic_layer_1 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, synaptic_params_1);
synaptic_layer_1t = simulation.AddSynapticLayer(source_neurons_2, target_neurons_2, synaptic_params_1t);
synaptic_layer_2 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, synaptic_params_2);
synaptic_layer_2t = simulation.AddSynapticLayer(source_neurons_2, target_neurons_2, synaptic_params_2t);
synaptic_layer_3 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, synaptic_params_3);
synaptic_layer_3t = simulation.AddSynapticLayer(source_neurons_2, target_neurons_2, synaptic_params_3t);
synaptic_layer_4 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, synaptic_params_4);
synaptic_layer_4t = simulation.AddSynapticLayer(source_neurons_2, target_neurons_2, synaptic_params_4t);
synaptic_layer_5 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, synaptic_params_5);
synaptic_layer_5t = simulation.AddSynapticLayer(source_neurons_2, target_neurons_2, synaptic_params_5t);
synaptic_layer_6 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, synaptic_params_6);
synaptic_layer_6t = simulation.AddSynapticLayer(source_neurons_2, target_neurons_2, synaptic_params_6t);
synaptic_layer_7 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, synaptic_params_7);
synaptic_layer_7t = simulation.AddSynapticLayer(source_neurons_2, target_neurons_2, synaptic_params_7t);
synaptic_layer_8 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, synaptic_params_8);
synaptic_layer_8t = simulation.AddSynapticLayer(source_neurons_2, target_neurons_2, synaptic_params_8t);




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

weights_1 = simulation.GetSelectedWeights(synaptic_layer_1)
weights_2 = simulation.GetSelectedWeights(synaptic_layer_2)
weights_3 = simulation.GetSelectedWeights(synaptic_layer_3)
weights_4 = simulation.GetSelectedWeights(synaptic_layer_4)
weights_5 = simulation.GetSelectedWeights(synaptic_layer_5)
weights_6 = simulation.GetSelectedWeights(synaptic_layer_6)
weights_7 = simulation.GetSelectedWeights(synaptic_layer_7)
weights_8 = simulation.GetSelectedWeights(synaptic_layer_8)
plt.figure()
plt.plot(times, reference, times, weights_1, times, weights_2, times, weights_3, times, weights_4, times, weights_5, times, weights_6, times, weights_7, times, weights_8)
plt.xlabel('time (ms)')
plt.ylabel('Synaptic weight (nS)')
plt.title('All trigger learning rule')
plt.show()





#
# print('Simulation finished')
#
# #PLOT RASTERPLOT
# rasterplot = plt.scatter(output_times, output_index, alpha=0.5)
# plt.xlim(0, total_simulation_time)
# plt.xlabel('time (s)')
# plt.ylabel('5 = spikes (AMPA, GABA and NMDA);   6 = square current (5 Hz);   7 = sin current (2 Hz)')
# plt.show()
