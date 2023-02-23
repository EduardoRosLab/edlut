#!/usr/bin/python

#import pyedlut.pyedlut as pyedlut
import pyedlut as pyedlut

import matplotlib.pyplot as plt

# Declare the simulation object
simulation = pyedlut.PySimulation_API()

simulation.SetRandomGeneratorSeed(28)

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
# # Get and print all the available Learning Rules in EDLUT
# LearningRuleList = simulation.GetAvailableLearningRules()
# simulation.PrintAvailableLearningRules()

N_synapses=1000
Initial_weight=0.1
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
        log_activity = True,
        output_activity = True)


# Define the synaptic parameters
source_neurons_1 = input_spike_layer_1
target_neurons_1 = []
for i in range(N_synapses):
    target_neurons_1.append(output_layer[0])

source_neurons_2 = []
source_neurons_2.append(input_spike_layer_2[0])
target_neurons_2 = output_layer


## Get the default parameter values of the neuron model
#default_param_lrule = simulation.GetLearningRuleDefParams('STDP');
#print('Default parameters for STDP learning rule:')
#for key in default_param_lrule.keys():
#    print(key)



# Define the learning rule parameters

#Post learning
# Create the learning rule
#learning_rule_name = 'SimetricCosSinSTDPAdditiveKernel'
#learning_rule_name = 'SimetricCosSTDPAdditiveKernel'
#learning_rule_name = 'STDPLS'
#learning_rule_name = 'STDP'
#learning_rule_name = 'TriphasicBufferedKernel'
#learning_rule_name = 'VogelsSTDP'
#learning_rule_name = ''
#Trigger learning
#learning_rule_name = 'ExpBufferedAdditiveKernel'
#learning_rule_name = 'ExpAdditiveKernel'
#learning_rule_name = 'CosAdditiveKernel'
#learning_rule_name = 'SimetricCosAdditiveKernel'
#learning_rule_name = 'SimetricCosBufferedAdditiveKernel'
#learning_rule_name = 'SimetricCosSinAdditiveKernel'
#learning_rule_name = 'SinAdditiveKernel'
#learning_rule_name = 'SinBufferedAdditiveKernel'
#Post and trigger learning_rule_name
#learning_rule_name = 'DopamineSTDP'


#learning_rule = simulation.AddLearningRule(learning_rule_name, simulation.GetLearningRuleDefParams(learning_rule_name))

####################POST LEARNING RULES########################
SimetricCosSinSTDPAdditiveKernel_params = {
        'max_min_dist': 0.050,
        'central_amp':0.01,
        'lateral_amp':-0.005
}
#learning_rule = simulation.AddLearningRule('SimetricCosSinSTDPAdditiveKernel', SimetricCosSinSTDPAdditiveKernel_params)

SimetricCosSTDPAdditiveKernel_params = {
        'tau':0.100,
        'exp':2,
        'fixed_change':0.001,
        'kernel_change':-0.010
}
learning_rule = simulation.AddLearningRule('SimetricCosSTDPAdditiveKernel', SimetricCosSTDPAdditiveKernel_params)

STDP_params = {
        'max_LTP':0.010,
        'tau_LTP':0.100,
        'max_LTD':0.020,
        'tau_LTD':0.100
}
#learning_rule = simulation.AddLearningRule('STDP', STDP_params)
#learning_rule = simulation.AddLearningRule('STDPLS', STDP_params)

TriphasicBufferedKernel_params = {
        'a':0.010,
        'alpha':0.03
}
#learning_rule = simulation.AddLearningRule('TriphasicBufferedKernel', TriphasicBufferedKernel_params)

VogelsSTDP_params = {
        'max_kernel_change':-0.020,
    	'tau_kernel_change':0.020,
    	'const_change':0.010
}
#learning_rule = simulation.AddLearningRule('VogelsSTDP', VogelsSTDP_params)

####################TRIGGER LEARNING RULES########################
ExpBufferedAdditiveKernel_params = {
        'kernel_peak':0.100,
        'fixed_change':0.001,
        'kernel_change':-0.010,
        'init_time':0.050
}
#learning_rule = simulation.AddLearningRule('ExpBufferedAdditiveKernel', ExpBufferedAdditiveKernel_params)

ExpAdditiveKernel_params = {
        'kernel_peak':0.050,
        'fixed_change':0.001,
        'kernel_change':-0.010,
}
#learning_rule = simulation.AddLearningRule('ExpAdditiveKernel', ExpAdditiveKernel_params)

CosAdditiveKernel_params = {
        'tau':0.100,
        'exp':2,
        'fixed_change':0.001,
        'kernel_change':-0.010
}
#learning_rule = simulation.AddLearningRule('CosAdditiveKernel', CosAdditiveKernel_params)
#learning_rule = simulation.AddLearningRule('SimetricCosAdditiveKernel', CosAdditiveKernel_params)
#learning_rule = simulation.AddLearningRule('SimetricCosBufferedAdditiveKernel', CosAdditiveKernel_params)

SimetricCosSinAdditiveKernel_params = {
        'max_min_dist': 0.050,
        'central_amp':0.01,
        'lateral_amp':-0.005
}
#learning_rule = simulation.AddLearningRule('SimetricCosSinAdditiveKernel', SimetricCosSinAdditiveKernel_params)

SinAdditiveKernel_params = {
        'kernel_peak':0.050,
        'fixed_change':0.001,
        'kernel_change':-0.010,
        'exp':2,
}
#learning_rule = simulation.AddLearningRule('SinAdditiveKernel', SinAdditiveKernel_params)
#learning_rule = simulation.AddLearningRule('SinBufferedAdditiveKernel', SinAdditiveKernel_params)

####################POST AND TRIGGER LEARNING RULES########################
DopamineSTDP_params = {
    	'k_plu_hig':1.2,
    	'k_plu_low':-0.3,
    	'tau_plu':0.100,
    	'k_min_hig':0.0,
    	'k_min_low':-0.4,
    	'tau_min':0.100,
    	'tau_eli':0.200,
    	'tau_dop':0.300,
    	'inc_dop':0.100,
    	'dop_max':30.0,
    	'dop_min':5.0,
    	'syn_pre_inc':0.010
}
#learning_rule = simulation.AddLearningRule('DopamineSTDP', DopamineSTDP_params)




#Synaptic parameters
lr_synaptic_params = {
        'weight': Initial_weight,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': learning_rule,
        'trigger_wchange': -1
    }

synaptic_params = {
        'weight': 10.0,
        'max_weight': 100.0,
        'type': 0,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': -1
        #'trigger_wchange': learning_rule
    }

# Create the list of synapses
synaptic_layer_1 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, lr_synaptic_params);
synaptic_layer_2 = simulation.AddSynapticLayer(source_neurons_2, target_neurons_2, synaptic_params);


#
#
# #DEFINE INPUT AND OUTPUT FILE DRIVERS
#simulation.AddFileOutputSpikeActivityDriver('Output_spikes.txt')
#simulation.AddFileOutputMonitorDriver('Output_neuron_state.txt', True)

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

weights = simulation.GetSelectedWeights(synaptic_layer_1)
plt.plot(times,weights,times,reference)
plt.xlabel('time (ms)')
plt.ylabel('Synaptic weight (nS)')
plt.title('learning rule')
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
