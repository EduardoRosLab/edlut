#!/usr/bin/python

import pyedlut as pyedlut

import matplotlib.pyplot as plt
import numpy as np

# Declare the simulation object
simulation = pyedlut.PySimulation_API()


poisson_generator_params = {
        'frequency': 50.0,
}

n_poisson = 3
input_poisson_generator_layer = simulation.AddNeuronLayer(
    num_neurons = n_poisson,
    model_name = 'PoissonGeneratorDeviceVector',
    param_dict = poisson_generator_params,
    log_activity = False,
    output_activity = True
)

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
    num_neurons = 1,
    model_name = 'AdExTimeDrivenModelVector',
    param_dict = output_params,
    log_activity = True,
    output_activity = True
)


source_neurons = []
target_neurons = []
for i in range(n_poisson):
    source_neurons.append(input_poisson_generator_layer[i])
    target_neurons.append(output_layer[0])


connection_type = [0]*n_poisson #AMPA; GABA, NMDA
wchange = [-1]*n_poisson
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


#DEFINE INPUT AND OUTPUT FILE DRIVERS
#simulation.AddFileInputSpikeActivityDriver('Input_spikes.cfg')
#simulation.AddFileInputCurrentActivityDriver('Input_currents.cfg')
#simulation.AddFileOutputSpikeActivityDriver('Output_spikes.txt')
simulation.AddFileOutputMonitorDriver('Output_state.txt', True)
#simulation.AddFileOutputWeightDriver('Output_weights.txt', 0.5)


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
simulation.SetSpecificNeuronParams(output_layer[0], output_params1)



################ SET FINAL PARAMETER IN PoissonGeneratorDeviceVector #####################
for i in range(n_poisson):
    poisson_params = {'frequency': 10.0 * (i+1)}
    simulation.SetSpecificNeuronParams(input_poisson_generator_layer[i], poisson_params)


# Run the simulation step-by-step
total_simulation_time = 10.0
simulation_bin = 1.0
for sim_time in np.arange(0.0+simulation_bin, total_simulation_time, simulation_bin):
    simulation.RunSimulation(sim_time)

    # Update all poissons every timebin
    for i in range(n_poisson):
        poisson_params = {'frequency': np.random.randint(0,2) * 100.0}
        simulation.SetSpecificNeuronParams(input_poisson_generator_layer[i], poisson_params)

# Retrieve output spike activity
output_times, output_index = simulation.GetSpikeActivity()



print('Simulation finished')

#PLOT RASTERPLOT
rasterplot = plt.scatter(output_times, output_index, alpha=0.5)
plt.xlim(0, total_simulation_time)
plt.xlabel('time (s)')
plt.show()
