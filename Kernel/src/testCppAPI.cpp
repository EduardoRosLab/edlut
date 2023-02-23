/***************************************************************************
 *                           testCppAPI.cpp                                *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Jesus Garrido                        *
 * email                : jesusgarrido@ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/***************************************************************************
 *   This program is an example on network simulations with EDLUT using    *
 *   the C++ API class.                                                    *
 ***************************************************************************/

#include "simulation/Simulation_API.h"

#include "../include/communication/ConnectionException.h"

#include <map>
#include <vector>
#include <boost/any.hpp>
#include <simulation/NetworkDescription.h>

int main(int ac, char *av[]) {

	try {
		// Declare the simulation object
		Simulation_API simulation;

		//Set multithread option
		std::map<std::string, boost::any> simulation_parameters;
		simulation_parameters["num_simulation_queues"] = int(4);
		simulation.SetSimulationParameters(simulation_parameters);


		// Get and print the available Neuron Models in EDLUT
		std::vector<std::string> availableNeuronModels = simulation.GetAvailableNeuronModels();
		Simulation_API::PrintAvailableNeuronModels();

		//Print information about each neuron model
		for (std::vector<std::string>::const_iterator it = availableNeuronModels.begin(); it != availableNeuronModels.end(); ++it){
			Simulation_API::PrintNeuronModelInfo(*it);
		}


		// Get and print the available Integration method in CPU for EDLUT
		std::vector<std::string> availableIntegrationMethods = simulation.GetAvailableIntegrationMethods();
		Simulation_API::PrintAvailableIntegrationMethods();


		// Get and print the available Learning Rules in EDLUT
		std::vector<std::string> availableLearningRules = simulation.GetAvailableLearningRules();
		Simulation_API::PrintAvailableLearningRules();


		// Get the default parameter values of InputSpikeNeuronModel
		std::map<std::string, boost::any> default_param_input_spikes = simulation.GetNeuronModelDefParams("InputSpikeNeuronModel");
		std::cout << "Default parameters for InputSpikeNeuronModel: " << std::endl;
		for (std::map<std::string, boost::any>::const_iterator it = default_param_input_spikes.begin(); it != default_param_input_spikes.end(); ++it){
			std::cout << it->first << std::endl;
		}

		// Create the input spike neuron layer
		std::vector<int> input_spike_layer = simulation.AddNeuronLayer(3,"InputSpikeNeuronModel",std::map<std::string,boost::any>(), true, true);


		// Get the default parameter values of PoissonGeneratorDeviceVector
		std::map<std::string, boost::any> default_param_poisson_generator = simulation.GetNeuronModelDefParams("PoissonGeneratorDeviceVector");
		std::cout << "Default parameters for PoissonGeneratorDeviceVector: " << std::endl;
		for (std::map<std::string, boost::any>::const_iterator it = default_param_poisson_generator.begin(); it != default_param_poisson_generator.end(); ++it){
			std::cout << it->first << std::endl;
		}

		// Define the PoissonGeneratorParameters
		std::map<std::string, boost::any> poisson_generator_params;
		poisson_generator_params["frequency"] = float(50.0);

		// Create the input poisson generator layer
		std::vector<int> input_poisson_generator_layer = simulation.AddNeuronLayer(3, "PoissonGeneratorDeviceVector", poisson_generator_params, true, true);



		// Get the default parameter values of InputCurrentNeuronModel
		std::map<std::string, boost::any> default_param_input_currents = simulation.GetNeuronModelDefParams("InputCurrentNeuronModel");
		std::cout << "Default parameters for InputCurrentNeuronModel: " << std::endl;
		for (std::map<std::string, boost::any>::const_iterator it = default_param_input_currents.begin(); it != default_param_input_currents.end(); ++it){
			std::cout << it->first << std::endl;
		}

		// Create the input current neuron layer
		std::vector<int> input_current_layer = simulation.AddNeuronLayer(1, "InputCurrentNeuronModel", std::map<std::string, boost::any>(), true, true);


		// Get the default parameter values of SinCurrentDeviceVector
		std::map<std::string, boost::any> default_param_sin_currents = simulation.GetNeuronModelDefParams("SinCurrentDeviceVector");
		std::cout << "Default parameters for SinCurrentDeviceVector: " << std::endl;
		for (std::map<std::string, boost::any>::const_iterator it = default_param_sin_currents.begin(); it != default_param_sin_currents.end(); ++it){
			std::cout << it->first << std::endl;
		}

		// Define the SinCurrentParameters
		std::map<std::string, boost::any> sin_current_params;
		sin_current_params["frequency"] = float(1.0);
		sin_current_params["amplitude"] = float(1.0);
		sin_current_params["offset"] = float(7.0);
		sin_current_params["phase"] = float(-0.0);

		// Create the sinusoidal current generator
		std::vector<int> sin_current_layer = simulation.AddNeuronLayer(1, "SinCurrentDeviceVector", sin_current_params, true, true);


		// Get the default parameter values of the neuron model
		std::map<std::string, boost::any> default_param = simulation.GetNeuronModelDefParams("LIFTimeDrivenModel");
		std::cout << "Default parameters for LIF Time-driven model: " << std::endl;
		for (std::map<std::string,boost::any>::const_iterator it=default_param.begin(); it!=default_param.end(); ++it){
			std::cout << it->first << std::endl;
		}

		// Get the default parameter values of the integration method
		std::map<std::string, boost::any> default_im_param = simulation.GetIntegrationMethodDefParams("Euler");
		std::cout << "Default parameters for Euler integration method: " << std::endl;
		for (std::map<std::string,boost::any>::const_iterator it=default_im_param.begin(); it!=default_im_param.end(); ++it){
			std::cout << it->first << std::endl;
		}

		// Define the neuron model parameters for the output layer
		std::map<std::string,boost::any> output_params;
		output_params["tau_ref"] = float(1.0);
		output_params["v_thr"] = float(-40.0);
		output_params["e_exc"] = float(0.0);
		output_params["e_inh"] = float(-80.0);
		output_params["e_leak"] = float(-65.0);
		output_params["g_leak"] = float(0.2);
		output_params["c_m"] = float(2.0);
		output_params["tau_exc"] = float(0.5);
		output_params["tau_inh"] = float(10.0);
		output_params["tau_nmda"] = float(15.0);

		ModelDescription IntegrationMethod;
		IntegrationMethod.model_name = "Euler";
		IntegrationMethod.param_map["step"] = float(1.0e-4);

		output_params["int_meth"] = IntegrationMethod;

		// Create the output layer
	//	std::vector<int> input_layer = simulation.AddNeuronLayer(2, "LIFTimeDrivenModel",default_param, true, true);
		std::vector<int> output_layer_LIF = simulation.AddNeuronLayer(4,"LIFTimeDrivenModel",output_params,true,true);


		// Define a neuron model with vectorized parameters that can be independently fixed after the initialization
		std::map<std::string, std::string> total_AdEx_info = simulation.GetNeuronModelInfo("AdExTimeDrivenModelVector");
		simulation.PrintNeuronModelInfo("AdExTimeDrivenModelVector");
		std::map<std::string, std::string> vector_AdEx_info1 = simulation.GetVectorizableParameters("AdExTimeDrivenModelVector");
		simulation.PrintVectorizableParameters("AdExTimeDrivenModelVector");
		std::map<std::string, std::string> vector_AdEx_info2 = simulation.GetVectorizableParameters("AdExTimeDrivenModel");
		simulation.PrintVectorizableParameters("AdExTimeDrivenModel");


		std::map<std::string, boost::any> default_param2 = simulation.GetNeuronModelDefParams("AdExTimeDrivenModelVector");
		std::map<std::string, boost::any> original_AdEx_param;
		original_AdEx_param["a"] = float(1.0f); //conductance (nS): VECTOR
		original_AdEx_param["b"] = float(9.0f); //spike trigger adaptation (pA): VECTOR
		original_AdEx_param["thr_slo_fac"] = float(2.0f); //threshold slope factor (mV): VECTOR
		original_AdEx_param["v_thr"] = float(-50.0f); //effective threshold potential (mV): VECTOR
		original_AdEx_param["tau_w"] = float(50.0f); //adaptation time constant (ms): VECTOR
		original_AdEx_param["e_exc"] = float(0.0f); //excitatory reversal potential (mV): FIXED
		original_AdEx_param["e_inh"] = float(-80.0f); //inhibitory reversal potential (mV): FIXED
		original_AdEx_param["e_reset"] = float(-80.0f); //reset potential (mV): VECTOR
		original_AdEx_param["e_leak"] = float(-65.0f); //effective leak potential (mV): VECTOR
		original_AdEx_param["g_leak"] = float(10.0f); //leak conductance (nS): VECTOR
		original_AdEx_param["c_m"] = float(110.0f); //membrane capacitance (pF): VECTOR
		original_AdEx_param["tau_exc"] = float(5.0f); //AMPA (excitatory) receptor time constant (ms): FIXED
		original_AdEx_param["tau_inh"] = float(10.0f); //GABA (inhibitory) receptor time constant (ms): FIXED
		original_AdEx_param["tau_nmda"] = float(20.0f); //NMDA (excitatory) receptor time constant (ms): FIXED
		original_AdEx_param["int_meth"] = IntegrationMethod;
		std::vector<int> output_layer_AdEx = simulation.AddNeuronLayer(4, "AdExTimeDrivenModelVector", original_AdEx_param, true, true);



		// Get the default parameter values of the learning rule
		std::map<std::string, boost::any> default_param_lrule = simulation.GetLearningRuleDefParams("STDP");
		std::cout << "Default parameters for STDP learning rule: " << std::endl;
		for (std::map<std::string,boost::any>::const_iterator it=default_param_lrule.begin(); it!=default_param_lrule.end(); ++it){
			std::cout << it->first << std::endl;
		}

		// Define the learning rule parameters
		std::map<std::string,boost::any> lrule_params;
		lrule_params["max_LTP"] = float(0.001);
		lrule_params["tau_LTP"] = float(0.010);
		lrule_params["max_LTD"] = float(0.005);
		lrule_params["tau_LTD"] = float(0.005);

		// Create the learning rule
		int STDP_rule = simulation.AddLearningRule("STDP", lrule_params);

	 //   // Define the synaptic parameters
		//////////////////////////////////////////////////////////////////////////////////////////////////
		////Source neurons
		//std::vector<int> exc_source_neurons(2, input_spike_layer[0]); //excitatory output synapses
		//std::vector<int> inh_source_neurons(2, input_spike_layer[1]); //inhibitory output synapses
		//std::vector<int> nmda_source_neurons(2, input_spike_layer[2]); //excitatory NMDA output synapses
		//std::vector<int> ext_I_source_neurons(2, input_current_layer[0]); //current output synapses

		//std::vector<int> target_neurons;
		//target_neurons.push_back(output_layer[0]);
		//target_neurons.push_back(output_layer[1]);

	 //   std::map<std::string,boost::any> exc_synaptic_params;
		//exc_synaptic_params["weight"] = float(1.0);
		//exc_synaptic_params["max_weight"] = float(100.0);
		//exc_synaptic_params["type"] = int(0);
		//exc_synaptic_params["delay"] = float(0.001);
		//exc_synaptic_params["wchange"] = STDP_rule;
	 //   // exc_synaptic_params["trigger_wchange"] = -1

		//std::map<std::string, boost::any> inh_synaptic_params;
		//inh_synaptic_params["weight"] = float(1.0);
		//inh_synaptic_params["max_weight"] = float(100.0);
		//inh_synaptic_params["type"] = int(1);
		//inh_synaptic_params["delay"] = float(0.001);
		//inh_synaptic_params["wchange"] = int(-1);
		//// inh_synaptic_params["trigger_wchange"] = -1

		//std::map<std::string, boost::any> nmda_synaptic_params;
		//nmda_synaptic_params["weight"] = float(1.0);
		//nmda_synaptic_params["max_weight"] = float(100.0);
		//nmda_synaptic_params["type"] = int(2);
		//nmda_synaptic_params["delay"] = float(0.001);
		//nmda_synaptic_params["wchange"] = int(-1);
		//// synaptic_params["trigger_wchange"] = -1

		//std::map<std::string, boost::any> ext_I_synaptic_params;
		//ext_I_synaptic_params["weight"] = float(1.0);
		//ext_I_synaptic_params["max_weight"] = float(100.0);
		//ext_I_synaptic_params["type"] = int(3);
		//ext_I_synaptic_params["delay"] = float(0.001);
		//ext_I_synaptic_params["wchange"] = int(-1);
		//// synaptic_params["trigger_wchange"] = -1

	 //   // Create the list of synapses
	 //   std::vector<int> exc_synaptic_layer = simulation.AddSynapticLayer(exc_source_neurons, target_neurons, exc_synaptic_params);
		//std::vector<int> inh_synaptic_layer = simulation.AddSynapticLayer(inh_source_neurons, target_neurons, inh_synaptic_params);
		//std::vector<int> nmda_synaptic_layer = simulation.AddSynapticLayer(nmda_source_neurons, target_neurons, nmda_synaptic_params);
		//std::vector<int> ext_I_synaptic_layer = simulation.AddSynapticLayer(ext_I_source_neurons, target_neurons, ext_I_synaptic_params);
		//////////////////////////////////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////////////////////////////////
		//Source neurons
		std::vector<int> source_neurons;
		//Source neuron for LIF target neurons
		for (int i = 0; i < 3; i++){
			source_neurons.push_back(input_spike_layer[i]);
		}
		for (int i = 0; i < 3; i++){
			source_neurons.push_back(input_poisson_generator_layer[i]);
		}
		source_neurons.push_back(input_current_layer[0]);
		source_neurons.push_back(sin_current_layer[0]);
		//Source neuron for AdEx target neurons
		for (int i = 0; i < 3; i++){
			source_neurons.push_back(input_spike_layer[i]);
		}
		for (int i = 0; i < 3; i++){
			source_neurons.push_back(input_poisson_generator_layer[i]);
		}
		source_neurons.push_back(input_current_layer[0]);
		source_neurons.push_back(sin_current_layer[0]);


		//target neurons
		std::vector<int> target_neurons;
		//LIF
		for (int i = 0; i < 3; i++){
			target_neurons.push_back(output_layer_LIF[0]);
		}
		for (int i = 0; i < 3; i++){
			target_neurons.push_back(output_layer_LIF[1]);
		}
		target_neurons.push_back(output_layer_LIF[2]);
		target_neurons.push_back(output_layer_LIF[3]);
		//AdEx
		for (int i = 0; i < 3; i++){
			target_neurons.push_back(output_layer_AdEx[0]);
		}
		for (int i = 0; i < 3; i++){
			target_neurons.push_back(output_layer_AdEx[1]);
		}
		target_neurons.push_back(output_layer_AdEx[2]);
		target_neurons.push_back(output_layer_AdEx[3]);



		//synaptic parameters (some parameters just define a value and other ones define a vector of values)
		std::vector<int> type;
		//0 = AMPA, 1 = GABA, 2 = NMDA  (LIF)
		for (int i = 0; i < 3; i++){
			type.push_back(i);
		}
		for (int i = 0; i < 3; i++){
			type.push_back(i);
		}
		//3 = EXT_I
		for (int i = 0; i < 2; i++){
			type.push_back(3);
		}

		//0 = AMPA, 1 = GABA, 2 = NMDA  (AdEx)
		for (int i = 0; i < 3; i++){
			type.push_back(i);
		}
		for (int i = 0; i < 3; i++){
			type.push_back(i);
		}
		//3 = EXT_I
		for (int i = 0; i < 2; i++){
			type.push_back(3);
		}


		std::vector<int> wchange;
		wchange.push_back(STDP_rule);
		wchange.push_back(STDP_rule);
		wchange.push_back(STDP_rule);
		for (int i = 3; i < 16; i++){
			wchange.push_back(-1);
		}
		std::map<std::string, boost::any> synaptic_params;
		synaptic_params["weight"] = float(1.0); // THIS PARAMETER IS OPTIONAL (AUTOMATICALLY INITILIZED TO 1)
		synaptic_params["max_weight"] = float(100.0); // THIS PARAMETER IS OPTIONAL (AUTOMATICALLY INITILIZED TO 1)
		synaptic_params["type"] = type; // THIS PARAMETER IS OPTIONAL (AUTOMATICALLY INITILIZED TO 0)
		synaptic_params["delay"] = float(0.001f); // THIS PARAMETER IS OPTIONAL (AUTOMATICALLY INITILIZED TO 0.001)
		synaptic_params["wchange"] = wchange; // THIS PARAMETER IS OPTIONAL (AUTOMATICALLY INITILIZED TO -1)
		synaptic_params["trigger_wchange"] = -1; // THIS PARAMETER IS OPTIONAL (AUTOMATICALLY INITILIZED TO -1)


		// Create the list of synapses
		std::vector<int> synaptic_layer = simulation.AddSynapticLayer(source_neurons, target_neurons, synaptic_params);
		////////////////////////////////////////////////////////////////////////////////////////////////



		/*
		for (std::vector<int>::const_iterator v_it=synaptic_layer.begin(); v_it!=synaptic_layer.end(); ++v_it) {
			std::cout << *v_it << ", ";
		}
		std::cout << std::endl;
		 */



		//DEFINE INPUT AND OUTPUT FILE DRIVERS
		simulation.AddFileInputSpikeActivityDriver("../utils/Input_spikes.cfg");
		simulation.AddFileInputCurrentActivityDriver("../utils/Input_currents.cfg");
		simulation.AddFileOutputSpikeActivityDriver("Output_spikes.cfg");
		simulation.AddFileOutputMonitorDriver("Output_states.cfg", true);
		simulation.AddFileOutputWeightDriver("Output_weights.cfg", 0.5f);




		// Initialize the network
		simulation.Initialize();

		////////////////////////////SET ADEXTIMEDRIVENMODELVECTO PARAMETERS///////////////////////////////
		std::map<std::string, boost::any> modified_AdEx_param1;
		modified_AdEx_param1["a"] = float(1.5f); //conductance (nS): VECTOR
		modified_AdEx_param1["b"] = float(9.5f); //spike trigger adaptation (pA): VECTOR
		modified_AdEx_param1["thr_slo_fac"] = float(2.5f); //threshold slope factor (mV): VECTOR
		modified_AdEx_param1["v_thr"] = float(-45.0f); //effective threshold potential (mV): VECTOR
		modified_AdEx_param1["tau_w"] = float(40.0f); //adaptation time constant (ms): VECTOR
		modified_AdEx_param1["e_reset"] = float(-78.0f); //reset potential (mV): VECTOR
		modified_AdEx_param1["e_leak"] = float(-60.0f); //effective leak potential (mV): VECTOR
		modified_AdEx_param1["g_leak"] = float(12.0f); //leak conductance (nS): VECTOR
		modified_AdEx_param1["c_m"] = float(100.0f); //membrane capacitance (pF): VECTOR
		simulation.SetSpecificNeuronParams(output_layer_AdEx[0], modified_AdEx_param1);

		std::map<std::string, boost::any> modified_AdEx_param2;
		modified_AdEx_param2["a"] = float(1.2f); //conductance (nS): VECTOR
		modified_AdEx_param2["b"] = float(9.1f); //spike trigger adaptation (pA): VECTOR
		//modified_AdEx_param2["thr_slo_fac"] = float(2.0f); //threshold slope factor (mV): VECTOR
		modified_AdEx_param2["v_thr"] = float(-40.0f); //effective threshold potential (mV): VECTOR
		modified_AdEx_param2["tau_w"] = float(60.0f); //adaptation time constant (ms): VECTOR
		//modified_AdEx_param2["e_reset"] = float(-80.0f); //reset potential (mV): VECTOR
		//modified_AdEx_param2["e_leak"] = float(-65.0f); //effective leak potential (mV): VECTOR
		modified_AdEx_param2["g_leak"] = float(9.0f); //leak conductance (nS): VECTOR
		modified_AdEx_param2["c_m"] = float(120.0f); //membrane capacitance (pF): VECTOR
		simulation.SetSpecificNeuronParams(output_layer_AdEx[1], modified_AdEx_param2);

		cout << "FINAL PARAMETER FOR AdExTimeDrivenModelVector MODEL" << endl;
		std::vector<std::map<std::string, boost::any> > final_AdEx_param = simulation.GetSpecificNeuronParams(output_layer_AdEx);
		for (std::vector<std::map<std::string, boost::any> >::iterator it = final_AdEx_param.begin(); it != final_AdEx_param.end(); ++it){
			for (std::map<std::string, boost::any>::iterator it2 = it->begin(); it2 != it->end(); ++it2){
				if (it2->first != string("int_meth") && it2->first != string("name")){
					cout << it2->first << ": " << boost::any_cast<float>(it2->second) << endl;
				}
			}
			cout << "----------------------------------------------" << endl;
		}

		////////////////////////////SET POISSONGENERATORDEVECEVECTOR PARAMETERS///////////////////////////////
		std::map<std::string, boost::any> modified_poissong_param_exc;
		modified_poissong_param_exc["frequency"] = float(70.0f); //conductance (nS): VECTOR
		simulation.SetSpecificNeuronParams(input_poisson_generator_layer[0], modified_poissong_param_exc);

		std::map<std::string, boost::any> modified_poissong_param_inh;
		modified_poissong_param_inh["frequency"] = float(30.0f); //conductance (nS): VECTOR
		simulation.SetSpecificNeuronParams(input_poisson_generator_layer[1], modified_poissong_param_inh);

		std::map<std::string, boost::any> modified_poissong_param_NMDA;
		modified_poissong_param_NMDA["frequency"] = float(15.0f); //conductance (nS): VECTOR
		simulation.SetSpecificNeuronParams(input_poisson_generator_layer[2], modified_poissong_param_NMDA);


		// Inject input spikes to the network
		std::vector<double> spike_times;
		std::vector<long int> spike_neurons;
		//spikes in excitatory input spike neuron
		for (float time=0.001; time<1.0; time+=0.005){
			spike_times.push_back(time);
			spike_neurons.push_back(input_spike_layer[0]);
		}
		//spikes in inhibitory input spike neuron
		for (float time=0.004; time<1.0; time+=0.050){
			spike_times.push_back(time);
			spike_neurons.push_back(input_spike_layer[1]);
		}
		//spikes in excitatory NMDA input spike neuron
		for (float time = 0.002; time<1.0; time += 0.010){
			spike_times.push_back(time);
			spike_neurons.push_back(input_spike_layer[2]);
		}
		simulation.AddExternalSpikeActivity(spike_times,spike_neurons);

		// Inject input currents to the network
		std::vector<double> current_times;
		std::vector<long int> current_neurons;
		std::vector<float> current_values;
		for (float time = 0.010; time<1.0; time += 0.200){
			current_times.push_back(time);
			current_neurons.push_back(input_current_layer[0]);
			current_values.push_back(2.0);

			current_times.push_back(time + 0.100);
			current_neurons.push_back(input_current_layer[0]);
			current_values.push_back(-1.0);
		}
		simulation.AddExternalCurrentActivity(current_times, current_neurons, current_values);





		// Run the simulation step-by-step
		float total_simulation_time = 1.0;
		float simulation_step = 0.01;
		for (float sim_time = 0; sim_time<total_simulation_time; sim_time+=simulation_step){
			simulation.RunSimulation(sim_time+simulation_step);
		}

		// Retrieve output spike activity
		std::vector<double> output_times;
		std::vector<long> output_index;
		simulation.GetSpikeActivity(output_times, output_index);

		// Print the output spike activity
		std::cout << "Output activity (time, neuron_index):" << std::endl;
		for (unsigned int index = 0; index<output_times.size(); ++index){
			std::cout << output_times[index] << "\t" << output_index[index] << std::endl;
		}

		// Print all the synaptics in a compressed format
		std::vector<int> N_equal_weights;
		std::vector<float> equal_weights;
		simulation.GetCompressedWeights(N_equal_weights, equal_weights);
		std::cout << "All compressed synaptic weights (N_synapses, weigth):" << std::endl;
		for (unsigned int index = 0; index<N_equal_weights.size(); ++index){
			std::cout << N_equal_weights[index] << "\t" << equal_weights[index] << std::endl;
		}

		// Print all the synaptics in a extended format
		std::vector<float> weights = simulation.GetWeights();
		std::cout << "All extended synaptic weights (index, weigth):" << std::endl;
		for (unsigned int index = 0; index<weights.size(); ++index){
			std::cout << index << "\t" << weights[index] << std::endl;
		}

		// Print a selection of synaptic weights
	  int tmp[] = { 0, 2, 4, 7, 1};
	  std::vector<int> indexes( tmp, tmp+5 );

		std::vector<float> weights2 = simulation.GetSelectedWeights(indexes);
		std::cout << "Selected synaptic weights (index, weigth):" << std::endl;
		for (unsigned int index = 0; index<weights2.size(); ++index){
			std::cout << indexes[index] << "\t" << weights2[index] << std::endl;
		}



		std::cout << "Simulation finished" << std::endl;

		/*
		std::vector<std::map<std::string,boost::any> > v = simulation.GetNeuronParams(std::vector<int>(1));
		for (std::vector<std::map<std::string,boost::any> >::const_iterator it1=v.begin(); it1!=v.end(); ++it1) {
			for (std::map<std::string, boost::any>::const_iterator it2 = it1->begin(); it2 != it1->end(); ++it2) {
				std::cout << it2->first << std::endl;
			}
		}
		 */
	}
	catch (ConnectionException Exc){
		cerr << Exc << endl;
		return 1;
	}
	catch (EDLUTFileException Exc){
		cerr << Exc << endl;
		return (int)Exc.GetErrorNum();
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		return (int)Exc.GetErrorNum();
	}

    return 0;
}
