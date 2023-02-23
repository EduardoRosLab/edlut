/***************************************************************************
 *                           Simulation_API.cpp                            *
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

#include "../../include/simulation/Simulation_API.h"
#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/NetworkDescription.h"
#include "../../include/simulation/RandomGenerator.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"
#include "../../include/learning_rules/LearningRuleFactory.h"
#include "../../include/learning_rules/LearningRule.h"
#include "../../include/integration_method/IntegrationMethodFactory.h"
#include "../../include/integration_method/IntegrationMethodFactoryInfo.h"
#include "../../include/neuron_model/NeuronModelFactory.h"
#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/LIFTimeDrivenModel.h"

#include <boost/any.hpp>
#include "../../include/spike/EDLUTException.h"
#include "../../include/spike/EDLUTFileException.h"
#include "../../include/communication/ConnectionException.h"

#include <algorithm>
#include <vector>



Simulation_API::Simulation_API(): simulation(0), initialized(false), simulation_properties(), input_spike_driver(),
		input_current_driver(), output_spike_driver(), file_input_spike_driver(0), file_output_spike_driver(0),
		file_input_current_driver(0), file_output_monitor_driver(0), file_output_weight_driver(0),
		num_neuron_created(0), neuron_layer_list(), learning_rule_list(), num_synapses_created(0), synaptic_layer_list() {

    // Set the default values for simulation properties here
    this->simulation_properties["num_threads"] = 1;
    this->simulation_properties["num_simulation_queues"] = 1;
    this->simulation_properties["resolution"] = 0.001;

}

Simulation_API::~Simulation_API() {
    if (this->simulation!=0){
        delete this->simulation;
        this->simulation = 0;
    }

	//All the drivers are automatically destroid in the simulation class.
}

std::vector<int> Simulation_API::AddNeuronLayer(int num_neurons, std::string model_name, std::map<std::string, boost::any> param_dict,
                                    bool log_activity, bool output_activity) noexcept(false){
	try{
		if (this->initialized){
			throw EDLUTException(TASK_ADD_NEURON_LAYER, ERROR_INITIALIZED_SIMULATION, REPAIR_EXECUTE_BEFORE_INITIALIZE_SIMULATION);
		}

		ModelDescription new_model;
		new_model.param_map = param_dict;
		new_model.model_name = model_name;

		NeuronLayerDescription new_layer;
		new_layer.num_neurons = num_neurons;
		new_layer.neuron_model = new_model;
		new_layer.log_activity = log_activity;
		new_layer.output_activity = output_activity;
		this->neuron_layer_list.push_back(new_layer);
		// Generate the vector of indexes for this layer
		std::vector<int> ind_vector = std::vector<int>(num_neurons,0);
		for (unsigned int i=0; i<ind_vector.size(); ++i) {
			ind_vector[i] = this->num_neuron_created++;
		}

		return ind_vector;
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

int Simulation_API::AddLearningRule(std::string model_name, std::map<std::string, boost::any> param_dict) noexcept(false){

	try{
		if (this->initialized){
			throw EDLUTException(TASK_ADD_LEARNING_RULE, ERROR_INITIALIZED_SIMULATION, REPAIR_EXECUTE_BEFORE_INITIALIZE_SIMULATION);
		}

		ModelDescription new_rule;
		new_rule.model_name = model_name;
		new_rule.param_map = param_dict;
		this->learning_rule_list.push_back(new_rule);
		return this->learning_rule_list.size()-1;
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}
std::vector<int> Simulation_API::AddSynapticLayer(const std::vector<int> & source_neuron, const std::vector<int> & target_neuron,
                                                  std::map<std::string, boost::any> param_dict) noexcept(false){
	try{
		if (this->initialized){
			throw EDLUTException(TASK_ADD_SYNAPTIC_LAYER, ERROR_INITIALIZED_SIMULATION, REPAIR_EXECUTE_BEFORE_INITIALIZE_SIMULATION);
		}

		SynapticLayerDescription new_layer;
		new_layer.source_neuron_list = source_neuron;
		new_layer.target_neuron_list = target_neuron;
		new_layer.param_map = param_dict;
		this->synaptic_layer_list.push_back(new_layer);
		// Generate the vector of indexes for this layer
		std::vector<int> ind_vector = std::vector<int>(source_neuron.size(),0);
		for (unsigned int i=0; i<ind_vector.size(); ++i) {
			ind_vector[i] = this->num_synapses_created++;
		}
		return ind_vector;
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::AddExternalSpikeActivity(const std::vector<double> & event_time, const std::vector<long int> & neuron_index) noexcept(false){
	try{
		if (!this->initialized) {
			throw EDLUTException(TASK_INPUT_SPIKE_DRIVER, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		}

		//we introduce the new activity in the driver.
		if (event_time.size()>0) {
			this->input_spike_driver.LoadInputs(this->simulation->GetQueue(), this->simulation->GetNetwork(),
													  event_time.size(), &event_time[0], &neuron_index[0]);
		}
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::AddExternalCurrentActivity(const std::vector<double> & event_time, const std::vector<long int> & neuron_index, const std::vector<float> & current_value) noexcept(false){
	try{
		if (!this->initialized) {
			throw EDLUTException(TASK_INPUT_CURRENT_DRIVER, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		}

		//we introduce the new activity in the driver.
		if (event_time.size()>0) {
			this->input_current_driver.LoadInputs(this->simulation->GetQueue(), this->simulation->GetNetwork(),
													  event_time.size(), &event_time[0], &neuron_index[0], &current_value[0]);
		}
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::GetSpikeActivity(std::vector<double> & event_time, std::vector<long int> & neuron_index) noexcept(false){
	try{
		if (!this->initialized) {
			throw EDLUTException(TASK_OUTPUT_SPIKE_DRIVER, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		}

		double * OutputSpikeTimes;
		long int * OutputSpikeCells;

		unsigned int OutputSpikes = this->output_spike_driver.GetBufferedSpikes(OutputSpikeTimes,OutputSpikeCells);

		if (OutputSpikes>0) {
			event_time.resize(OutputSpikes);
			neuron_index.resize(OutputSpikes);
			double * SpTimesPtr = OutputSpikeTimes;
			long int * SpCellsPtr = OutputSpikeCells;
			std::vector<double>::iterator itTimes = event_time.begin();
			std::vector<long int>::iterator itNeurons = neuron_index.begin();
			for (unsigned int counter=0; counter<OutputSpikes; ++counter,++SpTimesPtr, ++SpCellsPtr, ++itTimes, ++itNeurons) {
				*itTimes = *SpTimesPtr;
				*itNeurons = *SpCellsPtr;
			}
		}

		return;
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}


void Simulation_API::AddFileInputSpikeActivityDriver(string FileName) noexcept(false){
	try{
		if (this->initialized){
			throw EDLUTException(TASK_INITIALIZE_SIMULATION, ERROR_INITIALIZED_SIMULATION, REPAIR_EXECUTE_BEFORE_INITIALIZE_SIMULATION);
		}

		//Just one file input spike driver is possible.
		if (file_input_spike_driver != 0){
			cout << "WARNING: Just one file input spike driver can be defined. Previos one will be ignored." << endl;
			delete this->file_input_spike_driver;
		}
		file_input_spike_driver = new FileInputSpikeDriver(FileName.c_str());
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::AddFileInputCurrentActivityDriver(string FileName) noexcept(false){
	try{
		if (this->initialized){
			throw EDLUTException(TASK_INITIALIZE_SIMULATION, ERROR_INITIALIZED_SIMULATION, REPAIR_EXECUTE_BEFORE_INITIALIZE_SIMULATION);
		}

		//Just one file input spike driver is possible.
		if (file_input_current_driver != 0){
			cout << "WARNING: Just one file input current driver can be defined. Previos one will be ignored." << endl;
			delete this->file_input_current_driver;
		}
		file_input_current_driver = new FileInputCurrentDriver(FileName.c_str());
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::AddFileOutputSpikeActivityDriver(string FileName) noexcept(false){
	try{
		if (this->initialized){
			throw EDLUTException(TASK_INITIALIZE_SIMULATION, ERROR_INITIALIZED_SIMULATION, REPAIR_EXECUTE_BEFORE_INITIALIZE_SIMULATION);
		}

		//Just one file input spike driver is possible.
		if (file_output_spike_driver != 0){
			cout << "WARNING: Just one file output spike driver can be defined. Previos one will be ignored." << endl;
			delete this->file_output_spike_driver;
		}
		file_output_spike_driver = new FileOutputSpikeDriver(FileName.c_str(), false);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::AddFileOutputMonitorDriver(string FileName, bool monitor_state) noexcept(false){
	try{
		if (this->initialized){
			throw EDLUTException(TASK_INITIALIZE_SIMULATION, ERROR_INITIALIZED_SIMULATION, REPAIR_EXECUTE_BEFORE_INITIALIZE_SIMULATION);
		}

		//Just one file input spike driver is possible.
		if (file_output_monitor_driver != 0){
			cout << "WARNING: Just one file output monitor driver can be defined. Previos one will be ignored." << endl;
			delete this->file_output_monitor_driver;
		}
		file_output_monitor_driver = new FileOutputSpikeDriver(FileName.c_str(), monitor_state);

	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::AddFileOutputWeightDriver(string FileName, float save_weight_period) noexcept(false){
	try{
		if (this->initialized){
			throw EDLUTException(TASK_INITIALIZE_SIMULATION, ERROR_INITIALIZED_SIMULATION, REPAIR_EXECUTE_BEFORE_INITIALIZE_SIMULATION);
		}

		//Just one file input spike driver is possible.
		if (file_output_weight_driver != 0){
			cout << "WARNING: Just one file output weight driver can be defined. Previos one will be ignored." << endl;
			delete this->file_output_weight_driver;
		}
		file_output_weight_driver = new FileOutputWeightDriver(FileName.c_str());

		if (save_weight_period > 0){
			this->save_weight_period = save_weight_period;
		}
		else{
			this->save_weight_period = 10.0f;
			cout << "WARNING: save weight period must be greater than zero. This value has been fixed to " << this->save_weight_period<<"." << endl;
		}

	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}



void Simulation_API::GetCompressedWeights(std::vector<int> & N_equal_weights, std::vector<float> & equal_weights) noexcept(false){
	try{
		if (!this->initialized) {
			throw EDLUTException(TASK_GET_WEIGHTS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		}
		this->simulation->GetNetwork()->GetCompressedWeights(N_equal_weights, equal_weights);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

std::vector<float> Simulation_API::GetWeights() noexcept(false){
	try{
		if (!this->initialized) {
			throw EDLUTException(TASK_GET_WEIGHTS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		}
		return this->simulation->GetNetwork()->GetWeights();
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

std::vector<float> Simulation_API::GetSelectedWeights(std::vector<int> synaptic_indexes) noexcept(false){
	try{
		if (!this->initialized) {
			throw EDLUTException(TASK_GET_WEIGHTS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		}
		return this->simulation->GetNetwork()->GetSelectedWeights(synaptic_indexes);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::Initialize() noexcept(false) {
	try{
		if (this->initialized){
			throw EDLUTException(TASK_INITIALIZE_SIMULATION, ERROR_INITIALIZED_SIMULATION, REPAIR_EXECUTE_BEFORE_INITIALIZE_SIMULATION);
		}

		this->simulation = new Simulation(this->neuron_layer_list, this->learning_rule_list,
			this->synaptic_layer_list, 0.0, 0.0, boost::any_cast<int>(this->simulation_properties["num_simulation_queues"]));

		// The number of threads is automatically set to the number of queues
		// The spike time resolution is not set yet

		//Include in the simulation all the configured drivers
		this->simulation->AddInputSpikeDriver(&this->input_spike_driver);
		this->simulation->AddOutputSpikeDriver(&this->output_spike_driver);
		this->simulation->AddInputCurrentDriver(&this->input_current_driver);
		if (this->file_input_spike_driver != 0){
			this->simulation->AddInputSpikeDriver(this->file_input_spike_driver);
		}
		if (this->file_output_spike_driver != 0){
			this->simulation->AddOutputSpikeDriver(this->file_output_spike_driver);
		}
		if (this->file_input_current_driver != 0){
			this->simulation->AddInputCurrentDriver(this->file_input_current_driver);
		}
		if (this->file_output_monitor_driver != 0){
			this->simulation->AddMonitorActivityDriver(this->file_output_monitor_driver);
		}
		if (this->file_output_weight_driver != 0){
			this->simulation->AddOutputWeightDriver(this->file_output_weight_driver);
			this->simulation->SetSaveStep(this->save_weight_period);
		}

		// Get the external initial inputs (none in this simulation)
		this->simulation->InitSimulation();

		this->initialized = true;
	}
	catch (ConnectionException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
	catch (EDLUTFileException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
    return;
}

void Simulation_API::SetSimulationParameters(std::map<std::string, boost::any> param_dict) noexcept(false){
	try{
		if (this->initialized){
			throw EDLUTException(TASK_SET_SIMULATION_PARAMETERS, ERROR_INITIALIZED_SIMULATION, REPAIR_EXECUTE_BEFORE_INITIALIZE_SIMULATION);
		}

		for (std::map<std::string, boost::any>::const_iterator it = param_dict.begin(); it != param_dict.end(); ++it){
			std::map<std::string, boost::any>::iterator it_default = this->simulation_properties.find(it->first);
			if (it_default != this->simulation_properties.end()){
				it_default->second = it->second;
			}
			else {
				throw EDLUTException(TASK_SET_SIMULATION_PARAMETERS, ERROR_INVALID_SIMULATION_PARAMETER, REPAIR_CHECK_SIMULATION_PARAMETER);
			}
		}
		return;
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::RunSimulation(double end_simulation_time) noexcept(false){
	try{
		if (!this->initialized){
			throw EDLUTException(TASK_RUN_SIMULATION, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		}
		this->simulation->RunSimulationSlot(end_simulation_time);

		return;
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}

}

std::map<std::string, boost::any> Simulation_API::GetLearningRuleDefParams(std::string model_name) noexcept(false){
	try{
		return LearningRuleFactory::GetDefaultParameters(model_name);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

std::map<std::string, boost::any> Simulation_API::GetNeuronModelDefParams(std::string model_name) noexcept(false){
	try{
		return NeuronModelFactory::GetDefaultParameters(model_name);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

std::map<std::string, boost::any> Simulation_API::GetIntegrationMethodDefParams(std::string model_name) noexcept(false){
	try{
		return IntegrationMethodFactory<LIFTimeDrivenModel>::GetDefaultParameters(model_name);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

std::map<std::string, boost::any> Simulation_API::GetLearningRuleParams(int lrule_index) noexcept(false){
	try{
		if (!this->initialized){
			throw EDLUTException(TASK_GET_LEARNING_RULE_PARAMETERS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		}
		return this->simulation->GetNetwork()->GetLearningRuleAt(lrule_index)->GetParameters();
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}


void Simulation_API::SetLearningRuleParams(int lrule_index, std::map<std::string, boost::any> newParam) noexcept(false){
	try{
		if (!this->initialized){
			throw EDLUTException(TASK_SET_LEARNING_RULE_PARAMETERS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		}
		this->simulation->GetNetwork()->GetLearningRuleAt(lrule_index)->SetParameters(newParam);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

//float* Simulation_API::GetLayerState(std::vector<int> neuron_indices, int &n_states) noexcept(false){
//	try{
//		if (!this->initialized){
//			throw EDLUTException(TASK_GET_NEURON_PARAMS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
//		}
//		/*
//		std::vector<float> StateVector(neuron_index.size());
//		unsigned int n_var = this->simulation->GetNetwork()->GetNeuronAt(neuron_index[0])->GetVectorNeuronState()->GetNumberOfVariables();
//
//		for (unsigned int i = 0; i < neuron_index.size(); ++i) {
//			auto neuron = this->simulation->GetNetwork()->GetNeuronAt(neuron_index[i]);
//			auto state = neuron->GetVectorNeuronState();
//			for (unsigned int j = 0; j < n_var; ++j) {
//				StateVector[i][j] = state->GetStateVariableAt(i,j);
//			}
//		}
//		return StateVector;
//		*/

//		auto state = this->simulation->GetNetwork()->GetNeuronAt(neuron_indices[0])->GetVectorNeuronState();
//		n_states = state->NumberOfVariables;
//		return state->VectorNeuronStates;
//	}
//	catch (EDLUTException Exc){
//		cerr << Exc << endl;
//		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
//	}
//}

std::vector<std::map<std::string, boost::any> > Simulation_API::GetNeuronParams(std::vector<int> neuron_index) noexcept(false){
	try{
		if (!this->initialized){
			throw EDLUTException(TASK_GET_NEURON_PARAMS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		}
		std::vector<std::map<std::string, boost::any> > ParamVector(neuron_index.size());
		for (unsigned int i = 0; i < neuron_index.size(); ++i){
			auto neuron = this->simulation->GetNetwork()->GetNeuronAt(neuron_index[i]);
			ParamVector[i] = neuron->GetNeuronModel()->GetParameters();
		}
		return ParamVector;
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

std::vector<std::map<std::string, boost::any> > Simulation_API::GetSpecificNeuronParams(std::vector<int> neuron_index) noexcept(false){
	try{
		if (!this->initialized){
			throw EDLUTException(TASK_GET_NEURON_PARAMS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		}
		std::vector<std::map<std::string, boost::any> > ParamVector(neuron_index.size());
		for (unsigned int i = 0; i < neuron_index.size(); ++i) {
			auto neuron = this->simulation->GetNetwork()->GetNeuronAt(neuron_index[i]);
			ParamVector[i] = neuron->GetNeuronModel()->GetSpecificNeuronParameters(neuron->GetIndex_VectorNeuronState());
		}
		return ParamVector;
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::SetSpecificNeuronParams(int neuron_index, std::map<std::string, boost::any> newParam) noexcept(false){
	try{
		if (!this->initialized){
			throw EDLUTException(TASK_SET_NEURON_LAYER_PARAMS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		}
		this->simulation->GetNetwork()->GetNeuronAt(neuron_index)->GetNeuronModel()->SetSpecificNeuronParameters(this->simulation->GetNetwork()->GetNeuronAt(neuron_index)->GetIndex_VectorNeuronState(), newParam);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::SetRandomGeneratorSeed(int seed){
	RandomGenerator::srand(seed);
}

std::vector<std::string> Simulation_API::GetAvailableNeuronModels(){
	return NeuronModelFactory::GetAvailableNeuronModels();
}

void Simulation_API::PrintAvailableNeuronModels(){
	NeuronModelFactory::PrintAvailableNeuronModels();
}


std::map<std::string, std::string> Simulation_API::GetNeuronModelInfo(string neuronModelName) noexcept(false){
	try{
		return NeuronModelFactory::GetNeuronModelInfo(neuronModelName);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::PrintNeuronModelInfo(string neuronModelName) noexcept(false){
	try{
		NeuronModelFactory::PrintNeuronModelInfo(neuronModelName);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}


std::vector<std::string> Simulation_API::GetAvailableIntegrationMethods(){
	return IntegrationMethodFactoryInfo::GetAvailableIntegrationMethods();
}

void Simulation_API::PrintAvailableIntegrationMethods(){
	IntegrationMethodFactoryInfo::PrintAvailableIntegrationMethods();
}

std::vector<std::string> Simulation_API::GetAvailableLearningRules(){
	return LearningRuleFactory::GetAvailableLearningRules();
}

void Simulation_API::PrintAvailableLearningRules(){
	LearningRuleFactory::PrintAvailableLearningRules();
}



std::map<std::string, std::string> Simulation_API::GetVectorizableParameters(string neuronModelName) noexcept(false){
	try{
		return NeuronModelFactory::GetVectorizableParameters(neuronModelName);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}

void Simulation_API::PrintVectorizableParameters(string neuronModelName) noexcept(false){
	try{
		NeuronModelFactory::PrintVectorizableParameters(neuronModelName);
	}
	catch (EDLUTException Exc){
		cerr << Exc << endl;
		throw EDLUTException(TASK_EDLUT_INTERFACE, ERROR_EDLUT_INTERFACE, REPAIR_EDLUT_INTERFACE);
	}
}
