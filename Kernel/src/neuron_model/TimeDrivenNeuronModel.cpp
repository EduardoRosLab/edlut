/***************************************************************************
 *                           TimeDrivenNeuronModel.cpp                     *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "neuron_model/TimeDrivenNeuronModel.h"
//#include "neuron_model/TimeDrivenModel.h"
#include "neuron_model/VectorNeuronState.h"
#include "simulation/NetworkDescription.h"
#include "integration_method/IntegrationMethodFactory.h"
//#include "integration_method/Euler.h"
//#include "openmp/openmp.h"

TimeDrivenNeuronModel::TimeDrivenNeuronModel(TimeScale new_time_scale):
		TimeDrivenModel(new_time_scale),
		conductance_exp_values(0),
		N_conductances(0),
		integration_method(0),
		CurrentSynapsis(0){
}

TimeDrivenNeuronModel::~TimeDrivenNeuronModel() {
	delete integration_method;

	if(N_conductances!=0){
		delete conductance_exp_values;
	}

	if (this->CurrentSynapsis != 0){
		delete this->CurrentSynapsis;
	}
}


enum NeuronModelSimulationMethod TimeDrivenNeuronModel::GetModelSimulationMethod(){
	return TIME_DRIVEN_MODEL_CPU;
}

enum NeuronModelType TimeDrivenNeuronModel::GetModelType(){
	return NEURAL_LAYER;
}



void TimeDrivenNeuronModel::CheckValidIntegration(double CurrentTime){
	float valid_integration = 0.0f;
	for (int i = 0; i < this->State->SizeStates; i++){
		valid_integration += this->State->GetStateVariableAt(i, 0);
	}
	if (valid_integration != valid_integration){
		for (int i = 0; i < this->State->SizeStates; i++){
			if (this->State->GetStateVariableAt(i, 0) != this->State->GetStateVariableAt(i, 0)){
				cout << CurrentTime << ": Integration error in " << this->GetNeuronModelName() << endl;
				for (int z = 0; z < this->GetVectorNeuronState()->NumberOfVariables; z++){
					cout << this->State->GetStateVariableAt(i, z) << " ";
				}cout << endl;

				State->ResetState(i);
			}
		}
	}
}

void TimeDrivenNeuronModel::CheckValidIntegration(double CurrentTime, float valid_integration){
	if (valid_integration != valid_integration){
		for (int i = 0; i < this->State->SizeStates; i++){
			if (this->State->GetStateVariableAt(i, 0) != this->State->GetStateVariableAt(i, 0)){
				cout << CurrentTime << ": Integration error in " << this->GetNeuronModelName() << endl;
				for (int z = 0; z < this->GetVectorNeuronState()->NumberOfVariables; z++){
					cout << this->State->GetStateVariableAt(i, z) << " ";
				}cout << endl;

				State->ResetState(i);
			}
		}
	}
}


void TimeDrivenNeuronModel::Initialize_conductance_exp_values(int N_conductances, int N_elapsed_times){
	conductance_exp_values = new float[N_conductances *  N_elapsed_times]();
	this->N_conductances=N_conductances;
}

void TimeDrivenNeuronModel::Set_conductance_exp_values(int elapsed_time_index, int conductance_index, float value){
	conductance_exp_values[elapsed_time_index*N_conductances+conductance_index]=value;

}

float TimeDrivenNeuronModel::Get_conductance_exponential_values(int elapsed_time_index, int conductance_index){
	return conductance_exp_values[elapsed_time_index*N_conductances+conductance_index];
}

float * TimeDrivenNeuronModel::Get_conductance_exponential_values(int elapsed_time_index){
	return conductance_exp_values + elapsed_time_index*N_conductances;
}

void TimeDrivenNeuronModel::InitializeInputCurrentSynapseStructure(){
	if (this->CurrentSynapsis != 0){
		this->CurrentSynapsis->InitializeInputCurrentPerSynapseStructure();
	}
}

std::map<std::string,boost::any> TimeDrivenNeuronModel::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenModel::GetParameters();
	ModelDescription imethod;
	imethod.param_map = this->integration_method->GetParameters();
	imethod.model_name = boost::any_cast<std::string>(imethod.param_map["name"]);
	newMap["int_meth"] = imethod;
	return newMap;
}

void TimeDrivenNeuronModel::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("int_meth");
	if (it!=param_map.end()){
		if (this->integration_method != 0){
			delete this->integration_method;
		}
		ModelDescription newIntegration = boost::any_cast<ModelDescription>(it->second);
		this->integration_method = this->CreateIntegrationMethod(newIntegration);
		param_map.erase(it);

		//SET TIME-DRIVEN STEP SIZE
		this->SetTimeDrivenStepSize(this->integration_method->GetIntegrationTimeStep());
	}

	// Search for the parameters in the dictionary
	TimeDrivenModel::SetParameters(param_map);
	return;
}

