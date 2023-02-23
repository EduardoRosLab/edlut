/***************************************************************************
 *                           SinCurrentDeviceVector.cpp                    *
 *                           -------------------                           *
 * copyright            : (C) 2020 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "neuron_model/SinCurrentDeviceVector.h"
#include "neuron_model/VectorNeuronState.h"

#include "spike/Neuron.h"
#include "spike/Interconnection.h"


//this neuron model is implemented in a milisecond scale.
SinCurrentDeviceVector::SinCurrentDeviceVector() : TimeDrivenInputDevice(MilisecondScale), sampling_period (1.0f), vectorParameters(0){
	std::map<std::string, boost::any> param_map = SinCurrentDeviceVector::GetDefaultParameters();
	param_map["name"] = SinCurrentDeviceVector::GetName();
	this->SetParameters(param_map);

	//The output current will be stored in the first position of the VectorNeuronState for each device
	this->State = (VectorNeuronState *) new VectorNeuronState(1, true);
}


SinCurrentDeviceVector::~SinCurrentDeviceVector(void){
	if(vectorParameters!=0){
		delete vectorParameters;
	}
}


VectorNeuronState * SinCurrentDeviceVector::InitializeState(){
	return NULL;
}


bool SinCurrentDeviceVector::UpdateState(int index, double CurrentTime){
	for (int i = 0; i < this->GetVectorNeuronState()->GetSizeState(); i++){
		//Compute the new current value for each "device".
		float current = this->vectorParameters[i].offset + this->vectorParameters[i].amplitude * sin(float(CurrentTime)*this->vectorParameters[i].frequency * 2 * 3.141592 + this->vectorParameters[i].phase);

		this->GetVectorNeuronState()->SetStateVariableAt(i, 0, current);
	}

	return false;
}


enum NeuronModelOutputActivityType SinCurrentDeviceVector::GetModelOutputActivityType(){
	return OUTPUT_CURRENT;
}


ostream & SinCurrentDeviceVector::PrintInfo(ostream & out){
	out << "- Sinusoidal current device vector: " << SinCurrentDeviceVector::GetName() << endl;
	out << "\tSinusoidal frequency (frequency): " << this->frequency << "Hz" << endl;
	out << "\tSinusoidal amplitude (amplitude): " << this->amplitude << "pA" << endl;
	out << "\tSinusoidal offset (offset): " << this->offset << "pA" << endl;
	out << "\tSinusoidal phase (phase): " << this->phase << "rad" << endl;
	return out;
}


void SinCurrentDeviceVector::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	//Initialize neural state variables (output currents).
	float current = this->offset + this->amplitude * sin(this->phase);

	
	float initialization[] = {current};
	State->InitializeStates(N_neurons, initialization);
	
	vectorParameters = new VectorParameters[N_neurons];
	for(int i=0; i<N_neurons; i++){
		vectorParameters[i].frequency = this->frequency;
		vectorParameters[i].amplitude = this->amplitude;
		vectorParameters[i].offset = this->offset;
		vectorParameters[i].phase = this->phase;
	}
}


std::map<std::string,boost::any> SinCurrentDeviceVector::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenInputDevice::GetParameters();
	newMap["frequency"] = boost::any(this->frequency); // Sinusoidal frecuency (Hz): VECTOR
	newMap["amplitude"] = boost::any(this->amplitude); // Sinusoidal amplitude (pA): VECTOR
	newMap["offset"] = boost::any(this->offset); // Sinusoidal offset (pA): VECTOR
	newMap["phase"] = boost::any(this->phase); // Sinusoidal phase (rad): VECTOR
	return newMap;
}

std::map<std::string, boost::any> SinCurrentDeviceVector::GetSpecificNeuronParameters(int index) const noexcept(false){
	//This method must be executed after the initilization of the model.
	if (this->State->GetSizeState() == 0){
		//The neuron model has not been initialized
		throw EDLUTException(TASK_GET_NEURON_SPECIFIC_PARAMETERS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
	}

	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenInputDevice::GetParameters();
	newMap["frequency"] = boost::any(this->vectorParameters[index].frequency); // Sinusoidal frecuency (Hz): VECTOR
	newMap["amplitude"] = boost::any(this->vectorParameters[index].amplitude); // Sinusoidal amplitude (pA): VECTOR
	newMap["offset"] = boost::any(this->vectorParameters[index].offset); // Sinusoidal offset (pA): VECTOR
	newMap["phase"] = boost::any(this->vectorParameters[index].phase); // Sinusoidal phase (rad): VECTOR	return newMap;
	return newMap;
}

void SinCurrentDeviceVector::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("frequency");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->frequency = new_param;
		param_map.erase(it);
	}

	it=param_map.find("amplitude");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->amplitude = new_param;
		param_map.erase(it);
	}

	it=param_map.find("offset");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->offset = new_param;
		param_map.erase(it);
	}

	it=param_map.find("phase");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->phase = new_param;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	TimeDrivenInputDevice::SetParameters(param_map);

	//SET TIME-DRIVEN STEP SIZE (IN SECONDS)
	this->SetTimeDrivenStepSize(sampling_period / this->time_scale);

	return;
}

void SinCurrentDeviceVector::SetSpecificNeuronParameters(int index, std::map<std::string, boost::any> param_map) noexcept(false){
	//This method must be executed after the initilization of the model.
	if (this->State->GetSizeState() == 0){
		//The neuron model has not been initialized
		throw EDLUTException(TASK_GET_NEURON_SPECIFIC_PARAMETERS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
		//REVISAR
	}

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("frequency");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].frequency = new_param;
		param_map.erase(it);
	}

	it=param_map.find("amplitude");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].amplitude = new_param;
		param_map.erase(it);
	}

	it=param_map.find("offset");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].offset = new_param;
		param_map.erase(it);
	}

	it=param_map.find("phase");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].phase = new_param;
		param_map.erase(it);
	}

	NeuronModel::SetSpecificNeuronParameters(index, param_map);

	return;
}

std::map<std::string, std::string> SinCurrentDeviceVector::GetVectorizableParameters(){
	std::map<std::string, std::string> vectorizableParameters;
	vectorizableParameters["frequency"] = std::string("Sinusoidal frequency (Hz): VECTOR");
	vectorizableParameters["amplitude"] = std::string("Sinusoidal amplitude (pA): VECTOR");
	vectorizableParameters["offset"] = std::string("Sinusoidal offset (pA): VECTOR");
	vectorizableParameters["phase"] = std::string("Sinusoidal phase (rad): VECTOR");
	return vectorizableParameters;
}

std::map<std::string,boost::any> SinCurrentDeviceVector::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenInputDevice::GetDefaultParameters();
	newMap["frequency"] = boost::any(1.0f); // Sinusoidal frequency (Hz)
	newMap["amplitude"] = boost::any(200.0f); // Sinusoidal amplitude (pA)
	newMap["offset"] = boost::any(200.0f); // Sinusoidal offset (pA)
	newMap["phase"] = boost::any(0.0f); // Sinusoidal phase (rad)
	return newMap;
}

NeuronModel* SinCurrentDeviceVector::CreateNeuronModel(ModelDescription nmDescription){
	SinCurrentDeviceVector * nmodel = new SinCurrentDeviceVector();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription SinCurrentDeviceVector::ParseNeuronModel(std::string FileName) noexcept(false){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = SinCurrentDeviceVector::GetName();
	long Currentline = 0L;
	fh=fopen(FileName.c_str(),"rt");
	if(!fh) {
		throw EDLUTFileException(TASK_SIN_CURRENT_DEVICE_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	skip_comments(fh, Currentline);

	float param;
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_SIN_CURRENT_DEVICE_LOAD, ERROR_SIN_CURRENT_DEVICE_FREQUENCY, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["frequency"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_SIN_CURRENT_DEVICE_LOAD, ERROR_SIN_CURRENT_DEVICE_AMPLITUDE, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["amplitude"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_SIN_CURRENT_DEVICE_LOAD, ERROR_SIN_CURRENT_DEVICE_OFFSET, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["offset"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_SIN_CURRENT_DEVICE_LOAD, ERROR_SIN_CURRENT_DEVICE_PHASE, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["phase"] = boost::any(param);

	nmodel.param_map["name"] = boost::any(SinCurrentDeviceVector::GetName());

	fclose(fh);

	return nmodel;
}

std::string SinCurrentDeviceVector::GetName(){
	return "SinCurrentDeviceVector";
}

std::map<std::string, std::string> SinCurrentDeviceVector::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Sinusoidal current device able to generate and propagate a sinusoidal current (in pA) to other neuron models");
	newMap["frequency"] = std::string("Sinusoidal frequency (Hz): VECTOR");
	newMap["amplitude"] = std::string("Sinusoidal amplitude (pA): VECTOR");
	newMap["offset"] = std::string("Sinusoidal offset (pA): VECTOR");
	newMap["phase"] = std::string("Sinusoidal phase (rad): VECTOR");
	
	return newMap;
}
