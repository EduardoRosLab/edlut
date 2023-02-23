/***************************************************************************
 *                           PoissonGeneratorDeviceVector.cpp              *
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

#include "neuron_model/PoissonGeneratorDeviceVector.h"
#include "neuron_model/VectorNeuronState.h"

#include "spike/Neuron.h"
#include "spike/Interconnection.h"



//this neuron model is implemented in a milisecond scale.
PoissonGeneratorDeviceVector::PoissonGeneratorDeviceVector() : TimeDrivenInputDevice(MilisecondScale), sampling_period(0.1f), vectorParameters(0), randomGenerator(0){
	std::map<std::string, boost::any> param_map = PoissonGeneratorDeviceVector::GetDefaultParameters();
	param_map["name"] = PoissonGeneratorDeviceVector::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState *) new VectorNeuronState(1, true);
}


PoissonGeneratorDeviceVector::~PoissonGeneratorDeviceVector(void){
	if(vectorParameters!=0){
		delete vectorParameters;
	}

	if (this->randomGenerator != 0){
		for (int i = 0; i < this->GetVectorNeuronState()->GetSizeState(); i++){
			delete this->randomGenerator[i];
		}

		delete[] this->randomGenerator;
	}
}


VectorNeuronState * PoissonGeneratorDeviceVector::InitializeState(){
	return this->GetVectorNeuronState();
}



bool PoissonGeneratorDeviceVector::UpdateState(int index, double CurrentTime){
	//Reset the number of internal spikes in this update period
	this->State->NInternalSpikeIndexs = 0;

	for (int i = 0; i < this->GetVectorNeuronState()->GetSizeState(); i++){
		if (this->vectorParameters[i].frequency > 0.0f && (this->vectorParameters[i].frequency * this->GetTimeDrivenStepSize()) > randomGenerator[i]->frand()){
			this->State->InternalSpikeIndexs[this->State->NInternalSpikeIndexs] = i;
			this->State->NInternalSpikeIndexs++;
			this->State->SetStateVariableAt(i, 0, CurrentTime);
			this->State->NewFiredSpike(i);
		}
	}

	return false;
}


enum NeuronModelOutputActivityType PoissonGeneratorDeviceVector::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}


ostream & PoissonGeneratorDeviceVector::PrintInfo(ostream & out){
	out << "- Poisson generator device vector: " << PoissonGeneratorDeviceVector::GetName() << endl;
	out << "\tPoisson generator device frequency (frequency): " << this->frequency << "Hz" << endl;
	return out;
}


void PoissonGeneratorDeviceVector::InitializeStates(int N_neurons, int OpenMPQueueIndex){

	float initialization[] = {0.0f};
	State->InitializeStates(N_neurons, initialization);

	vectorParameters = new VectorParameters[N_neurons];
	for(int i=0; i<N_neurons; i++){
		vectorParameters[i].frequency = this->frequency;
	}


	this->randomGenerator = (RandomGenerator **) new RandomGenerator * [N_neurons];
	for (int i = 0; i<N_neurons; i++){
		this->randomGenerator[i] = new RandomGenerator();
	}
}


std::map<std::string,boost::any> PoissonGeneratorDeviceVector::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = TimeDrivenInputDevice::GetParameters();
	newMap["frequency"] = boost::any(this->frequency); // Poisson generator device frecuency (Hz): VECTOR
	return newMap;
}

std::map<std::string, boost::any> PoissonGeneratorDeviceVector::GetSpecificNeuronParameters(int index) const noexcept(false){
	//This method must be executed after the initilization of the model.
	if (this->State->GetSizeState() == 0){
		//The neuron model has not been initialized
		throw EDLUTException(TASK_GET_NEURON_SPECIFIC_PARAMETERS, ERROR_NON_INITIALIZED_SIMULATION, REPAIR_EXECUTE_AFTER_INITIALIZE_SIMULATION);
	}

	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenInputDevice::GetParameters();
	newMap["frequency"] = boost::any(this->vectorParameters[index].frequency); // Poisson generator frecuency (Hz): VECTOR
	return newMap;
}

void PoissonGeneratorDeviceVector::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("frequency");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->frequency = new_param;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	TimeDrivenInputDevice::SetParameters(param_map);

	//SET TIME-DRIVEN STEP SIZE (IN SECONDS)
	this->SetTimeDrivenStepSize(sampling_period / this->time_scale);

	return;
}

void PoissonGeneratorDeviceVector::SetSpecificNeuronParameters(int index, std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("frequency");
	if (it!=param_map.end()){
		float new_param = boost::any_cast<float>(it->second);
		this->vectorParameters[index].frequency = new_param;
		param_map.erase(it);
	}

	NeuronModel::SetSpecificNeuronParameters(index, param_map);

	return;
}

std::map<std::string, std::string> PoissonGeneratorDeviceVector::GetVectorizableParameters(){
	std::map<std::string, std::string> vectorizableParameters;
	vectorizableParameters["frequency"] = std::string("Poisson generator device frequency (Hz): VECTOR");
	return vectorizableParameters;
}

std::map<std::string,boost::any> PoissonGeneratorDeviceVector::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenInputDevice::GetDefaultParameters();
	newMap["frequency"] = boost::any(10.0f); // Poisson generator device frequency (Hz): VECTOR
	return newMap;
}

NeuronModel* PoissonGeneratorDeviceVector::CreateNeuronModel(ModelDescription nmDescription){
	PoissonGeneratorDeviceVector * nmodel = new PoissonGeneratorDeviceVector();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription PoissonGeneratorDeviceVector::ParseNeuronModel(std::string FileName) noexcept(false){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = PoissonGeneratorDeviceVector::GetName();
	long Currentline = 0L;
	fh=fopen(FileName.c_str(),"rt");
	if(!fh) {
		throw EDLUTFileException(TASK_POISSON_GENERATOR_DEVICE_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	skip_comments(fh, Currentline);

	float param;
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_POISSON_GENERATOR_DEVICE_LOAD, ERROR_POISSON_GENERATOR_DEVICE_FREQUENCY, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["frequency"] = boost::any(param);

	nmodel.param_map["name"] = boost::any(PoissonGeneratorDeviceVector::GetName());

	fclose(fh);

	return nmodel;
}

std::string PoissonGeneratorDeviceVector::GetName(){
	return "PoissonGeneratorDeviceVector";
}

std::map<std::string, std::string> PoissonGeneratorDeviceVector::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Poisson generator device able to generate and propagate spike trains to other neuron models");
	newMap["frequency"] = std::string("Poisson generator device frequency (Hz): VECTOR");

	return newMap;
}
