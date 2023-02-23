/***************************************************************************
 *                           ExpBufferedWeightChange.cpp                   *
 *                           -------------------                           *
 * copyright            : (C) 2019 by Francisco Naveros                    *
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


#include "../../include/learning_rules/ExpBufferedWeightChange.h"
#include "../../include/learning_rules/BufferedActivityTimes.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"
#include <boost/any.hpp>

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

ExpBufferedWeightChange::ExpBufferedWeightChange():AdditiveKernelChange(), initTime(0),
bufferedActivityTimesNoTrigger(0), kernelLookupTable(0){
	// Set the default values for the learning rule parameters
	this->SetParameters(ExpBufferedWeightChange::GetDefaultParameters());
}

ExpBufferedWeightChange::~ExpBufferedWeightChange(){
	if(bufferedActivityTimesNoTrigger!=0){
		delete bufferedActivityTimesNoTrigger;
	}
	if(kernelLookupTable!=0){
		delete kernelLookupTable;
	}
}


void ExpBufferedWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	if (this->kernelpeak <= this->initTime){
		this->kernelpeak = this->initTime + 1e-6;
	}

	double step_size = 0.0001;
	double tolerance = 1e-2;

	this->maxTimeMeasured = this->kernelpeak;
	while (1){
		this->maxTimeMeasured += step_size;
		if ((1.0 / (this->kernelpeak - this->initTime))*this->maxTimeMeasured*exp(-(this->maxTimeMeasured / (this->kernelpeak - this->initTime)) + 1)< tolerance){
			break;
		}
	}
	this->maxTimeMeasured += this->initTime;


	this->N_elements = this->maxTimeMeasured / step_size + 1;
	kernelLookupTable = new float[this->N_elements];


	this->inv_maxTimeMeasured = 1.0f / this->maxTimeMeasured;
	//Precompute the kernel in the look-up table.
	for (int i = 0; i<N_elements; i++){
		double time = maxTimeMeasured*i / N_elements;
		if(time < this->initTime){
			kernelLookupTable[i] = 0.0;
		}else{
			double time_ = time - this->initTime;
		  kernelLookupTable[i] = (1.0/(this->kernelpeak-this->initTime))*time_*exp(-(time_/(this->kernelpeak-this->initTime))+1);
		}
	}

	//Inicitialize de buffer of activity
	bufferedActivityTimesNoTrigger = new BufferedActivityTimes(NumberOfNeurons);
}

int ExpBufferedWeightChange::GetNumberOfVar() const{
	return 2;
}

float ExpBufferedWeightChange::GetInitTime() const{
	return this->initTime;
}


ModelDescription ExpBufferedWeightChange::ParseLearningRule(FILE * fh) noexcept(false) {
	ModelDescription lrule = AdditiveKernelChange::ParseLearningRule(fh);

	float inittime;
	if(fscanf(fh,"%f",&inittime)!=1){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_EXP_BUFFERED_WEIGHT_CHANGE_LOAD);
	}

	//load the kernel peak value from the ModelDescription structure to check if the inittime value is valid.
	std::map<std::string, boost::any>::iterator it = lrule.param_map.find("kernel_peak");
	float new_kernel_peak = boost::any_cast<float>(it->second);
	if (inittime < 0.0 || inittime >= new_kernel_peak){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_EXP_BUFFERED_WEIGHT_CHANGE_INIT_TIME, REPAIR_LEARNING_RULE_VALUES);
	}

	lrule.model_name = ExpBufferedWeightChange::GetName();
	lrule.param_map["init_time"] = boost::any(inittime);
	return lrule;
}

ostream & ExpBufferedWeightChange::PrintInfo(ostream & out){
	out << "- ExpBufferedAdditiveKernel Learning Rule: " << endl;
	out << "\t kernel_peak:" << this->kernelpeak << endl;
	out << "\t fixed_change: " << this->fixwchange << endl;
	out << "\t kernel_change: " << this->kernelwchange << endl;
	out << "\t init_time: " << this->initTime << endl;
	return out;
}

void ExpBufferedWeightChange::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string, boost::any>::iterator it = param_map.find("kernel_peak"); // auxiliar parameter required for verification
	if (it != param_map.end()){
		float newKernelPeak = boost::any_cast<float>(it->second);

		it = param_map.find("init_time");
		if (it != param_map.end()){
			float newInitTime = boost::any_cast<float>(it->second);

			if (newInitTime < 0.0f || newInitTime >= newKernelPeak) {
				throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_EXP_BUFFERED_WEIGHT_CHANGE_INIT_TIME, REPAIR_LEARNING_RULE_VALUES);
			}
			this->initTime = newInitTime;
			param_map.erase(it);
		}
	}

	AdditiveKernelChange::SetParameters(param_map);

}

void ExpBufferedWeightChange::ApplyPreSynapticSpike(Interconnection * Connection, double SpikeTime){

	if (Connection->GetTriggerConnection() == false){
		Connection->IncrementWeight(this->fixwchange);
		int neuron_index = Connection->GetTarget()->GetIndex();
		int synapse_index = Connection->LearningRuleIndex_withTrigger_insideTargetNeuron;
		this->bufferedActivityTimesNoTrigger->InsertElement(neuron_index, SpikeTime, SpikeTime - this->maxTimeMeasured, synapse_index);

	}
	else{
		Neuron * TargetNeuron = Connection->GetTarget();
		int neuron_index = TargetNeuron->GetIndex();
		unsigned int LearningRuleId = this->GetLearningRuleIndex();

		int N_elements = bufferedActivityTimesNoTrigger->ProcessElements(neuron_index, SpikeTime - this->maxTimeMeasured);
		SpikeData * spike_data = bufferedActivityTimesNoTrigger->GetOutputSpikeData();

		float value = 0;
		for (int i = 0; i < N_elements; i++){
			Interconnection * interi = TargetNeuron->GetInputConnectionWithTriggerSynapticLearningAt(LearningRuleId,spike_data[i].synapse_index);

			double ElapsedTime = SpikeTime - spike_data[i].time;
			int tableIndex = ElapsedTime*this->N_elements*this->inv_maxTimeMeasured;
			value = this->kernelLookupTable[tableIndex];

			// Update synaptic weight
			interi->IncrementWeight(this->kernelwchange*value);
		}
	}
}


LearningRule* ExpBufferedWeightChange::CreateLearningRule(ModelDescription lrDescription){
	ExpBufferedWeightChange * lrule = new ExpBufferedWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> ExpBufferedWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = AdditiveKernelChange::GetParameters();
	newMap["init_time"] = boost::any(this->initTime);
	return newMap;
}


std::map<std::string,boost::any> ExpBufferedWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = AdditiveKernelChange::GetDefaultParameters();
	newMap["init_time"] = boost::any(0.050f);
	return newMap;
}
