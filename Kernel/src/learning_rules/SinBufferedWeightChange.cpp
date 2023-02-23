/***************************************************************************
 *                           SinBufferedWeightChange.cpp                   *
 *                           -------------------                           *
 * copyright            : (C) 2016 by Francisco Naveros                    *
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


#include "../../include/learning_rules/SinBufferedWeightChange.h"
#include "../../include/learning_rules/BufferedActivityTimes.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"
#include <boost/any.hpp>

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

SinBufferedWeightChange::SinBufferedWeightChange():AdditiveKernelChange(), exponent(0),
bufferedActivityTimesNoTrigger(0), kernelLookupTable(0){
	// Set the default values for the learning rule parameters
	this->SetParameters(SinBufferedWeightChange::GetDefaultParameters());
}

SinBufferedWeightChange::~SinBufferedWeightChange(){
	if(bufferedActivityTimesNoTrigger!=0){
		delete bufferedActivityTimesNoTrigger;
	}
	if(kernelLookupTable!=0){
		delete kernelLookupTable;
	}
}


void SinBufferedWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	float tau = this->kernelpeak / atan((float)exponent);
	float factor = 1. / (exp(-atan((float)this->exponent))*pow(sin(atan((float)this->exponent)), (int) this->exponent));

	if (tau == 0){
		tau = 1e-6;
	}

	double step_size = 0.0001;
	double tolerance = 1e-6;

	this->maxTimeMeasured = this->kernelpeak;
	while (1){
		this->maxTimeMeasured += step_size;
		if ((exp(-this->maxTimeMeasured / tau)*pow(sin(this->maxTimeMeasured / tau), double(exponent))*factor) < tolerance){
			break;
		}
	}
	if (this->maxTimeMeasured > this->kernelpeak * 4){
		this->maxTimeMeasured = this->kernelpeak * 4;
	}

	this->N_elements = this->maxTimeMeasured / step_size + 1;
	kernelLookupTable = new float[this->N_elements];


	this->inv_maxTimeMeasured = 1.0f / this->maxTimeMeasured;
	//Precompute the kernel in the look-up table.
	for (int i = 0; i<N_elements; i++){
		double time = maxTimeMeasured*i / (N_elements*tau);
		kernelLookupTable[i] = exp(-time)*pow(sin(time), double(exponent))*factor;
	}

	//Inicitialize de buffer of activity
	bufferedActivityTimesNoTrigger = new BufferedActivityTimes(NumberOfNeurons);
}

int SinBufferedWeightChange::GetNumberOfVar() const{
	return this->exponent+2;
}

int SinBufferedWeightChange::GetExponent() const{
	return this->exponent;
}

ModelDescription SinBufferedWeightChange::ParseLearningRule(FILE * fh) noexcept(false) {
	ModelDescription lrule = AdditiveKernelChange::ParseLearningRule(fh);

	int exponent;
	if(fscanf(fh,"%i",&exponent)!=1){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_SIN_WEIGHT_CHANGE_LOAD);
	}
	if (exponent <= 0 || exponent % 2 == 1 || exponent > 20){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_SIN_WEIGHT_CHANGE_EXPONENT, REPAIR_LEARNING_RULE_VALUES);
	}

	lrule.model_name = SinBufferedWeightChange::GetName();
	lrule.param_map["exp"] = boost::any(exponent);
	return lrule;
}

ostream & SinBufferedWeightChange::PrintInfo(ostream & out){
	out << "- SinAdditiveKernel Learning Rule: " << endl;
	out << "\t kernel_peak:" << this->kernelpeak << endl;
	out << "\t fixed_change: " << this->fixwchange << endl;
	out << "\t kernel_change: " << this->kernelwchange << endl;
	out << "\t exp: " << this->exponent << endl;
	return out;
}

void SinBufferedWeightChange::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("exp");
	if (it!=param_map.end()){
		int newexponent = boost::any_cast<int>(it->second);
		if (newexponent<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_SIN_WEIGHT_CHANGE_EXPONENT,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->exponent = newexponent;
		param_map.erase(it);
	}

	AdditiveKernelChange::SetParameters(param_map);

}

void SinBufferedWeightChange::ApplyPreSynapticSpike(Interconnection * Connection, double SpikeTime){

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


LearningRule* SinBufferedWeightChange::CreateLearningRule(ModelDescription lrDescription){
	SinBufferedWeightChange * lrule = new SinBufferedWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> SinBufferedWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = AdditiveKernelChange::GetParameters();
	newMap["exp"] = boost::any(this->exponent);
	return newMap;
}


std::map<std::string,boost::any> SinBufferedWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = AdditiveKernelChange::GetDefaultParameters();
	newMap["exp"] = boost::any(2);
	return newMap;
}
