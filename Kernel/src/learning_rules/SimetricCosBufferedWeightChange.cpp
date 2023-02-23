/***************************************************************************
 *                           SimetricCosBufferedWeightChange.cpp           *
 *                           -------------------                           *
 * copyright            : (C) 2016 by Francisco Naveros and Niceto Luque   *
 * email                : fnaveros@ugr.es nluque@ugr.es                    *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#include "../../include/learning_rules/SimetricCosBufferedWeightChange.h"
#include "../../include/learning_rules/BufferedActivityTimes.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"
#include <boost/any.hpp>

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

SimetricCosBufferedWeightChange::SimetricCosBufferedWeightChange() :WithTriggerSynaptic(),
		bufferedActivityTimesTrigger(0), bufferedActivityTimesNoTrigger(0), kernelLookupTable(0){
	// Set the default values for the learning rule parameters
	this->SetParameters(SimetricCosBufferedWeightChange::GetDefaultParameters());
}

SimetricCosBufferedWeightChange::~SimetricCosBufferedWeightChange(){
	if(bufferedActivityTimesNoTrigger!=0){
		delete bufferedActivityTimesNoTrigger;
	}
	if(bufferedActivityTimesTrigger!=0){
		delete bufferedActivityTimesTrigger;
	}
	if(kernelLookupTable!=0){
		delete kernelLookupTable;
	}
}


void SimetricCosBufferedWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	double step_size = 0.0001;
	double tolerance = 1e-6;
	this->maxTimeMeasured = 0;
	while (1){
		this->maxTimeMeasured += step_size;
		if (exp(-maxTimeMeasured*exponent)*cos(1.5708f*maxTimeMeasured)*cos(1.5708f*maxTimeMeasured)< tolerance){
			break;
		}
	}

	this->N_elements = this->maxTimeMeasured / step_size + 1;
	kernelLookupTable = new float[this->N_elements];
	this->inv_maxTimeMeasured = 1.0f / this->maxTimeMeasured;
	//Precompute the kernel in the look-up table.
	for (int i = 0; i<N_elements; i++){
		double time = maxTimeMeasured*i / (N_elements*tau);
		kernelLookupTable[i] = exp(-time*exponent)*cos(1.5708f*time)*cos(1.5708f*time);
	}

	//Inicitialize de buffer of activity
	bufferedActivityTimesNoTrigger = new BufferedActivityTimes(NumberOfNeurons);
	bufferedActivityTimesTrigger = new BufferedActivityTimes(NumberOfNeurons);
}


ModelDescription SimetricCosBufferedWeightChange::ParseLearningRule(FILE * fh) noexcept(false) {
	ModelDescription lrule;

	int exponent;
	float tau, fixwchange, kernelwchange;
	if(fscanf(fh,"%f",&tau)!=1 ||
	   fscanf(fh,"%d",&exponent)!=1 ||
	   fscanf(fh,"%f",&fixwchange)!=1 ||
	   fscanf(fh,"%f",&kernelwchange)!=1){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_COS_WEIGHT_CHANGE_LOAD);
	}
	if (tau <= 0){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_COS_WEIGHT_CHANGE_TAU, REPAIR_LEARNING_RULE_VALUES);
	}
	if (exponent <= 0){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_COS_WEIGHT_CHANGE_EXPONENT, REPAIR_LEARNING_RULE_VALUES);
	}

	lrule.model_name = SimetricCosBufferedWeightChange::GetName();
	lrule.param_map["tau"] = boost::any(tau);
	lrule.param_map["exp"] = boost::any(exponent);
	lrule.param_map["fixed_change"] = boost::any(fixwchange);
	lrule.param_map["kernel_change"] = boost::any(kernelwchange);
	return lrule;
}

void SimetricCosBufferedWeightChange::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("tau");
	if (it!=param_map.end()){
		float newkernelpeak = boost::any_cast<float>(it->second);
		if (newkernelpeak <= 0){
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_COS_WEIGHT_CHANGE_TAU, REPAIR_LEARNING_RULE_VALUES);
		}
		this->tau = newkernelpeak;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	it=param_map.find("exp");
	if (it!=param_map.end()){
		int newexponent = boost::any_cast<int>(it->second);
		if (newexponent<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_COS_WEIGHT_CHANGE_EXPONENT,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->exponent = newexponent;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	it=param_map.find("fixed_change");
	if (it!=param_map.end()){
		this->fixwchange = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	it=param_map.find("kernel_change");
	if (it!=param_map.end()){
		this->kernelwchange = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

	WithTriggerSynaptic::SetParameters(param_map);

}

void SimetricCosBufferedWeightChange::ApplyPreSynapticSpike(Interconnection * Connection, double SpikeTime){

	if (Connection->GetTriggerConnection() == false){
		//LTP
		Connection->IncrementWeight(this->fixwchange);
		Neuron * TargetNeuron = Connection->GetTarget();
		int neuron_index = TargetNeuron->GetIndex();
		int synapse_index = Connection->LearningRuleIndex_withTrigger_insideTargetNeuron;
		this->bufferedActivityTimesNoTrigger->InsertElement(neuron_index, SpikeTime, SpikeTime - this->maxTimeMeasured, synapse_index);


		//LTD
		int N_elements = bufferedActivityTimesTrigger->ProcessElements(neuron_index, SpikeTime - this->maxTimeMeasured);
		SpikeData * spike_data = bufferedActivityTimesTrigger->GetOutputSpikeData();

		float value = 0;
		for (int i = 0; i < N_elements; i++){
			double ElapsedTime = SpikeTime - spike_data[i].time;
			int tableIndex = ElapsedTime*this->N_elements*this->inv_maxTimeMeasured;
			value += this->kernelLookupTable[tableIndex];
		}
		Connection->IncrementWeight(this->kernelwchange*value);
	}
	else{
		Neuron * TargetNeuron = Connection->GetTarget();
		int neuron_index = TargetNeuron->GetIndex();
		this->bufferedActivityTimesTrigger->InsertElement(neuron_index, SpikeTime, SpikeTime - this->maxTimeMeasured, 0);
		unsigned int LearningRuleId = this->GetLearningRuleIndex();


		//LTD
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



ostream & SimetricCosBufferedWeightChange::PrintInfo(ostream & out){

	out << "- SimetricCosBufferedAdditiveKernel Learning Rule: " << endl;
	out << "\t tau:" << this->tau << endl;
	out << "\t exp:" << this->exponent << endl;
	out << "\t fixed_change: " << this->fixwchange << endl;
	out << "\t kernel_change: " << this->kernelwchange << endl;
	return out;
}

LearningRule* SimetricCosBufferedWeightChange::CreateLearningRule(ModelDescription lrDescription){
	SimetricCosBufferedWeightChange * lrule = new SimetricCosBufferedWeightChange();
    lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> SimetricCosBufferedWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = WithTriggerSynaptic::GetParameters();
	newMap["tau"] = boost::any(this->tau);
	newMap["exp"] = boost::any(this->exponent);
	newMap["fixed_change"] = boost::any(this->fixwchange);
	newMap["kernel_change"] = boost::any(this->kernelwchange);
	return newMap;
}


std::map<std::string,boost::any> SimetricCosBufferedWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = WithTriggerSynaptic::GetDefaultParameters();
	newMap["tau"] = boost::any(0.100f);
	newMap["exp"] = boost::any(2);
	newMap["fixed_change"] = boost::any(0.001f);
	newMap["kernel_change"] = boost::any(-0.010f);
	return newMap;
}
