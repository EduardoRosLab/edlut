/***************************************************************************
 *                           SimetricCosSTDPWeightChange.cpp               *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#include "../../include/learning_rules/SimetricCosSTDPWeightChange.h"

#include "../../include/learning_rules/SimetricCosSTDPState.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"
#include "../../include/spike/Neuron.h"


SimetricCosSTDPWeightChange::SimetricCosSTDPWeightChange():WithPostSynaptic(){
	this->SetParameters(SimetricCosSTDPWeightChange::GetDefaultParameters());
}

SimetricCosSTDPWeightChange::~SimetricCosSTDPWeightChange(){

}


void SimetricCosSTDPWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	this->State=(ConnectionState *) new SimetricCosSTDPState(NumberOfSynapses+NumberOfNeurons, this->tau, this->exponent);
}

void SimetricCosSTDPWeightChange::ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime){

	int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();

	//LTP
	// Second case: the weight change is linked to this connection
	Connection->IncrementWeight(this->fixwchange);

	// Update the presynaptic activity
	State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

	// Add the presynaptic spike influence
	State->ApplyPresynapticSpike(LearningRuleIndex);


	//LTD
	//aplicate internal spike kernel to "future propagate spike"
	int SecondLearningRuleIndex = Connection->GetTarget()->GetIndex();

	// Update the presynaptic activity
	State->SetNewUpdateTime(SecondLearningRuleIndex, SpikeTime, false);

	// Update synaptic weight
	Connection->IncrementWeight(this->kernelwchange*State->GetPresynapticActivity(SecondLearningRuleIndex));
}


void SimetricCosSTDPWeightChange::ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime){
	int SecondLearningRuleIndex = neuron->GetIndex();
	State->SetNewUpdateTime(SecondLearningRuleIndex, SpikeTime, false);
	State->ApplyPresynapticSpike(SecondLearningRuleIndex);

	unsigned int LearningRuleId = this->GetLearningRuleIndex();

	for (int i = 0; i < neuron->GetInputNumberWithPostSynapticLearning(LearningRuleId); ++i){
		Interconnection * interi = neuron->GetInputConnectionWithPostSynapticLearningAt(LearningRuleId,i);

		//Aplicate propagate spike kernel to "future internal spike"
		int LearningRuleIndex = neuron->IndexInputLearningConnections[0][LearningRuleId][i];

		// Update the presynaptic activity
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

		// Update synaptic weight
		interi->IncrementWeight(this->kernelwchange*State->GetPresynapticActivity(LearningRuleIndex));
	}
}


ModelDescription SimetricCosSTDPWeightChange::ParseLearningRule(FILE * fh) noexcept(false) {
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

	lrule.model_name = SimetricCosSTDPWeightChange::GetName();
	lrule.param_map["tau"] = boost::any(tau);
	lrule.param_map["exp"] = boost::any(exponent);
	lrule.param_map["fixed_change"] = boost::any(fixwchange);
	lrule.param_map["kernel_change"] = boost::any(kernelwchange);
	return lrule;
}

void SimetricCosSTDPWeightChange::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

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

	WithPostSynaptic::SetParameters(param_map);

}

ostream & SimetricCosSTDPWeightChange::PrintInfo(ostream & out){
	out << "- SimetricCosSTDPAdditiveKernel Learning Rule: " << endl;
	out << "\t tau:" << this->tau << endl;
	out << "\t exp:" << this->exponent << endl;
	out << "\t fixed_change: " << this->fixwchange << endl;
	out << "\t kernel_change: " << this->kernelwchange << endl;
	return out;
}

LearningRule* SimetricCosSTDPWeightChange::CreateLearningRule(ModelDescription lrDescription){
	SimetricCosSTDPWeightChange * lrule = new SimetricCosSTDPWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> SimetricCosSTDPWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetParameters();
	newMap["tau"] = boost::any(this->tau);
	newMap["exp"] = boost::any(this->exponent);
	newMap["fixed_change"] = boost::any(this->fixwchange);
	newMap["kernel_change"] = boost::any(this->kernelwchange);
	return newMap;
}


std::map<std::string,boost::any> SimetricCosSTDPWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetDefaultParameters();
	newMap["tau"] = boost::any(0.100f);
	newMap["exp"] = boost::any(2);
	newMap["fixed_change"] = boost::any(0.001f);
	newMap["kernel_change"] = boost::any(-0.010f);
	return newMap;
}
