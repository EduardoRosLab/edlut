/***************************************************************************
 *                           STDPWeightChange.cpp                          *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#include "../../include/learning_rules/SimetricCosSinSTDPWeightChange.h"

#include "../../include/learning_rules/SimetricCosSinSTDPState.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"
#include "../../include/spike/Neuron.h"


SimetricCosSinSTDPWeightChange::SimetricCosSinSTDPWeightChange():WithPostSynaptic(){
	// Set the default values for the learning rule parameters
    this->SetParameters(SimetricCosSinSTDPWeightChange::GetDefaultParameters());
}

SimetricCosSinSTDPWeightChange::~SimetricCosSinSTDPWeightChange(){

}


void SimetricCosSinSTDPWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	this->State=(ConnectionState *) new SimetricCosSinSTDPState(NumberOfSynapses+NumberOfNeurons, this->MaxMinDistance, this->CentralAmplitudeFactor, this->LateralAmplitudeFactor);
}

void SimetricCosSinSTDPWeightChange::ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime){
	int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();

	// Update the presynaptic activity
	State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

	// Add the presynaptic spike influence
	State->ApplyPresynapticSpike(LearningRuleIndex);


	//LTD
	int SecondLearningRuleIndex = Connection->GetTarget()->GetIndex();

	// Update the presynaptic activity
	State->SetNewUpdateTime(SecondLearningRuleIndex, SpikeTime, false);

	// Update synaptic weight
	Connection->IncrementWeight(State->GetPresynapticActivity(SecondLearningRuleIndex));
}


void SimetricCosSinSTDPWeightChange::ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime){
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
		interi->IncrementWeight(State->GetPresynapticActivity(LearningRuleIndex));
	}

}


ModelDescription SimetricCosSinSTDPWeightChange::ParseLearningRule(FILE * fh) noexcept(false) {
	ModelDescription lrule;

	float lMaxMinDistance, lCentralAmplitudeFactor, lLateralAmplitudeFactor;
	if(fscanf(fh,"%f",&lMaxMinDistance)!=1 ||
	   fscanf(fh,"%f",&lCentralAmplitudeFactor)!=1 ||
	   fscanf(fh,"%f",&lLateralAmplitudeFactor)!=1){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_COS_SIN_WEIGHT_CHANGE_LOAD);
	}
	if (lMaxMinDistance <= 0){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_COS_SIN_WEIGHT_CHANGE_AMPLITUDE, REPAIR_LEARNING_RULE_VALUES);
	}
	if (lLateralAmplitudeFactor <= 0){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_COS_SIN_WEIGHT_CHANGE_SIGNS, REPAIR_LEARNING_RULE_VALUES);
	}

	lrule.model_name = SimetricCosSinSTDPWeightChange::GetName();
	lrule.param_map["max_min_dist"] = boost::any(lMaxMinDistance);
	lrule.param_map["central_amp"] = boost::any(lCentralAmplitudeFactor);
	lrule.param_map["lateral_amp"] = boost::any(lLateralAmplitudeFactor);
	return lrule;
}

void SimetricCosSinSTDPWeightChange::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("max_min_dist");
	if (it!=param_map.end()){
		float newmaxmindistance = boost::any_cast<float>(it->second);
		if (newmaxmindistance<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_COS_SIN_WEIGHT_CHANGE_AMPLITUDE,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->MaxMinDistance = newmaxmindistance;
		param_map.erase(it);
	}

	it=param_map.find("central_amp");
	if (it!=param_map.end()){
		this->CentralAmplitudeFactor = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

	it=param_map.find("lateral_amp");
	if (it!=param_map.end()){
		this->LateralAmplitudeFactor = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

    WithPostSynaptic::SetParameters(param_map);

}

ostream & SimetricCosSinSTDPWeightChange::PrintInfo(ostream & out){
	out << "- SimetricCosSinSTDPAdditiveKernel Learning Rule: " << endl;
	out << "\t max_min_dist:" << this->MaxMinDistance << endl;
	out << "\t central_amp:" << this->CentralAmplitudeFactor << endl;
	out << "\t lateral_amp: " << this->LateralAmplitudeFactor << endl;
	return out;
}

LearningRule* SimetricCosSinSTDPWeightChange::CreateLearningRule(ModelDescription lrDescription){
	SimetricCosSinSTDPWeightChange * lrule = new SimetricCosSinSTDPWeightChange();
    lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> SimetricCosSinSTDPWeightChange::GetParameters(){
    std::map<std::string,boost::any> newMap = WithPostSynaptic::GetParameters();
    newMap["max_min_dist"] = boost::any(this->MaxMinDistance);
    newMap["central_amp"] = boost::any(this->CentralAmplitudeFactor);
    newMap["lateral_amp"] = boost::any(this->LateralAmplitudeFactor);
    return newMap;
}

std::map<std::string,boost::any> SimetricCosSinSTDPWeightChange::GetDefaultParameters(){
    std::map<std::string,boost::any> newMap = WithPostSynaptic::GetDefaultParameters();
    newMap["max_min_dist"] = boost::any(0.050f);
    newMap["central_amp"] = boost::any(0.010f);
    newMap["lateral_amp"] = boost::any(-0.005f);
    return newMap;
}
