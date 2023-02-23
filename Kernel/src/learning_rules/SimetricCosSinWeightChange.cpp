/***************************************************************************
 *                           SimetricCosSinWeightChange.cpp                *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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


#include "../../include/learning_rules/SimetricCosSinWeightChange.h"

#include "../../include/learning_rules/SimetricCosSinState.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

SimetricCosSinWeightChange::SimetricCosSinWeightChange():WithTriggerSynaptic(){
	// Set the default values for the learning rule parameters
	this->SetParameters(SimetricCosSinWeightChange::GetDefaultParameters());
}

SimetricCosSinWeightChange::~SimetricCosSinWeightChange(){

}


void SimetricCosSinWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	this->State=(ConnectionState *) new SimetricCosSinState(NumberOfSynapses, this->MaxMinDistance, this->CentralAmplitudeFactor, this->LateralAmplitudeFactor);
}


ModelDescription SimetricCosSinWeightChange::ParseLearningRule(FILE * fh) noexcept(false) {
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

	lrule.model_name = SimetricCosSinWeightChange::GetName();
	lrule.param_map["max_min_dist"] = boost::any(lMaxMinDistance);
	lrule.param_map["central_amp"] = boost::any(lCentralAmplitudeFactor);
	lrule.param_map["lateral_amp"] = boost::any(lLateralAmplitudeFactor);
	return lrule;
}

void SimetricCosSinWeightChange::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

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

	WithTriggerSynaptic::SetParameters(param_map);

}

void SimetricCosSinWeightChange::ApplyPreSynapticSpike(Interconnection * Connection, double SpikeTime){
	unsigned int LearningRuleId = this->GetLearningRuleIndex();

	if (Connection->GetTriggerConnection() == false){
		int LearningRuleIndex = Connection->GetLearningRuleIndex_withTrigger();

		// Update the presynaptic activity
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

		// Add the presynaptic spike influence
		State->ApplyPresynapticSpike(LearningRuleIndex);


		//LTD
		int N_TriggerConnection = Connection->GetTarget()->GetN_TriggerConnectionPerRule(LearningRuleId);
		if (N_TriggerConnection>0){
			Interconnection ** inter = Connection->GetTarget()->GetTriggerConnectionPerRule(LearningRuleId);
			for (int i = 0; i<N_TriggerConnection; i++){
				// Apply sinaptic plasticity driven by teaching signal
				int LearningRuleIndex = inter[i]->GetLearningRuleIndex_withTrigger();

				// Update the presynaptic activity
				State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

				// Update synaptic weight
				Connection->IncrementWeight(State->GetPresynapticActivity(LearningRuleIndex));
			}
		}
	}
	else{
		int LearningRuleIndex = Connection->GetLearningRuleIndex_withTrigger();

		// Update the presynaptic activity
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);
		// Add the presynaptic spike influence
		State->ApplyPresynapticSpike(LearningRuleIndex);



		Neuron * TargetNeuron = Connection->GetTarget();

		for (int i = 0; i<TargetNeuron->GetInputNumberWithTriggerSynapticLearning(LearningRuleId); ++i){
			//We implement this weight increment mechanisme in order to improve the cache friendly. Thus we do not need to red a large number of none consecutive synapses from memory.
			//This weight increments will be acumulated in the synaptic weights when a spike will be propagated for this synapses.
			int LearningRuleIndex = TargetNeuron->IndexInputLearningConnections[1][LearningRuleId][i];

			if (LearningRuleIndex >= 0){
				Interconnection * interi = TargetNeuron->GetInputConnectionWithTriggerSynapticLearningAt(LearningRuleId,i);

				// Update the presynaptic activity
				State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

				// Update synaptic weight
				interi->IncrementWeight(State->GetPresynapticActivity(LearningRuleIndex));
			}
		}
	}
}


ostream & SimetricCosSinWeightChange::PrintInfo(ostream & out){
	out << "- SimetricCosSinAdditiveKernel Learning Rule: " << endl;
	out << "\t max_min_dist:" << this->MaxMinDistance << endl;
	out << "\t central_amp:" << this->CentralAmplitudeFactor << endl;
	out << "\t lateral_amp: " << this->LateralAmplitudeFactor << endl;
	return out;
}

LearningRule* SimetricCosSinWeightChange::CreateLearningRule(ModelDescription lrDescription){
	SimetricCosSinWeightChange * lrule = new SimetricCosSinWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> SimetricCosSinWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = WithTriggerSynaptic::GetParameters();
	newMap["max_min_dist"] = boost::any(this->MaxMinDistance);
	newMap["central_amp"] = boost::any(this->CentralAmplitudeFactor);
	newMap["lateral_amp"] = boost::any(this->LateralAmplitudeFactor);
	return newMap;
}


std::map<std::string,boost::any> SimetricCosSinWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = WithTriggerSynaptic::GetDefaultParameters();
	newMap["max_min_dist"] = boost::any(0.050f);
	newMap["central_amp"] = boost::any(0.010f);
	newMap["lateral_amp"] = boost::any(-0.005f);
	return newMap;
}
