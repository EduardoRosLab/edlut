/***************************************************************************
 *                           CosWeightChange.cpp                           *
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


#include "../../include/learning_rules/SimetricCosWeightChange.h"

#include "../../include/learning_rules/SimetricCosState.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

SimetricCosWeightChange::SimetricCosWeightChange():WithTriggerSynaptic(){
	this->SetParameters(SimetricCosWeightChange::GetDefaultParameters());
}

SimetricCosWeightChange::~SimetricCosWeightChange(){

}


void SimetricCosWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	this->State=(ConnectionState *) new SimetricCosState(NumberOfSynapses, this->tau, this->exponent);
}


ModelDescription SimetricCosWeightChange::ParseLearningRule(FILE * fh) noexcept(false) {
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

	lrule.model_name = SimetricCosWeightChange::GetName();
	lrule.param_map["tau"] = boost::any(tau);
	lrule.param_map["exp"] = boost::any(exponent);
	lrule.param_map["fixed_change"] = boost::any(fixwchange);
	lrule.param_map["kernel_change"] = boost::any(kernelwchange);
	return lrule;
}

void SimetricCosWeightChange::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

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

void SimetricCosWeightChange::ApplyPreSynapticSpike(Interconnection * Connection, double SpikeTime){
	unsigned int LearningRuleId = this->GetLearningRuleIndex();
	if (Connection->GetTriggerConnection() == false){
		int LearningRuleIndex = Connection->GetLearningRuleIndex_withTrigger();

		//LTP
		// Second case: the weight change is linked to this connection
		Connection->IncrementWeight(this->fixwchange);

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
				Connection->IncrementWeight(this->kernelwchange*State->GetPresynapticActivity(LearningRuleIndex));
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
		unsigned int LearningRuleId = this->GetLearningRuleIndex();

		for (int i = 0; i<TargetNeuron->GetInputNumberWithTriggerSynapticLearning(LearningRuleId); ++i){
			//We implement this weight increment mechanisme in order to improve the cache friendly. Thus we do not need to red a large number of none consecutive synapses from memory.
			//This weight increments will be acumulated in the synaptic weights when a spike will be propagated for this synapses.
			int LearningRuleIndex = TargetNeuron->IndexInputLearningConnections[1][LearningRuleId][i];

			if (LearningRuleIndex >= 0){
				Interconnection * interi = TargetNeuron->GetInputConnectionWithTriggerSynapticLearningAt(LearningRuleId,i);

				// Update the presynaptic activity
				State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

				// Update synaptic weight
				interi->IncrementWeight(this->kernelwchange*State->GetPresynapticActivity(LearningRuleIndex));
			}
		}
	}
}



ostream & SimetricCosWeightChange::PrintInfo(ostream & out){
	out << "- SimetricCosAdditiveKernel Learning Rule: " << endl;
	out << "\t tau:" << this->tau << endl;
	out << "\t exp:" << this->exponent << endl;
	out << "\t fixed_change: " << this->fixwchange << endl;
	out << "\t kernel_change: " << this->kernelwchange << endl;
	return out;
}

LearningRule* SimetricCosWeightChange::CreateLearningRule(ModelDescription lrDescription){
	SimetricCosWeightChange * lrule = new SimetricCosWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> SimetricCosWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = WithTriggerSynaptic::GetParameters();
	newMap["tau"] = boost::any(this->tau);
	newMap["exp"] = boost::any(this->exponent);
	newMap["fixed_change"] = boost::any(this->fixwchange);
	newMap["kernel_change"] = boost::any(this->kernelwchange);
	return newMap;
}


std::map<std::string,boost::any> SimetricCosWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = WithTriggerSynaptic::GetDefaultParameters();
	newMap["tau"] = boost::any(0.100f);
	newMap["exp"] = boost::any(2);
	newMap["fixed_change"] = boost::any(0.001f);
	newMap["kernel_change"] = boost::any(-0.010f);
	return newMap;
}
