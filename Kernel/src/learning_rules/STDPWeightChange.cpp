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

#include "../../include/learning_rules/STDPWeightChange.h"

#include "../../include/learning_rules/STDPState.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"


STDPWeightChange::STDPWeightChange():WithPostSynaptic(){
	// Set the default values for the learning rule parameters
	this->SetParameters(STDPWeightChange::GetDefaultParameters());
}

STDPWeightChange::~STDPWeightChange(){

}


void STDPWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	this->State=(ConnectionState *) new STDPState(NumberOfSynapses, this->tauLTP, this->tauLTD);
}

void STDPWeightChange::ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime){
	unsigned int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();

	// Apply synaptic activity decaying rule
	State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

	// Apply presynaptic spike
	State->ApplyPresynapticSpike(LearningRuleIndex);

	// Apply weight change
	Connection->IncrementWeight(-this->MaxChangeLTD*State->GetPostsynapticActivity(LearningRuleIndex));

	return;
}

void STDPWeightChange::ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime){
	unsigned int LearningRuleId = this->GetLearningRuleIndex();
	for (int i = 0; i<neuron->GetInputNumberWithPostSynapticLearning(LearningRuleId); ++i){
		Interconnection * interi = neuron->GetInputConnectionWithPostSynapticLearningAt(LearningRuleId,i);

		unsigned int LearningRuleIndex = neuron->IndexInputLearningConnections[0][LearningRuleId][i];

		// Apply synaptic activity decaying rule
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, true);

		// Apply postsynaptic spike
		State->ApplyPostsynapticSpike(LearningRuleIndex);

		// Update synaptic weight
		float WeightChange = this->MaxChangeLTP*State->GetPresynapticActivity(LearningRuleIndex);
		interi->IncrementWeight(WeightChange);
	}

}

ModelDescription STDPWeightChange::ParseLearningRule(FILE * fh) noexcept(false) {
	ModelDescription lrule;

	float ltauLTP, lMaxChangeLTP, ltauLTD, lMaxChangeLTD;
	if(fscanf(fh,"%f",&lMaxChangeLTP)!=1 ||
	   fscanf(fh,"%f",&ltauLTP)!=1 ||
	   fscanf(fh,"%f",&lMaxChangeLTD)!=1 ||
	   fscanf(fh,"%f",&ltauLTD)!=1){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_STDP_WEIGHT_CHANGE_LOAD);
	}
	if (ltauLTP <= 0 || ltauLTD <= 0){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_STDP_WEIGHT_CHANGE_TAUS, REPAIR_LEARNING_RULE_VALUES);
	}

	lrule.model_name = STDPWeightChange::GetName();
	lrule.param_map["max_LTP"] = boost::any(lMaxChangeLTP);
	lrule.param_map["tau_LTP"] = boost::any(ltauLTP);
	lrule.param_map["max_LTD"] = boost::any(lMaxChangeLTD);
	lrule.param_map["tau_LTD"] = boost::any(ltauLTD);

	return lrule;
}

void STDPWeightChange::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){

	// Search for the parameters in the dictionary
	std::map<std::string, boost::any>::iterator it = param_map.find("max_LTP");
	if (it != param_map.end()){
		this->MaxChangeLTP = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	it=param_map.find("tau_LTP");
	if (it!=param_map.end()){
		float newtauLTP = boost::any_cast<float>(it->second);
		if (newtauLTP<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_STDP_WEIGHT_CHANGE_TAUS,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->tauLTP = newtauLTP;
		param_map.erase(it);
	}

	it = param_map.find("max_LTD");
	if (it != param_map.end()){
		this->MaxChangeLTD = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	it=param_map.find("tau_LTD");
	if (it!=param_map.end()){
		float newtauLTD = boost::any_cast<float>(it->second);
		if (newtauLTD<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_STDP_WEIGHT_CHANGE_TAUS,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->tauLTD = newtauLTD;
		param_map.erase(it);
	}

	WithPostSynaptic::SetParameters(param_map);
}


ostream & STDPWeightChange::PrintInfo(ostream & out){
	out << "- STDP Learning Rule: " << endl;
	out << "\t max_LTP:" << this->MaxChangeLTP << endl;
	out << "\t tau_LTP:" << this->tauLTP << endl;
	out << "\t max_LTD:" << this->MaxChangeLTD << endl;
	out << "\t tau_LTD:" << this->tauLTD << endl;
	return out;
}

float STDPWeightChange::GetMaxWeightChangeLTP() const{
	return this->MaxChangeLTP;
}

void STDPWeightChange::SetMaxWeightChangeLTP(float NewMaxChange){
	this->MaxChangeLTP = NewMaxChange;
}

float STDPWeightChange::GetMaxWeightChangeLTD() const{
	return this->MaxChangeLTD;
}

void STDPWeightChange::SetMaxWeightChangeLTD(float NewMaxChange){
	this->MaxChangeLTD = NewMaxChange;
}

LearningRule* STDPWeightChange::CreateLearningRule(ModelDescription lrDescription){
	STDPWeightChange * lrule = new STDPWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> STDPWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetParameters();
	newMap["max_LTP"] = boost::any(this->MaxChangeLTP);
	newMap["tau_LTP"] = boost::any(this->tauLTP);
	newMap["max_LTD"] = boost::any(this->MaxChangeLTD);
	newMap["tau_LTD"] = boost::any(this->tauLTD);
	return newMap;
}

std::map<std::string,boost::any> STDPWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetDefaultParameters();
	newMap["max_LTP"] = boost::any(0.010f);
	newMap["tau_LTP"] = boost::any(0.100f);
	newMap["max_LTD"] = boost::any(0.020f);
	newMap["tau_LTD"] = boost::any(0.100f);
	return newMap;
}
