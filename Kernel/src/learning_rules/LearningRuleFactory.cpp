/***************************************************************************
 *                           LearningRuleFactory.cpp                       *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Jesus Garrido                        *
 * email                : jesusgarrido@ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "learning_rules/LearningRuleFactory.h"

#include "learning_rules/ExpBufferedWeightChange.h"
#include "learning_rules/ExpWeightChange.h"
#include "learning_rules/CosWeightChange.h"
#include "learning_rules/SimetricCosBufferedWeightChange.h"
#include "learning_rules/SimetricCosSinSTDPWeightChange.h"
#include "learning_rules/SimetricCosSinWeightChange.h"
#include "learning_rules/SimetricCosSTDPWeightChange.h"
#include "learning_rules/SimetricCosWeightChange.h"
#include "learning_rules/SinBufferedWeightChange.h"
#include "learning_rules/SinWeightChange.h"
#include "learning_rules/STDPLSWeightChange.h"
#include "learning_rules/STDPWeightChange.h"


std::map<std::string, LearningRuleFactory::LearningRuleClass > LearningRuleFactory::LearningRuleMap;

std::vector<std::string> LearningRuleFactory::GetAvailableLearningRules(){
	std::vector<std::string> availableLearningRules;
	availableLearningRules.push_back(ExpBufferedWeightChange::GetName());
	availableLearningRules.push_back(ExpWeightChange::GetName());
	availableLearningRules.push_back(CosWeightChange::GetName());
	availableLearningRules.push_back(SimetricCosBufferedWeightChange::GetName());
	availableLearningRules.push_back(SimetricCosSinSTDPWeightChange::GetName());
	availableLearningRules.push_back(SimetricCosSinWeightChange::GetName());
	availableLearningRules.push_back(SimetricCosSTDPWeightChange::GetName());
	availableLearningRules.push_back(SimetricCosWeightChange::GetName());
	availableLearningRules.push_back(SinBufferedWeightChange::GetName());
	availableLearningRules.push_back(SinWeightChange::GetName());
	availableLearningRules.push_back(STDPLSWeightChange::GetName());
	availableLearningRules.push_back(STDPWeightChange::GetName());

	return availableLearningRules;
}

void LearningRuleFactory::PrintAvailableLearningRules(){
	std::vector<std::string> availableLearningRules = GetAvailableLearningRules();
	cout << "- Available Learning Rules in EDLUT:" << endl;
	for (std::vector<std::string>::const_iterator it = availableLearningRules.begin(); it != availableLearningRules.end(); ++it){
		std::cout << "\t" << *it << std::endl;
	}
}

void LearningRuleFactory::InitializeLearningRuleFactory(){
    if (LearningRuleFactory::LearningRuleMap.empty()){
		LearningRuleFactory::LearningRuleMap[ExpBufferedWeightChange::GetName()] =
			LearningRuleFactory::LearningRuleClass(
			ExpBufferedWeightChange::ParseLearningRule,
			ExpBufferedWeightChange::CreateLearningRule,
			ExpBufferedWeightChange::GetDefaultParameters);
        LearningRuleFactory::LearningRuleMap[ExpWeightChange::GetName()] =
                LearningRuleFactory::LearningRuleClass(
                        ExpWeightChange::ParseLearningRule,
                        ExpWeightChange::CreateLearningRule,
                        ExpWeightChange::GetDefaultParameters);
        LearningRuleFactory::LearningRuleMap[CosWeightChange::GetName()] =
                LearningRuleFactory::LearningRuleClass(
                        CosWeightChange::ParseLearningRule,
                        CosWeightChange::CreateLearningRule,
                        CosWeightChange::GetDefaultParameters);
        LearningRuleFactory::LearningRuleMap[SimetricCosBufferedWeightChange::GetName()] =
                LearningRuleFactory::LearningRuleClass(
                        SimetricCosBufferedWeightChange::ParseLearningRule,
                        SimetricCosBufferedWeightChange::CreateLearningRule,
                        SimetricCosBufferedWeightChange::GetDefaultParameters);
        LearningRuleFactory::LearningRuleMap[SimetricCosSinSTDPWeightChange::GetName()] =
                LearningRuleFactory::LearningRuleClass(
                        SimetricCosSinSTDPWeightChange::ParseLearningRule,
                        SimetricCosSinSTDPWeightChange::CreateLearningRule,
                        SimetricCosSinSTDPWeightChange::GetDefaultParameters);
        LearningRuleFactory::LearningRuleMap[SimetricCosSinWeightChange::GetName()] =
                LearningRuleFactory::LearningRuleClass(
                        SimetricCosSinWeightChange::ParseLearningRule,
                        SimetricCosSinWeightChange::CreateLearningRule,
                        SimetricCosSinWeightChange::GetDefaultParameters);
        LearningRuleFactory::LearningRuleMap[SimetricCosSTDPWeightChange::GetName()] =
                LearningRuleFactory::LearningRuleClass(
                        SimetricCosSTDPWeightChange::ParseLearningRule,
                        SimetricCosSTDPWeightChange::CreateLearningRule,
                        SimetricCosSTDPWeightChange::GetDefaultParameters);
        LearningRuleFactory::LearningRuleMap[SimetricCosWeightChange::GetName()] =
                LearningRuleFactory::LearningRuleClass(
                        SimetricCosWeightChange::ParseLearningRule,
                        SimetricCosWeightChange::CreateLearningRule,
                        SimetricCosWeightChange::GetDefaultParameters);
        LearningRuleFactory::LearningRuleMap[SinBufferedWeightChange::GetName()] =
                LearningRuleFactory::LearningRuleClass(
                        SinBufferedWeightChange::ParseLearningRule,
                        SinBufferedWeightChange::CreateLearningRule,
                        SinBufferedWeightChange::GetDefaultParameters);
        LearningRuleFactory::LearningRuleMap[SinWeightChange::GetName()] =
                LearningRuleFactory::LearningRuleClass(
                        SinWeightChange::ParseLearningRule,
                        SinWeightChange::CreateLearningRule,
                        SinWeightChange::GetDefaultParameters);
        LearningRuleFactory::LearningRuleMap[STDPLSWeightChange::GetName()] =
                LearningRuleFactory::LearningRuleClass(
                        STDPLSWeightChange::ParseLearningRule,
                        STDPLSWeightChange::CreateLearningRule,
                        STDPLSWeightChange::GetDefaultParameters);
        LearningRuleFactory::LearningRuleMap[STDPWeightChange::GetName()] =
                LearningRuleFactory::LearningRuleClass(
                        STDPWeightChange::ParseLearningRule,
                        STDPWeightChange::CreateLearningRule,
                        STDPWeightChange::GetDefaultParameters);
    }
}

LearningRule * LearningRuleFactory::CreateLearningRule(ModelDescription lruleDescription){
    // Find the rule description name in the learning rule map
    LearningRuleFactory::InitializeLearningRuleFactory();
    std::map<std::string, LearningRuleFactory::LearningRuleClass >::const_iterator it =
            LearningRuleFactory::LearningRuleMap.find(lruleDescription.model_name);
    if (it==LearningRuleFactory::LearningRuleMap.end()){
        throw EDLUTException(TASK_NETWORK_LOAD_LEARNING_RULES, ERROR_NETWORK_LEARNING_RULE_NAME,
                                 REPAIR_NETWORK_LEARNING_RULE_NAME);
    }
    LearningRule * lrule = it->second.createFunc(lruleDescription);
    return lrule;
}

ModelDescription LearningRuleFactory::ParseLearningRule(std::string ident, FILE * fh) {
    LearningRuleFactory::InitializeLearningRuleFactory();
    // Find the rule description name in the learning rule map
    std::map<std::string, LearningRuleFactory::LearningRuleClass >::const_iterator it =
            LearningRuleFactory::LearningRuleMap.find(ident);
    if (it == LearningRuleFactory::LearningRuleMap.end()) {
        throw EDLUTException(TASK_NETWORK_LOAD_LEARNING_RULES, ERROR_NETWORK_LEARNING_RULE_NAME,
                             REPAIR_NETWORK_LEARNING_RULE_NAME);
    }
    return it->second.parseFunc(fh);
}

std::map<std::string,boost::any> LearningRuleFactory::GetDefaultParameters(std::string ident){
    LearningRuleFactory::InitializeLearningRuleFactory();
    // Find the rule description name in the learning rule map
    std::map<std::string, LearningRuleFactory::LearningRuleClass >::const_iterator it =
            LearningRuleFactory::LearningRuleMap.find(ident);
    if (it == LearningRuleFactory::LearningRuleMap.end()) {
        throw EDLUTException(TASK_NETWORK_LOAD_LEARNING_RULES, ERROR_NETWORK_LEARNING_RULE_NAME,
                             REPAIR_NETWORK_LEARNING_RULE_NAME);
    }
    return it->second.getDefaultParamFunc();
}
