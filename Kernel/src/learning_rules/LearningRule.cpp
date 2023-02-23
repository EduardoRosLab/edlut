/***************************************************************************
 *                           LearningRule.cpp                              *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
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

#include "../../include/learning_rules/LearningRule.h"


LearningRule::LearningRule(): learningRuleIndex(-1), State(0), counter(0){
	this->SetParameters(LearningRule::GetDefaultParameters());
}

LearningRule::~LearningRule(){
	if(State!=0){
		delete State;
	}
}

void LearningRule::SetParameters(std::map<std::string, boost::any> param_map) noexcept(false){
	if (!param_map.empty()){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD,ERROR_LEARNING_RULE_UNKNOWN_PARAMETER, REPAIR_LEARNING_RULE_PARAMETER_NAME);
	}
}

ConnectionState * LearningRule::GetConnectionState(){
	return this->State;
}

std::map<std::string,boost::any> LearningRule::GetParameters(){
	// Return an empty dictionary
	return std::map<std::string,boost::any> ();
}

std::map<std::string,boost::any> LearningRule::GetDefaultParameters(){
	return std::map<std::string,boost::any> ();
}
