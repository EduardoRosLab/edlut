/***************************************************************************
 *                           WithTriggerSynaptic.cpp                       *
 *                           -------------------                           *
 * copyright            : (C) 2023 by Francisco Naveros                    *
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

#include "../../include/learning_rules/WithTriggerSynaptic.h"


WithTriggerSynaptic::WithTriggerSynaptic():LearningRule(){
	this->SetParameters(WithTriggerSynaptic::GetDefaultParameters());
}

WithTriggerSynaptic::~WithTriggerSynaptic(){

}

bool WithTriggerSynaptic::ImplementPostSynaptic(){
	return false;
}

bool WithTriggerSynaptic::ImplementTriggerSynaptic(){
	return true;
}

std::map<std::string,boost::any> WithTriggerSynaptic::GetParameters(){
	std::map<std::string,boost::any> newMap = LearningRule::GetParameters();
	return newMap;
}


std::map<std::string,boost::any> WithTriggerSynaptic::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = LearningRule::GetDefaultParameters();
	return newMap;
}
