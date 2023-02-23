/***************************************************************************
 *                           WithPostAndTriggerSynaptic.cpp                *
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

#include "../../include/learning_rules/WithPostAndTriggerSynaptic.h"


WithPostAndTriggerSynaptic::WithPostAndTriggerSynaptic():LearningRule(){
	this->SetParameters(WithPostAndTriggerSynaptic::GetDefaultParameters());
}

WithPostAndTriggerSynaptic::~WithPostAndTriggerSynaptic(){

}

bool WithPostAndTriggerSynaptic::ImplementPostSynaptic(){
	return true;
}

bool WithPostAndTriggerSynaptic::ImplementTriggerSynaptic(){
	return true;
}

std::map<std::string,boost::any> WithPostAndTriggerSynaptic::GetParameters(){
	std::map<std::string,boost::any> newMap = LearningRule::GetParameters();
	return newMap;
}


std::map<std::string,boost::any> WithPostAndTriggerSynaptic::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = LearningRule::GetDefaultParameters();
	return newMap;
}
