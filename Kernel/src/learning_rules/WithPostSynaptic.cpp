/***************************************************************************
 *                           WithPostSynaptic.cpp                          *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
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

#include "../../include/learning_rules/WithPostSynaptic.h"


WithPostSynaptic::WithPostSynaptic():LearningRule(){

}

WithPostSynaptic::~WithPostSynaptic(){

}

bool WithPostSynaptic::ImplementPostSynaptic(){
	return true;
}

bool WithPostSynaptic::ImplementTriggerSynaptic(){
	return false;
}

std::map<std::string,boost::any> WithPostSynaptic::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = LearningRule::GetDefaultParameters();
	return newMap;
}
