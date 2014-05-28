/***************************************************************************
 *                           WithoutPostSynaptic.cpp                       *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
 * email                : fnaveros@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/learning_rules/WithoutPostSynaptic.h"


WithoutPostSynaptic::WithoutPostSynaptic():LearningRule(), trigger(0){
}

WithoutPostSynaptic::~WithoutPostSynaptic(){

}

void WithoutPostSynaptic::ApplyPostSynapticSpike(Interconnection * Connection, double SpikeTime){
	return;
}

bool WithoutPostSynaptic::ImplementPostSynaptic(){
	return false;
}



