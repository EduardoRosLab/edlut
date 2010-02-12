/***************************************************************************
 *                           SRMModel.cpp                                  *
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

#include "../../include/neuron_model/SRMModel.h"

SRMModel::SRMModel(string NeuronModelID): NeuronModel(NeuronModelID) {
	// TODO Auto-generated constructor stub

}

SRMModel::~SRMModel() {
	// TODO Auto-generated destructor stub
}

BufferedState * SRMModel::InitializeState(){

}

void SRMModel::UpdateState(BufferedState & State, double ElapsedTime){

}

void SRMModel::SynapsisEffect(BufferedState & State, const Interconnection * InputConnection){

}

double SRMModel::NextFiringPrediction(BufferedState & State){

}

