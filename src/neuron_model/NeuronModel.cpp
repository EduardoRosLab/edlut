/***************************************************************************
 *                           NeuronModel.cpp                               *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@atc.ugr.es         *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/neuron_model/VectorNeuronState.h"

NeuronModel::NeuronModel(string NeuronTypeID, string NeuronModelID): TypeID(NeuronTypeID),ModelID(NeuronModelID), InitialState(0) {
	// TODO Auto-generated constructor stub

}

NeuronModel::~NeuronModel() {
	// TODO Auto-generated destructor stub
	if (this->InitialState!=0){
		delete this->InitialState;
	}
}

string NeuronModel::GetTypeID(){
	return this->TypeID;
}

string NeuronModel::GetModelID(){
	return this->ModelID;
}

//VectorNeuronState * NeuronModel::GetVectorNeuronState(){
//	return this->InitialState;
//}