/***************************************************************************
 *                           NeuronModel.cpp                               *
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

#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/neuron_model/NeuronState.h"

NeuronModel::NeuronModel(string NeuronModelID): ModelID(NeuronModelID), InitialState(0) {
	// TODO Auto-generated constructor stub

}

NeuronModel::~NeuronModel() {
	// TODO Auto-generated destructor stub
	if (this->InitialState!=0){
		delete this->InitialState;
	}
}

string NeuronModel::GetModelID(){
	return this->ModelID;
}
