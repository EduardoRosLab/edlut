/***************************************************************************
 *                           NeuronModel.cpp                               *
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

#include "../../include/neuron_model/NeuronModel.h"

NeuronModel::NeuronModel(string NeuronModelID): ModelID(NeuronModelID) {
	// TODO Auto-generated constructor stub

}

NeuronModel::~NeuronModel() {
	// TODO Auto-generated destructor stub
}

string NeuronModel::GetModelID(){
	return this->ModelID;
}
