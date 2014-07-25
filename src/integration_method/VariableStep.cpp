/***************************************************************************
 *                           VariableStep.cpp                              *
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


#include "../../include/integration_method/VariableStep.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"


VariableStep::VariableStep(TimeDrivenNeuronModel * NewModel, string integrationMethodType, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, bool jacobian, bool inverse):IntegrationMethod(NewModel,integrationMethodType,N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, jacobian, inverse){
}

VariableStep::~VariableStep(){

}

enum IntegrationMethodType VariableStep::GetMethodType(){
	return VARIABLE_STEP;
}


