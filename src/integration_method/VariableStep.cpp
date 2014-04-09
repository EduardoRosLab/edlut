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


VariableStep::VariableStep(string integrationMethodType, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int N_CPU_thread, bool jacobian, bool inverse):IntegrationMethod(integrationMethodType,N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, N_CPU_thread, jacobian, inverse){
}

VariableStep::~VariableStep(){

}

enum IntegrationMethodType VariableStep::GetMethodType(){
	return VARIABLE_STEP;
}


