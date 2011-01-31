/***************************************************************************
 *                           LIFTimeDrivenModelRK.cpp                      *
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

#include "../../include/neuron_model/LIFTimeDrivenModelRK.h"
#include "../../include/neuron_model/NeuronState.h"

#include <iostream>
#include <cmath>
#include <string>


LIFTimeDrivenModelRK::LIFTimeDrivenModelRK(string NeuronModelID): LIFTimeDrivenModel(NeuronModelID) {
}

LIFTimeDrivenModelRK::~LIFTimeDrivenModelRK(){

}
		
bool LIFTimeDrivenModelRK::UpdateState(NeuronState * State, double CurrentTime){

	float last_update = State->GetLastUpdateTime();
	
	float elapsed_time = CurrentTime - last_update;

	State->AddElapsedTime(elapsed_time);
	
	float last_spike = State->GetLastSpikeTime();

	float vm = State->GetStateVariableAt(0);
	float gexc = State->GetStateVariableAt(1);
	float ginh = State->GetStateVariableAt(2);

	bool spike = false;

	if (last_spike > this->tref) {
		// 4th order Runge-Kutta terms
		// 1st term
		float k1 = (gexc * (this->eexc - vm) + ginh * (this->einh - vm) + grest * (this->erest-vm))/this->cm;

		// 2nd term
		float gexcaux = gexc * exp(-((elapsed_time/2)/this->texc));
		float ginhaux = ginh * exp(-((elapsed_time/2)/this->tinh));
		float yaux = vm+(k1*elapsed_time/2);
		float k2 = (gexcaux * (this->eexc - yaux) + ginhaux * (this->einh - yaux) + grest * (this->erest - yaux))/this->cm;

		// 3rd term
		gexcaux = gexc * exp(-((elapsed_time/2)/this->texc));
		ginhaux = ginh * exp(-((elapsed_time/2)/this->tinh));
		yaux = vm+(k2*elapsed_time/2);
		float k3 = (gexcaux * (this->eexc - yaux) + ginhaux * (this->einh - yaux) + grest * (this->erest - yaux))/this->cm;

		// 4rd term
		gexcaux = gexc * exp(-(elapsed_time/2)/this->texc);
		ginhaux = ginh * exp(-(elapsed_time/2)/this->tinh);
		yaux = vm+(k3*elapsed_time);
		float k4 = (gexcaux * (this->eexc - yaux) + ginhaux * (this->einh - yaux) + grest * (this->erest - yaux))/this->cm;

		vm = vm + (k1+2*k2+2*k3+k4)*elapsed_time/6;

		if (vm > this->vthr){
			State->NewFiredSpike();
			spike = true;
			vm = this->erest;
		}
	}
	gexc = gexc * exp(-(elapsed_time/this->texc));
	ginh = ginh * exp(-(elapsed_time/this->tinh));

	State->SetStateVariableAt(0,vm);
	State->SetStateVariableAt(1,gexc);
	State->SetStateVariableAt(2,ginh);
	State->SetLastUpdateTime(CurrentTime);

	return spike;
}
