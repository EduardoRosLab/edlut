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
	float gampa = State->GetStateVariableAt(1);
	float gnmda = State->GetStateVariableAt(2);
	float ginh = State->GetStateVariableAt(3);
	float ggj = State->GetStateVariableAt(4);

	float nextgampa = gampa * exp(-(elapsed_time/this->tampa));
	float nextgnmda = gnmda * exp(-(elapsed_time/this->tnmda));
	float nextginh = ginh * exp(-(elapsed_time/this->tinh));
	float nextggj = ggj * exp(-(elapsed_time/this->tgj));
	

	bool spike = false;

	if (last_spike > this->tref) {
		// 4th order Runge-Kutta terms
		// 1st term
		float iampa = gampa*(this->eexc-vm);
		float gnmdainf = 1.0/(1.0 + exp(-62.0*vm)*1.2/3.57);
		float inmda = gnmda*gnmdainf*(this->eexc-vm);
		float iinh = ginh*(this->einh-vm);
		
		float k1 = (iampa + inmda + iinh + grest * (this->erest-vm))*1.e-9/this->cm;

		// 2nd term
		float gampaaux = gampa * exp(-((elapsed_time/2)/this->tampa));
		float gnmdaaux = gnmda * exp(-((elapsed_time/2)/this->tnmda));
		float ginhaux = ginh * exp(-((elapsed_time/2)/this->tinh));
		float yaux = vm+(k1*elapsed_time/2);
		
		float iampaaux = gampaaux*(this->eexc-yaux);
		float gnmdainfaux = 1.0/(1.0 + exp(-62.0*yaux)*1.2/3.57);
		float inmdaaux = gnmdaaux*gnmdainfaux*(this->eexc-yaux);
		float iinhaux = ginhaux*(this->einh-yaux);
				
		float k2 = (iampaaux + inmdaaux + iinhaux + grest * (this->erest - yaux))*1.e-9/this->cm;

		// 3rd term
		yaux = vm+(k2*elapsed_time/2);

		iampaaux = gampaaux*(this->eexc-yaux);
		gnmdainfaux = 1.0/(1.0 + exp(-62.0*yaux)*1.2/3.57);
		inmdaaux = gnmdaaux*gnmdainfaux*(this->eexc-yaux);
		iinhaux = ginhaux*(this->einh-yaux);
		
		float k3 = (iampaaux + inmdaaux + iinhaux + grest * (this->erest - yaux))*1.e-9/this->cm;

		// 4rd term
		yaux = vm+(k3*elapsed_time);

		iampaaux = nextgampa*(this->eexc-yaux);
		gnmdainfaux = 1.0/(1.0 + exp(-62.0*yaux)*1.2/3.57);
		inmdaaux = nextgampa*gnmdainfaux*(this->eexc-yaux);
		iinhaux = nextginh*(this->einh-yaux);
		
		float k4 = (iampaaux + inmdaaux + iinhaux + grest * (this->erest - yaux))*1.e-9/this->cm;

		vm = vm + (k1+2*k2+2*k3+k4)*elapsed_time/6;

		float vm_cou = vm + this->fgj * ggj;

		if (vm_cou > this->vthr){
			State->NewFiredSpike();
			spike = true;
			vm = this->erest;
		}
	}
	
	gampa = nextgampa;
	gnmda = nextgnmda;
	ginh = nextginh;
	ggj = nextggj;

	State->SetStateVariableAt(0,vm);
	State->SetStateVariableAt(1,gampa);
	State->SetStateVariableAt(2,gnmda);
	State->SetStateVariableAt(3,ginh);
	State->SetStateVariableAt(4,ggj);
	State->SetLastUpdateTime(CurrentTime);

	return spike;
}
