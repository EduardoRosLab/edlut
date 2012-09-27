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
#include "../../include/neuron_model/VectorNeuronState.h"

#include <iostream>
#include <cmath>
#include <string>
#ifdef _OPENMP
	#include <omp.h>
#endif

LIFTimeDrivenModelRK::LIFTimeDrivenModelRK(string NeuronTypeID, string NeuronModelID): LIFTimeDrivenModel(NeuronTypeID, NeuronModelID) {
}

LIFTimeDrivenModelRK::~LIFTimeDrivenModelRK(){

}


//bool LIFTimeDrivenModelRK::UpdateState(int index, VectorNeuronState * State, double CurrentTime){
//
//	bool * internalSpike=State->getInternalSpike();
//	int Size=State->GetSizeState();
//
//	for (int i=0; i< Size; i++){
//		float last_update = State->GetLastUpdateTime(i);
//		
//		float elapsed_time = CurrentTime - last_update;
//
//		State->AddElapsedTime(i,elapsed_time);
//		
//		float last_spike = State->GetLastSpikeTime(i);
//
//		float vm = State->GetStateVariableAt(i,0);
//		float gampa = State->GetStateVariableAt(i,1);
//		float gnmda = State->GetStateVariableAt(i,2);
//		float ginh = State->GetStateVariableAt(i,3);
//		float ggj = State->GetStateVariableAt(i,4);
//
//		float nextgampa = gampa * exp(-(elapsed_time/this->tampa));
//		float nextgnmda = gnmda * exp(-(elapsed_time/this->tnmda));
//		float nextginh = ginh * exp(-(elapsed_time/this->tinh));
//		float nextggj = ggj * exp(-(elapsed_time/this->tgj));
//		
//
//		bool spike = false;
//
//		if (last_spike > this->tref) {
//			// 4th order Runge-Kutta terms
//			// 1st term
//			float iampa = gampa*(this->eexc-vm);
//			float gnmdainf = 1.0/(1.0 + exp(-62.0*vm)*1.2/3.57);
//			float inmda = gnmda*gnmdainf*(this->eexc-vm);
//			float iinh = ginh*(this->einh-vm);
//			
//			float k1 = (iampa + inmda + iinh + grest * (this->erest-vm))*1.e-9/this->cm;
//
//			// 2nd term
//			float gampaaux = gampa * exp(-((elapsed_time/2)/this->tampa));
//			float gnmdaaux = gnmda * exp(-((elapsed_time/2)/this->tnmda));
//			float ginhaux = ginh * exp(-((elapsed_time/2)/this->tinh));
//			float yaux = vm+(k1*elapsed_time/2);
//			
//			float iampaaux = gampaaux*(this->eexc-yaux);
//			float gnmdainfaux = 1.0/(1.0 + exp(-62.0*yaux)*1.2/3.57);
//			float inmdaaux = gnmdaaux*gnmdainfaux*(this->eexc-yaux);
//			float iinhaux = ginhaux*(this->einh-yaux);
//					
//			float k2 = (iampaaux + inmdaaux + iinhaux + grest * (this->erest - yaux))*1.e-9/this->cm;
//
//			// 3rd term
//			yaux = vm+(k2*elapsed_time/2);
//
//			iampaaux = gampaaux*(this->eexc-yaux);
//			gnmdainfaux = 1.0/(1.0 + exp(-62.0*yaux)*1.2/3.57);
//			inmdaaux = gnmdaaux*gnmdainfaux*(this->eexc-yaux);
//			iinhaux = ginhaux*(this->einh-yaux);
//			
//			float k3 = (iampaaux + inmdaaux + iinhaux + grest * (this->erest - yaux))*1.e-9/this->cm;
//
//			// 4rd term
//			yaux = vm+(k3*elapsed_time);
//
//			iampaaux = nextgampa*(this->eexc-yaux);
//			gnmdainfaux = 1.0/(1.0 + exp(-62.0*yaux)*1.2/3.57);
//			inmdaaux = nextgampa*gnmdainfaux*(this->eexc-yaux);
//			iinhaux = nextginh*(this->einh-yaux);
//			
//			float k4 = (iampaaux + inmdaaux + iinhaux + grest * (this->erest - yaux))*1.e-9/this->cm;
//
//			vm = vm + (k1+2*k2+2*k3+k4)*elapsed_time/6;
//
//			float vm_cou = vm + this->fgj * ggj;
//
//			if (vm_cou > this->vthr){
//				State->NewFiredSpike(i);
//				spike = true;
//				vm = this->erest;
//			}
//		}
//		
//		internalSpike[i]=spike;
//
//		gampa = nextgampa;
//		gnmda = nextgnmda;
//		ginh = nextginh;
//		ggj = nextggj;
//
//		State->SetStateVariableAt(i,0,vm);
//		State->SetStateVariableAt(i,1,gampa);
//		State->SetStateVariableAt(i,2,gnmda);
//		State->SetStateVariableAt(i,3,ginh);
//		State->SetStateVariableAt(i,4,ggj);
//		State->SetLastUpdateTime(i,CurrentTime);
//	}
//	return false;
//}


bool LIFTimeDrivenModelRK::UpdateState(int index, VectorNeuronState * State, double CurrentTime){

	float inv_cm=1.e-9/this->cm;
	
	bool * internalSpike=State->getInternalSpike();
	int Size=State->GetSizeState();

	float last_update = State->GetLastUpdateTime(0);
	
	float elapsed_time = CurrentTime - last_update;

	float last_spike;

	float exp_gampa = exp(-(elapsed_time/this->tampa));
	float exp_gnmda = exp(-(elapsed_time/this->tnmda));
	float exp_ginh = exp(-(elapsed_time/this->tinh));
	float exp_ggj = exp(-(elapsed_time/this->tgj));

	float exp_gampa2 = exp(-((elapsed_time/2)/this->tampa));
	float exp_gnmda2 = exp(-((elapsed_time/2)/this->tnmda));
	float exp_ginh2 = exp(-((elapsed_time/2)/this->tinh));

	float vm, gampa, gnmda, ginh, ggj;

	float nextgampa, nextgnmda, nextginh, nextggj;

	bool spike;

	float k1, k2, k3, k4;

	float iampa, gnmdainf, inmda, iinh;

	float gampaaux, gnmdaaux, ginhaux, yaux, iampaaux, gnmdainfaux, inmdaaux, iinhaux;

	float vm_cou;

	int i;

	#pragma omp parallel for default(none) shared(Size, State, internalSpike, CurrentTime, elapsed_time, exp_gampa, exp_gnmda, exp_ginh, exp_ggj, inv_cm, exp_gampa2, exp_gnmda2, exp_ginh2) private(i,last_spike,vm, gampa, gnmda, ginh, ggj,nextgampa, nextgnmda, nextginh, nextggj, spike, k1, k2, k3, k4, iampa, gnmdainf, inmda, iinh, gampaaux, gnmdaaux, ginhaux, yaux, iampaaux, gnmdainfaux, inmdaaux, iinhaux, vm_cou)
	for (i=0; i< Size; i++){

		State->AddElapsedTime(i,elapsed_time);
		
		last_spike = State->GetLastSpikeTime(i);

		vm = State->GetStateVariableAt(i,0);
		gampa = State->GetStateVariableAt(i,1);
		gnmda = State->GetStateVariableAt(i,2);
		ginh = State->GetStateVariableAt(i,3);
		ggj = State->GetStateVariableAt(i,4);

		nextgampa = gampa * exp_gampa;
		nextgnmda = gnmda * exp_gnmda;
		nextginh = ginh * exp_ginh;
		nextggj = ggj * exp_ggj;
		

		spike = false;

		if (last_spike > this->tref) {
			// 4th order Runge-Kutta terms
			// 1st term
			iampa = gampa*(this->eexc-vm);
			//gnmdainf = 1.0/(1.0 + exp(-62.0*vm)*1.2/3.57);
			gnmdainf = 1.0/(1.0 + exp(-62.0*vm)*0.336134453);
			inmda = gnmda*gnmdainf*(this->eexc-vm);
			iinh = ginh*(this->einh-vm);
			
			k1 = (iampa + inmda + iinh + grest * (this->erest-vm))*inv_cm;

			// 2nd term
			gampaaux = gampa * exp_gampa2;
			gnmdaaux = gnmda * exp_gnmda2;
			ginhaux = ginh * exp_ginh2;
			yaux = vm+(k1*elapsed_time/2);
			
			iampaaux = gampaaux*(this->eexc-yaux);
			//gnmdainfaux = 1.0/(1.0 + exp(-62.0*yaux)*1.2/3.57);
			gnmdainfaux = 1.0/(1.0 + exp(-62.0*yaux)*0.336134453);
			inmdaaux = gnmdaaux*gnmdainfaux*(this->eexc-yaux);
			iinhaux = ginhaux*(this->einh-yaux);
					
			k2 = (iampaaux + inmdaaux + iinhaux + grest * (this->erest - yaux))*inv_cm;

			// 3rd term
			yaux = vm+(k2*elapsed_time/2);

			iampaaux = gampaaux*(this->eexc-yaux);
			//gnmdainfaux = 1.0/(1.0 + exp(-62.0*yaux)*1.2/3.57);
			gnmdainfaux = 1.0/(1.0 + exp(-62.0*yaux)*0.336134453);
			inmdaaux = gnmdaaux*gnmdainfaux*(this->eexc-yaux);
			iinhaux = ginhaux*(this->einh-yaux);
			
			k3 = (iampaaux + inmdaaux + iinhaux + grest * (this->erest - yaux))*inv_cm;

			// 4rd term
			yaux = vm+(k3*elapsed_time);

			iampaaux = nextgampa*(this->eexc-yaux);
			//gnmdainfaux = 1.0/(1.0 + exp(-62.0*yaux)*1.2/3.57);
			gnmdainfaux = 1.0/(1.0 + exp(-62.0*yaux)*0.336134453);
			inmdaaux = nextgampa*gnmdainfaux*(this->eexc-yaux);
			iinhaux = nextginh*(this->einh-yaux);
			
			k4 = (iampaaux + inmdaaux + iinhaux + grest * (this->erest - yaux))*inv_cm;

			vm = vm + (k1+2*k2+2*k3+k4)*elapsed_time/6;

			vm_cou = vm + this->fgj * ggj;
			
			if (vm_cou > this->vthr){
				State->NewFiredSpike(i);
				spike = true;
				vm = this->erest;
			}
		}
		
		internalSpike[i]=spike;


		gampa = nextgampa;
		gnmda = nextgnmda;
		ginh = nextginh;
		ggj = nextggj;

		State->SetStateVariableAt(i,0,vm);
		State->SetStateVariableAt(i,1,gampa);
		State->SetStateVariableAt(i,2,gnmda);
		State->SetStateVariableAt(i,3,ginh);
		State->SetStateVariableAt(i,4,ggj);
		State->SetLastUpdateTime(i,CurrentTime);
	}
	return false;
}

