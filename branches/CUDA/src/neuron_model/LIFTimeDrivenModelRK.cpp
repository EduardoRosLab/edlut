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
//
//	int Size=State->GetSizeState();
//
//	int i;
//	double last_update,elapsed_time,last_spike;
//	float vm,gexc,ginh;
//	bool spike;
//	float nextgexc,nextginh,k1,gexcaux,ginhaux,yaux,k2,k3,k4;
//
//#pragma omp parallel for default(none) shared(Size, State, internalSpike, CurrentTime) private(i,last_update,elapsed_time,last_spike,vm,gexc,ginh,spike,nextgexc,nextginh,k1,gexcaux,ginhaux,yaux,k2,k3,k4)
//	for (int i=0; i<Size ; i++){
//
//		last_update = State->GetLastUpdateTime(i);
//		elapsed_time = CurrentTime - last_update;
//	
//		State->AddElapsedTime(i, elapsed_time);
//	
//		last_spike = State->GetLastSpikeTime(i);
//
//		vm = State->GetStateVariableAt(i,0);
//		gexc = State->GetStateVariableAt(i,1);
//		ginh = State->GetStateVariableAt(i,2);
//
//		spike = false;
//
//		nextgexc = gexc * exp(-(elapsed_time/this->texc));
//		nextginh = ginh * exp(-(elapsed_time/this->tinh));
//
//		if (last_spike > this->tref) {
//			// 4th order Runge-Kutta terms
//			// 1st term
//			k1 = (gexc * (this->eexc - vm) + ginh * (this->einh - vm) + grest * (this->erest-vm))/this->cm;
//
//			// 2nd term
//			gexcaux = gexc * exp(-((elapsed_time/2)/this->texc));
//			ginhaux = ginh * exp(-((elapsed_time/2)/this->tinh));
//			yaux = vm+(k1*elapsed_time/2);
//			k2 = (gexcaux * (this->eexc - yaux) + ginhaux * (this->einh - yaux) + grest * (this->erest - yaux))/this->cm;
//
//			// 3rd term
//			yaux = vm+(k2*elapsed_time/2);
//			k3 = (gexcaux * (this->eexc - yaux) + ginhaux * (this->einh - yaux) + grest * (this->erest - yaux))/this->cm;
//
//			// 4rd term
//			gexcaux = nextgexc;
//			ginhaux = nextginh;
//			yaux = vm+(k3*elapsed_time);
//			k4 = (gexcaux * (this->eexc - yaux) + ginhaux * (this->einh - yaux) + grest * (this->erest - yaux))/this->cm;
//
//			vm = vm + (k1+2*k2+2*k3+k4)*elapsed_time/6;
//
//			if (vm > this->vthr){
//				State->NewFiredSpike(i);
//				spike = true;
//				vm = this->erest;
//			}
//		}
//		internalSpike[i]=spike;
//
//		gexc = nextgexc;
//		ginh = nextginh;
//
//		State->SetStateVariableAt(i,0,vm);
//		State->SetStateVariableAt(i,1,gexc);
//		State->SetStateVariableAt(i,2,ginh);
//		State->SetLastUpdateTime(i,CurrentTime);
//	}
//	return false;
//}




bool LIFTimeDrivenModelRK::UpdateState(int index, VectorNeuronState * State, double CurrentTime){
	float inv_cm=1/this->cm;
	
	double last_update = State->GetLastUpdateTime(0);
	float elapsed_time = CurrentTime - last_update;

	float exponential1=exp(-(elapsed_time/this->texc));
	float exponential2=exp(-(elapsed_time/this->tinh));
	float exponential3=exp(-((elapsed_time/2)/this->texc));
	float exponential4=exp(-((elapsed_time/2)/this->tinh));

	bool * internalSpike=State->getInternalSpike();


	int Size=State->GetSizeState();

	int i;
	double last_spike;
	float vm,gexc,ginh;
	bool spike;
	float nextgexc,nextginh,k1,gexcaux,ginhaux,yaux,k2,k3,k4;

	if(Size>1000){
#pragma omp parallel for default(none) shared(Size,last_update, elapsed_time, State, internalSpike, CurrentTime,exponential1,exponential2,exponential3,exponential4,inv_cm) private(i,last_spike,vm,gexc,ginh,spike,nextgexc,nextginh,k1,gexcaux,ginhaux,yaux,k2,k3,k4)
	for (i=0; i<Size ; i++){
	
		State->AddElapsedTime(i, elapsed_time);
	
		last_spike = State->GetLastSpikeTime(i);

		vm = State->GetStateVariableAt(i,0);
		gexc = State->GetStateVariableAt(i,1);
		ginh = State->GetStateVariableAt(i,2);

		spike = false;

//SI EXPONENTIAL1 O EXPONENTIAL2 VALE (0.5, 1] -> CurrentTime<=0.000346, EL TIEMPO NECESARIO PARA LA
//MULTIPLICACIÓN SE DISPARA.
		nextgexc = gexc * exponential1;
		nextginh = ginh * exponential2;


		if (last_spike > this->tref) {
			// 4th order Runge-Kutta terms
			// 1st term
			k1 = (gexc * (this->eexc - vm) + ginh * (this->einh - vm) + grest * (this->erest-vm))*inv_cm;
		
			// 2nd term
			gexcaux = gexc * exponential3;
			ginhaux = ginh * exponential4;
			yaux = vm+(k1*elapsed_time/2);
			k2 = (gexcaux * (this->eexc - yaux) + ginhaux * (this->einh - yaux) + grest * (this->erest - yaux))*inv_cm;

			// 3rd term
			yaux = vm+(k2*elapsed_time/2);
			k3 = (gexcaux * (this->eexc - yaux) + ginhaux * (this->einh - yaux) + grest * (this->erest - yaux))*inv_cm;

			// 4rd term
			gexcaux = nextgexc;
			ginhaux = nextginh;
			yaux = vm+(k3*elapsed_time);
			k4 = (gexcaux * (this->eexc - yaux) + ginhaux * (this->einh - yaux) + grest * (this->erest - yaux))*inv_cm;

			vm += (k1+2*(k2+k3)+k4)*elapsed_time/6;

			if (vm > this->vthr){
				State->NewFiredSpike(i);
				spike = true;
				vm = this->erest;
			}
		}
		internalSpike[i]=spike;

		State->SetStateVariableAt(i,0,vm);
		State->SetStateVariableAt(i,1,nextgexc);
		State->SetStateVariableAt(i,2,nextginh);

		State->SetLastUpdateTime(i,CurrentTime);
	}
	}

	else{
	for (i=0; i<Size ; i++){
	
		State->AddElapsedTime(i, elapsed_time);
	
		last_spike = State->GetLastSpikeTime(i);

		vm = State->GetStateVariableAt(i,0);
		gexc = State->GetStateVariableAt(i,1);
		ginh = State->GetStateVariableAt(i,2);

		spike = false;

//SI EXPONENTIAL1 O EXPONENTIAL2 VALE (0.5, 1] -> CurrentTime<=0.000346, EL TIEMPO NECESARIO PARA LA
//MULTIPLICACIÓN SE DISPARA.
		nextgexc = gexc * exponential1;
		nextginh = ginh * exponential2;


		if (last_spike > this->tref) {
			// 4th order Runge-Kutta terms
			// 1st term
			k1 = (gexc * (this->eexc - vm) + ginh * (this->einh - vm) + grest * (this->erest-vm))*inv_cm;
		
			// 2nd term
			gexcaux = gexc * exponential3;
			ginhaux = ginh * exponential4;
			yaux = vm+(k1*elapsed_time/2);
			k2 = (gexcaux * (this->eexc - yaux) + ginhaux * (this->einh - yaux) + grest * (this->erest - yaux))*inv_cm;

			// 3rd term
			yaux = vm+(k2*elapsed_time/2);
			k3 = (gexcaux * (this->eexc - yaux) + ginhaux * (this->einh - yaux) + grest * (this->erest - yaux))*inv_cm;

			// 4rd term
			gexcaux = nextgexc;
			ginhaux = nextginh;
			yaux = vm+(k3*elapsed_time);
			k4 = (gexcaux * (this->eexc - yaux) + ginhaux * (this->einh - yaux) + grest * (this->erest - yaux))*inv_cm;

			vm += (k1+2*(k2+k3)+k4)*elapsed_time/6;

			if (vm > this->vthr){
				State->NewFiredSpike(i);
				spike = true;
				vm = this->erest;
			}
		}
		internalSpike[i]=spike;

		State->SetStateVariableAt(i,0,vm);
		State->SetStateVariableAt(i,1,nextgexc);
		State->SetStateVariableAt(i,2,nextginh);

		State->SetLastUpdateTime(i,CurrentTime);
	}
	}

	return false;
}