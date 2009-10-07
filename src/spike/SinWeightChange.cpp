/***************************************************************************
 *                           AdditiveWeightChange.cpp                      *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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


#include "../../include/spike/SinWeightChange.h"

#include "../../include/spike/Interconnection.h"

#include <math.h>

#define A 1./2.

const float SinWeightChange::terms[11][11]  = {{1,0,0,0,0,0,0,0,0,0,0},
	{A,-A,0,0,0,0,0,0,0,0,0},
	{3./2.*pow(A,2),-4./2.*pow(A,2),1./2.*pow(A,2),0,0,0,0,0,0,0,0},
	{10./4.*pow(A,3),-15./4.*pow(A,3),6./4.*pow(A,3),-1./4.*pow(A,3),0,0,0,0,0,0,0},
	{35./8.*pow(A,4),-56./8.*pow(A,4),28./8.*pow(A,4),-8./8.*pow(A,4),1./8.*pow(A,4),0,0,0,0,0,0},
	{126./16.*pow(A,5),-210./16.*pow(A,5),120./16.*pow(A,5),-45./16.*pow(A,5),10./16.*pow(A,5),-1./16.*pow(A,5),0,0,0,0,0},
	{231./16.*pow(A,6),-99./4.*pow(A,6),495./32.*pow(A,6),-55./8.*pow(A,6),66./32.*pow(A,6),-3./8.*pow(A,6),1./32.*pow(A,6),0,0,0,0},
	{429./16.*pow(A,7),-3003./64.*pow(A,7),1001./32.*pow(A,7),-1001./64.*pow(A,7),91./16.*pow(A,7),-91./64.*pow(A,7),7./32.*pow(A,7),-1./64.*pow(A,7),0,0,0},
	{6435./128.*pow(A,8),-715./8.*pow(A,8),1001./16.*pow(A,8),-273./8.*pow(A,8),455./32.*pow(A,8),-35./8.*pow(A,8),15./16.*pow(A,8),-1./8.*pow(A,8),1./128.*pow(A,8),0,0},
	{12155./128.*pow(A,9),-21879./128.*pow(A,9),1989./16.*pow(A,9),-4641./64.*pow(A,9),1071./32.*pow(A,9),-765./64.*pow(A,9),51./16.*pow(A,9),-153./256.*pow(A,9),9./128.*pow(A,9),-1./256.*pow(A,9),0},
	{46189./256.*pow(A,10),-20995./64.*pow(A,10),62985./256.*pow(A,10),-4845./32.*pow(A,10),4845./64.*pow(A,10),-969./32.*pow(A,10),4845./512.*pow(A,10),-285./128.*pow(A,10),95./256.*pow(A,10),-5./128.*pow(A,10),1./512.*pow(A,10)}};
	
SinWeightChange::SinWeightChange(int NewExponent):exponent(NewExponent){
}

int SinWeightChange::GetNumberOfVar() const{
	return this->exponent+2;
}

int SinWeightChange::GetExponent() const{
	return this->exponent;
}

void SinWeightChange::update_activity(double time,Interconnection * Connection,bool spike){
	// CHANGED
	// VERSION USING ANALYTICALLY SOLVED EQUATIONS
	float delta_t = (time-Connection->GetLastSpikeTime());
	float tau = this->GetMaxPos()/atan(exponent);
	float factor = 1./(exp(-atan(this->exponent))*pow(sin(atan(this->exponent)),this->exponent));
	
	if (tau==0){
		tau = 1e-6;	
	}
	
	float expon = exp(-delta_t/tau);
	
	// Update the activity value
	float OldExpon = Connection->GetActivityAt(1);
		
	float NewActivity = factor*OldExpon*this->terms[this->exponent/2][0]*expon;
	
	float NewExpon = OldExpon * expon;
	
	if(spike){  // if spike, we need to increase the e1 variable
		NewExpon += 1;
	}
	Connection->SetActivityAt(1,NewExpon); 
	
	for (int grade=2; grade<=this->exponent; grade+=2){
		float OldVarCos = Connection->GetActivityAt(grade);
		float OldVarSin = Connection->GetActivityAt(grade+1);
		
		NewActivity += factor*(expon*this->terms[this->exponent/2][grade/2]*(OldVarCos*cos(grade*delta_t/tau)-OldVarSin*sin(grade*delta_t/tau)));
		
		float NewVarCos = (OldVarCos*cos(grade*delta_t/tau)-OldVarSin*sin(grade*delta_t/tau))*expon;
		float NewVarSin = (OldVarSin*cos(grade*delta_t/tau)+OldVarCos*sin(grade*delta_t/tau))*expon;
		
		if(spike){  // if spike, we need to increase the e1 variable
			NewVarCos += 1;
		}
		
		Connection->SetActivityAt(grade,NewVarCos);
		Connection->SetActivityAt(grade+1,NewVarSin);			
	}
	
	Connection->SetActivityAt(0,NewActivity);
}
