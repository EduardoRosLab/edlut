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


#include "../../include/spike/AdditiveWeightChange.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include <math.h>

int AdditiveWeightChange::GetNumberOfVar() const{
	return 2;
}

void AdditiveWeightChange::update_activity(double time,Interconnection * Connection,bool spike){
	// CHANGED
	// VERSION USING ANALYTICALLY SOLVED EQUATIONS
	float delta_t = (time-Connection->GetLastSpikeTime());
	float tau = this->GetMaxPos();
	if (tau==0){
		tau = 1e-6;	
	}
	float quot = delta_t/tau;
	float ex = exp(-quot);

	float OldE1 = Connection->GetActivityAt(1);
	float OldE = Connection->GetActivityAt(0);
	float NewE = (OldE+quot*OldE1)*ex;
	float NewE1 = OldE1*ex; 
	
	if(spike){  // if spike, we need to increase the e1 variable
		NewE1 += 1;
	}
	
	Connection->SetActivityAt(1,NewE1);
	Connection->SetActivityAt(0,NewE);
}

void AdditiveWeightChange::ApplyWeightChange(Interconnection * Connection, double SpikeTime){
	
	// Second case: the weight change is linked to this connection
	float NewWeight = Connection->GetWeight()+this->GetA1Pre();

	// CHANGED
	update_activity(SpikeTime,Connection, true);
	
	if(NewWeight>Connection->GetMaxWeight())
		NewWeight=Connection->GetMaxWeight();
	else if(NewWeight<0.0)
		NewWeight=0.0;
	
	Connection->SetLastSpikeTime(SpikeTime);	
	Connection->SetWeight(NewWeight);
	
	if(this->GetTrigger() == 1){
		for(int i=0; i<Connection->GetTarget()->GetInputNumber(); ++i){
			Interconnection * interi=Connection->GetTarget()->GetInputConnectionAt(i);
		    AdditiveWeightChange * wchani=(AdditiveWeightChange *)interi->GetWeightChange();
		    if (wchani!=0){
		    	//CHANGED
		     	wchani->update_activity(SpikeTime,interi, false);
		     	float NewWeight = interi->GetWeight()+wchani->GetA2PrePre()*interi->GetActivityAt(0);
		     	//
		     	if(NewWeight>interi->GetMaxWeight())
		       		NewWeight=interi->GetMaxWeight();
		     	else if(NewWeight<0.0)
		       		NewWeight=0.0;
		       		
		       	interi->SetWeight(NewWeight);
		    }
		}
	}
}
