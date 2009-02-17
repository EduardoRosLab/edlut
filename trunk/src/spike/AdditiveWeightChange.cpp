#include "./include/AdditiveWeightChange.h"

#include "./include/Interconnection.h"
#include "./include/Neuron.h"

#include <math.h>

void AdditiveWeightChange::update_activity(double time,Interconnection * Connection,bool spike){
	// CHANGED
	// VERSION USING ANALYTICALLY SOLVED EQUATIONS
	float delta_t = (time-Connection->GetLastSpikeTime());
	float tau = 0.1;
	float quot = delta_t/tau;
	float ex = exp(-quot);

	float OldE1 = Connection->GetActivityAt(0);
	float OldE = Connection->GetActivityAt(1);
	float NewE = (OldE+quot*OldE1)*ex;
	float NewE1 = OldE1*ex; 
	
	if(spike){  // if spike, we need to increase the e1 variable
		NewE1 += 1;
	}
	
	Connection->SetActivityAt(0,NewE1);
	Connection->SetActivityAt(1,NewE);
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
		    WeightChange * wchani=interi->GetWeightChange();
		    if (wchani!=0){
		    	//CHANGED
		     	update_activity(SpikeTime,interi, false);
		     	float NewWeight = interi->GetWeight()+wchani->GetA2PrePre()*interi->GetActivityAt(1);
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
