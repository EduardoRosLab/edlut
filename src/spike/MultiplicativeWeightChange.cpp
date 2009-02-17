#include "./include/MultiplicativeWeightChange.h"

#include "./include/Interconnection.h"
#include "./include/Neuron.h"

#include <math.h>

void MultiplicativeWeightChange::ApplyWeightChange(Interconnection * Connection, double SpikeTime){
	int indexp;
	float activ;
	WeightChange *wchani;
	
	for(indexp=0;indexp<this->GetNumExps();indexp++){
		Connection->SetActivityAt(indexp,1+Connection->GetActivityAt(indexp)*exp((Connection->GetLastSpikeTime()-SpikeTime)*this->GetLparAt(indexp)));
	}
	
	Connection->SetLastSpikeTime(SpikeTime);
	Connection->SetWeight(Connection->GetWeight() + this->GetA1Pre()*((this->GetA1Pre() > 0.0)?(Connection->GetMaxWeight()-Connection->GetWeight()):Connection->GetWeight()));
	
	if(this->GetTrigger() == 1){
		Interconnection *interi;
		for(int i=0; i<Connection->GetTarget()->GetInputNumber(); ++i){
			interi=Connection->GetTarget()->GetInputConnectionAt(i);
			wchani=interi->GetWeightChange();
			if (wchani!=0){
				activ=0;
				for(indexp=0;indexp<wchani->GetNumExps();indexp++){
					activ+=wchani->GetCparAt(indexp)*interi->GetActivityAt(indexp)*exp((interi->GetLastSpikeTime()-SpikeTime)*wchani->GetLparAt(indexp));
				}
				interi->SetWeight(interi->GetWeight() + wchani->GetA2PrePre()*activ*((wchani->GetA2PrePre() > 0.0)?(interi->GetMaxWeight()-interi->GetWeight()):interi->GetWeight()));
			}
        }
	}
}
