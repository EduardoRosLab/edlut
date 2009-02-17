#include "./include/SaveWeightsEvent.h"

#include "./include/Simulation.h"
#include "./include/EventQueue.h"

SaveWeightsEvent::SaveWeightsEvent():Event(0){
}
   	
SaveWeightsEvent::SaveWeightsEvent(double NewTime): Event(NewTime){
}
   		
SaveWeightsEvent::~SaveWeightsEvent(){
}

void SaveWeightsEvent::ProcessEvent(Simulation * CurrentSimulation){
	CurrentSimulation->SaveWeights();
	if (CurrentSimulation->GetSaveStep()>0.0){
		SaveWeightsEvent * NewEvent = new SaveWeightsEvent(this->GetTime()+CurrentSimulation->GetSaveStep());
		CurrentSimulation->GetQueue()->InsertEvent(NewEvent);
	}		
}
   	

