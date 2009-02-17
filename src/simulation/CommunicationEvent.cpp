#include "./include/CommunicationEvent.h"

#include "./include/Simulation.h"
#include "./include/EventQueue.h"

CommunicationEvent::CommunicationEvent():Event(0){
}
   	
CommunicationEvent::CommunicationEvent(double NewTime): Event(NewTime){
}
   		
CommunicationEvent::~CommunicationEvent(){
}

void CommunicationEvent::ProcessEvent(Simulation * CurrentSimulation){
	
	// Send the outputs
	CurrentSimulation->SendOutput();
	
	// Get the inputs
	CurrentSimulation->GetInput();
	
	if (CurrentSimulation->GetSimulationStep()>0.0){
		CommunicationEvent * NewEvent = new CommunicationEvent(this->GetTime()+CurrentSimulation->GetSimulationStep());
		CurrentSimulation->GetQueue()->InsertEvent(NewEvent);
	}		
}
   	

