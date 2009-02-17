#include "./include/EndSimulationEvent.h"

#include "./include/Simulation.h"

EndSimulationEvent::EndSimulationEvent():Event(0){
}
   	
EndSimulationEvent::EndSimulationEvent(double NewTime): Event(NewTime){
}
   		
EndSimulationEvent::~EndSimulationEvent(){
}

void EndSimulationEvent::ProcessEvent(Simulation * CurrentSimulation){
	CurrentSimulation->EndSimulation();		
}
   	


