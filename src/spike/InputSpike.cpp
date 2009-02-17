#include ".\include\InputSpike.h"

#include "./include/Neuron.h"
#include "../simulation/include/Simulation.h"

#include "../communication/include/OutputSpikeDriver.h"

InputSpike::InputSpike():Spike() {
}
   	
InputSpike::InputSpike(double NewTime, Neuron * NewSource): Spike(NewTime,NewSource){
}
   		
InputSpike::~InputSpike(){
}

void InputSpike::ProcessEvent(Simulation * CurrentSimulation){
	
	Neuron * neuron=this->source;  // source of the spike
    
    CurrentSimulation->WriteSpike(this);
	
	CurrentSimulation->WritePotential(neuron->GetLastUpdate(), this->GetSource(), neuron->GetStateVarAt(1));
		
    //spike.time+=Net.inters[neuron->outconind].delay;
	neuron->GenerateOutputActivity((Spike *) this,CurrentSimulation->GetQueue());
}

   	
