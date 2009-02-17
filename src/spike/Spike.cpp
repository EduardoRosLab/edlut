#include "./include/Spike.h"

#include "./include/Neuron.h"

Spike::Spike():Event(0), source(0){
}
   	
Spike::Spike(double NewTime, Neuron * NewSource): Event(NewTime), source(NewSource){
}
   		
Spike::~Spike(){
}
   	
Neuron * Spike::GetSource () const{
	return source;
}
   		
void Spike::SetSource (Neuron * NewSource){
	source = NewSource;
}
