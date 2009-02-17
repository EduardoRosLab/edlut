#include "./include/Event.h"

#include "./include/Simulation.h"

Event::Event():time(0){
}
   	
Event::Event(double NewTime): time(NewTime){
}
   		
Event::~Event(){
}
   	
double Event::GetTime() const{
	return time;
}
   		
void Event::SetTime (double NewTime){
	time = NewTime;
}


