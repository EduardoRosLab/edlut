#include "./include/InputSpikeDriver.h"

InputSpikeDriver::~InputSpikeDriver(){
	
}

 bool InputSpikeDriver::IsFinished() const{
 	return this->Finished;
 }
