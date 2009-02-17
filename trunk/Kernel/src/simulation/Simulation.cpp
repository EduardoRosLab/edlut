#include "./include/Simulation.h"

#include "../spike/include/Network.h"
#include "../spike/include/Spike.h"
#include "../spike/include/Neuron.h"

#include "./include/EventQueue.h"
#include "./include/EndSimulationEvent.h"
#include "./include/SaveWeightsEvent.h"
#include "./include/CommunicationEvent.h"

#include "../communication/include/OutputSpikeDriver.h"
#include "../communication/include/InputSpikeDriver.h"
#include "../communication/include/OutputWeightDriver.h"

Simulation::Simulation(const char * NetworkFile, const char * WeightsFile, double SimulationTime, double NewSimulationStep) throw (EDLUTException): Net(0), Queue(0), InputSpike(), OutputSpike(), OutputWeight(), Totsimtime(SimulationTime), SimulationStep(NewSimulationStep), SaveWeightStep(0), EndOfSimulation(false), Updates(0), Heapoc(0){
	Queue = new EventQueue();
	Net = new Network(NetworkFile, WeightsFile, this->Queue);
}

Simulation::Simulation(const Simulation & ant):Net(ant.Net), Queue(ant.Queue), InputSpike(ant.InputSpike), OutputSpike(ant.OutputSpike), OutputWeight(ant.OutputWeight), Totsimtime(ant.Totsimtime), SaveWeightStep(ant.SaveWeightStep), EndOfSimulation(ant.EndOfSimulation), Updates(ant.Updates), Heapoc(ant.Heapoc){
}

Simulation::~Simulation(){
}

void Simulation::EndSimulation(){
	this->EndOfSimulation = true;
}

void Simulation::SetSaveStep(float NewSaveStep){
	this->SaveWeightStep = NewSaveStep;	
}

double Simulation::GetSaveStep(){
	return this->SaveWeightStep;	
}

void Simulation::SetSimulationStep(double NewSimulationStep){
	this->SimulationStep = NewSimulationStep;		
}
		
double Simulation::GetSimulationStep(){
	return this->SimulationStep;	
}

void Simulation::RunSimulation()  throw (EDLUTException){
	this->CurrentSimulationTime = 0.0;
	
	// Get the external initial inputs
	this->GetInput();
	
	// Add a final simulation event
	this->Queue->InsertEvent(new EndSimulationEvent(this->Totsimtime));
	
	// Add the first save weight event
	if (this->SaveWeightStep>0.0F){
		this->Queue->InsertEvent(new SaveWeightsEvent(this->SaveWeightStep));	
	}
	
	// Add the first communication event
	if (this->SimulationStep>0.0F){
		this->Queue->InsertEvent(new CommunicationEvent(this->SimulationStep));		
	}	
	
	while(!this->EndOfSimulation){
		Event * NewEvent;
		
		NewEvent=this->Queue->RemoveEvent();
			
		if(NewEvent->GetTime() == -1){
			break;
		}
		
		Updates++;
		Heapoc+=Queue->Size();
			
		if(NewEvent->GetTime() - this->CurrentSimulationTime < -0.0001){
			cerr << "Internal error: Bad spike time. Spike: " << NewEvent->GetTime() << " Current: " << this->CurrentSimulationTime << endl;
		}
			
		this->CurrentSimulationTime=NewEvent->GetTime(); // only for checking

		NewEvent->ProcessEvent(this);
		
		delete NewEvent;
	}
}

void Simulation::WriteSpike(const Spike * spike){
	Neuron * neuron=spike->GetSource();  // source of the spike
    
    if(neuron->IsMonitored()){
		for (list<OutputSpikeDriver *>::iterator it=this->MonitorSpike.begin(); it!=this->MonitorSpike.end(); ++it){
			(*it)->WriteSpike(spike);
		}
	}
	   	
	if(neuron->IsOutput()){
		for (list<OutputSpikeDriver *>::iterator it=this->OutputSpike.begin(); it!=this->OutputSpike.end(); ++it){
			(*it)->WriteSpike(spike);
		}
	}		
}

void Simulation::WritePotential(float time, Neuron * neuron, float value){
	if(neuron->IsMonitored()){
		for (list<OutputSpikeDriver *>::iterator it=this->MonitorSpike.begin(); it!=this->MonitorSpike.end(); ++it){
			if ((*it)->IsWritePotentialCapable()){
				(*it)->WritePotential(time, neuron, value);
			}
		}
	}
	   	
	if(neuron->IsOutput()){
		for (list<OutputSpikeDriver *>::iterator it=this->OutputSpike.begin(); it!=this->OutputSpike.end(); ++it){
			if ((*it)->IsWritePotentialCapable()){
				(*it)->WritePotential(time, neuron, value);
			}
		}
	}		
}

void Simulation::SaveWeights(){
	cout << "Saving weights in time " << this->CurrentSimulationTime << endl;
	for (list<OutputWeightDriver *>::iterator it=this->OutputWeight.begin(); it!=this->OutputWeight.end(); ++it){
		(*it)->WriteWeights(this->Net,this->CurrentSimulationTime);
	}
	
}

void Simulation::SendOutput(){
	cout << "Sending outputs in time " << this->CurrentSimulationTime << endl;
	for (list<OutputSpikeDriver *>::iterator it=this->OutputSpike.begin(); it!=this->OutputSpike.end(); ++it){
		if ((*it)->IsBuffered()){
			(*it)->FlushBuffers();	
		}
	}
}

void Simulation::GetInput(){
	cout << "Getting inputs in time " << this->CurrentSimulationTime << endl;
	for (list<InputSpikeDriver *>::iterator it=this->InputSpike.begin(); it!=this->InputSpike.end(); ++it){
		if (!(*it)->IsFinished()){
			(*it)->LoadInputs(this->Queue, this->Net);
		}
	}
}

Network * Simulation::GetNetwork() const{
	return this->Net;	
}

EventQueue * Simulation::GetQueue() const{
	return this->Queue;
}
		
double Simulation::GetTotalSimulationTime() const{
	return this->Totsimtime;	
}
		
long long Simulation::GetSimulationUpdates() const{
	return this->Updates;	
}
		
long long Simulation::GetHeapAcumSize() const{
	return this->Heapoc;
}		

void Simulation::AddInputSpikeDriver(InputSpikeDriver * NewInput){
	this->InputSpike.push_back(NewInput);	
}
		 
void Simulation::RemoveInputSpikeDriver(InputSpikeDriver * NewInput){
	this->InputSpike.remove(NewInput);	
}
		 
void Simulation::AddOutputSpikeDriver(OutputSpikeDriver * NewOutput){
	this->OutputSpike.push_back(NewOutput);	
}
		
void Simulation::RemoveOutputSpikeDriver(OutputSpikeDriver * NewOutput){
	this->OutputSpike.remove(NewOutput);
}

void Simulation::AddMonitorActivityDriver(OutputSpikeDriver * NewMonitor){
	this->MonitorSpike.push_back(NewMonitor);	
}
		
void Simulation::RemoveMonitorActivityDriver(OutputSpikeDriver * NewMonitor){
	this->MonitorSpike.remove(NewMonitor);
}

void Simulation::AddOutputWeightDriver(OutputWeightDriver * NewOutput){
	this->OutputWeight.push_back(NewOutput);	
}
		
void Simulation::RemoveOutputWeightDriver(OutputWeightDriver * NewOutput){
	this->OutputWeight.remove(NewOutput);
}
