/***************************************************************************
 *                           Simulation.cpp                                *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/simulation/Simulation.h"

#include "../../include/spike/Network.h"
#include "../../include/spike/Spike.h"
#include "../../include/spike/InputSpike.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/ParamReader.h"
#include "../../include/simulation/EventQueue.h"
#include "../../include/simulation/EndSimulationEvent.h"
#include "../../include/simulation/StopSimulationEvent.h"
#include "../../include/simulation/SaveWeightsEvent.h"
#include "../../include/simulation/CommunicationEvent.h"

#include "../../include/communication/OutputSpikeDriver.h"
#include "../../include/communication/InputSpikeDriver.h"
#include "../../include/communication/OutputWeightDriver.h"

#include "../include/simulation/ParameterException.h"

#include "../include/communication/ConnectionException.h"

#include "../include/spike/EDLUTFileException.h"
#include "../include/spike/EDLUTException.h"


		
Simulation::Simulation(const char * NetworkFile, const char * WeightsFile, double SimulationTime, double NewSimulationStep, double NewSaveWeightStep) throw (EDLUTException): Net(0), Queue(0), InputSpike(), OutputSpike(), OutputWeight(), Totsimtime(SimulationTime), SimulationStep(NewSimulationStep), SaveWeightStep(NewSaveWeightStep), CurrentSimulationTime(0), EndOfSimulation(false), Updates(0), Heapoc(0){
	Queue = new EventQueue();
	Net = new Network(NetworkFile, WeightsFile, this->Queue);

	// Add a final simulation event
	this->Queue->InsertEvent(new EndSimulationEvent(Totsimtime));

	// Add the first communication event
	if (this->SimulationStep>0.0F){
		this->Queue->InsertEvent(new CommunicationEvent(this->SimulationStep));		
	}

	// Add the first save weight event
	if (this->SaveWeightStep>0.0F){
		this->Queue->InsertEvent(new SaveWeightsEvent(this->SaveWeightStep));	
	}
	
}

Simulation::Simulation(const Simulation & ant):Net(ant.Net), Queue(ant.Queue), InputSpike(ant.InputSpike), OutputSpike(ant.OutputSpike), OutputWeight(ant.OutputWeight), Totsimtime(ant.Totsimtime), SaveWeightStep(ant.SaveWeightStep), EndOfSimulation(ant.EndOfSimulation), Updates(ant.Updates), Heapoc(ant.Heapoc){
}

Simulation::~Simulation(){
}

void Simulation::EndSimulation(){
	this->EndOfSimulation = true;
}

void Simulation::StopSimulation(){
	this->StopOfSimulation = true;
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

double Simulation::GetFinalSimulationTime(){
	return this->Totsimtime;	
}
		
void Simulation::RunSimulation()  throw (EDLUTException){
       while(!this->EndOfSimulation){
		Event * NewEvent;
		
		NewEvent=this->Queue->RemoveEvent();
			
		if(NewEvent->GetTime() == -1){
			break;
		}
		
		Updates++;
		Heapoc+=Queue->Size();
			
		if(NewEvent->GetTime() - this->CurrentSimulationTime < -0.0001){
			cerr << 
                        "Internal error: Bad spike time. Spike: " <<
                        NewEvent->GetTime() <<
                        " Current: " <<
                        this->CurrentSimulationTime <<
                        endl;
		}
			
		this->CurrentSimulationTime=NewEvent->GetTime(); // only for checking

                NewEvent->ProcessEvent(this);
		
		delete NewEvent;
       }
}

void Simulation::RunSimulationSlot(double preempt_time)  throw (EDLUTException){
	
	
	this->StopOfSimulation = false;
	
	// Add a stop simulation event in preempt_time
	this->Queue->InsertEvent(new StopSimulationEvent(preempt_time));       
	
	while(!this->EndOfSimulation && !this->StopOfSimulation){
		Event * NewEvent;
		
		NewEvent=this->Queue->RemoveEvent();
			
		if(NewEvent->GetTime() == -1){
			break;
		}
		
		Updates++;
		Heapoc+=Queue->Size();
			
		if(NewEvent->GetTime() - this->CurrentSimulationTime < -0.0001){
			cerr << 
                        "Internal error: Bad spike time. Spike: " <<
                        NewEvent->GetTime() <<
                        " Current: " <<
                        this->CurrentSimulationTime <<
                        endl;
		}
			
		this->CurrentSimulationTime=NewEvent->GetTime(); // only for checking

                NewEvent->ProcessEvent(this);
		
		delete NewEvent;
       }
}

void Simulation::WriteSpike(const Spike * spike){
	Neuron * neuron=spike->GetSource();  // source of the spike
        // For LSAM: Increment neuron's spike counter
	neuron->SetSpikeCounter(1 + neuron->GetSpikeCounter());
        // For LSAM: Increment total spike counter
        this->SetTotalSpikeCounter(1 + this->GetTotalSpikeCounter());
    	
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

long Simulation::GetTotalSpikeCounter(){
	return this->TotalSpikeCounter;	
}

void Simulation::SetTotalSpikeCounter(int value) {
	this->TotalSpikeCounter = value;	
}

Network * Simulation::GetNetwork() const{
	return this->Net;	
}

EventQueue * Simulation::GetQueue() const{
	return this->Queue;
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

