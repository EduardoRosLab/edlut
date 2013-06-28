/***************************************************************************
 *                           Simulation.cpp                                *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido, Richard Carrillo and  *
 *						: Francisco Naveros                                *
 * email                : jgarrido@atc.ugr.es, fnaveros@atc.ugr.es         *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/*
 * \note: this file Simulation_GPU.cpp must be used instead of file Simulation.cpp to 
 * implement a CPU-GPU hybrid architecture.
*/

#if defined(_WIN32) || defined(_WIN64)
   #include "windows.h"
#endif

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/StopSimulationEvent.h"
#include "../../include/simulation/TimeEventOneNeuron.h"
#include "../../include/simulation/TimeEventAllNeurons.h"
#include "../../include/simulation/TimeEventAllNeurons_GPU.h"

#include "../../include/spike/Network.h"
#include "../../include/spike/Spike.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/EventQueue.h"
#include "../../include/simulation/EndSimulationEvent.h"
#include "../../include/simulation/SaveWeightsEvent.h"
#include "../../include/simulation/CommunicationEvent.h"

#include "../../include/communication/OutputSpikeDriver.h"
#include "../../include/communication/InputSpikeDriver.h"
#include "../../include/communication/OutputWeightDriver.h"

#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU.h"

Simulation::Simulation(const char * NetworkFile, const char * WeightsFile, double SimulationTime, double NewSimulationStep) throw (EDLUTException): Net(0), Queue(0), InputSpike(), OutputSpike(), OutputWeight(), Totsimtime(SimulationTime), SimulationStep(NewSimulationStep), TimeDrivenStep(0), MaxSlotConsumedTime(0), TimeDrivenStepGPU(0), SaveWeightStep(0), EndOfSimulation(false), Updates(0), Heapoc(0){
	Queue = new EventQueue();
	Net = new Network(NetworkFile, WeightsFile, this->Queue);
}

Simulation::Simulation(const Simulation & ant):Net(ant.Net), Queue(ant.Queue), InputSpike(ant.InputSpike), OutputSpike(ant.OutputSpike), OutputWeight(ant.OutputWeight), Totsimtime(ant.Totsimtime), TimeDrivenStep(ant.TimeDrivenStep), MaxSlotConsumedTime(ant.MaxSlotConsumedTime), TimeDrivenStepGPU(ant.TimeDrivenStepGPU), SaveWeightStep(ant.SaveWeightStep), EndOfSimulation(ant.EndOfSimulation), Updates(ant.Updates), Heapoc(ant.Heapoc){
}

Simulation::~Simulation(){
	if (this->Net){
		delete this->Net;
		this->Net=NULL;
	}

	if (this->Queue){
		delete this->Queue;
		this->Queue=NULL;
	}
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

void Simulation::SetTimeDrivenStep(double NewTimeDrivenStep){
	this->TimeDrivenStep = NewTimeDrivenStep;		
}
		
double Simulation::GetTimeDrivenStep(){
	return this->TimeDrivenStep;	
}

void Simulation::SetTimeDrivenStepGPU(double NewTimeDrivenStepGPU){
	this->TimeDrivenStepGPU = NewTimeDrivenStepGPU;		
}
		
double Simulation::GetTimeDrivenStepGPU(){
	return this->TimeDrivenStepGPU;	
}

void Simulation::SetMaxSlotConsumedTime(double NewMaxSlotConsumedTime){
    LARGE_INTEGER freq;
    this->MaxSlotConsumedTime = 0UL;
#if defined(_WIN32) || defined(_WIN64)
    if(QueryPerformanceFrequency(&freq)){
    	this->MaxSlotConsumedTime = (unsigned long)(freq.QuadPart*NewMaxSlotConsumedTime);
    }
#endif
}
		
double Simulation::GetMaxSlotConsumedTime(){
    double max_slot_consumed_time;
    LARGE_INTEGER freq;
   	max_slot_consumed_time=0UL;
#if defined(_WIN32) || defined(_WIN64)
    if(QueryPerformanceFrequency(&freq)){
    	max_slot_consumed_time=this->MaxSlotConsumedTime/(double)freq.QuadPart;
    }
#endif
	return max_slot_consumed_time;
}

void Simulation::InitSimulation() throw (EDLUTException){
	this->CurrentSimulationTime = 0.0;

	// Get the external initial inputs
	this->GetInput();
	
	// Add a final simulation event
	if (this->Totsimtime >= 0){
		this->Queue->InsertEvent(new EndSimulationEvent(this->Totsimtime));
	}
	
	// Add the first save weight event
	if (this->SaveWeightStep>0.0F){
		this->Queue->InsertEvent(new SaveWeightsEvent(this->SaveWeightStep));	
	}
	
	// Add the first communication event
	if (this->SimulationStep>0.0F){
		this->Queue->InsertEvent(new CommunicationEvent(this->SimulationStep));		
	}	

	
	// Add the CPU time-driven simulation events
	int * N_TimeDrivenNeuron=this->GetNetwork()->GetTimeDrivenNeuronNumber();
	for(int z=0; z<this->GetNetwork()->GetNneutypes(); z++){
		if(N_TimeDrivenNeuron[z]>0){
			TimeDrivenNeuronModel * model=(TimeDrivenNeuronModel *) this->GetNetwork()->GetNeuronModelAt(z);
			//If this model implement a fixed step integration method, one TimeEvent can manage all
			//neurons in this neuron model
			if(model->integrationMethod->GetMethodType()==FIXED_STEP){
				this->Queue->InsertEvent(new TimeEventAllNeurons(model->integrationMethod->PredictedElapsedTime[0],z));
			}
			//If this model implement a variable step integration method, it is necesary to 
			//implement a TimeEvent for each neuron in this neuron model.
			else{
				for(int i=0; i<N_TimeDrivenNeuron[z]; i++){
					this->Queue->InsertEvent(new TimeEventOneNeuron(model->integrationMethod->PredictedElapsedTime[i], z, i));
				}
			}
		}
	}

	// Add the GPU time-driven simulation events
	int * N_TimeDrivenNeuron_GPU=this->GetNetwork()->GetTimeDrivenNeuronNumberGPU();
	for(int z=0; z<this->GetNetwork()->GetNneutypes(); z++){
		if(N_TimeDrivenNeuron_GPU[z]>0){
			TimeDrivenNeuronModel_GPU * model=(TimeDrivenNeuronModel_GPU *) this->GetNetwork()->GetNeuronModelAt(z);
			this->Queue->InsertEvent(new TimeEventAllNeurons_GPU(model->GetTimeDrivenStep_GPU(),z));
		}
	}

	SetTotalSpikeCounter(0);

}

void Simulation::RunSimulation()  throw (EDLUTException){
	this->CurrentSimulationTime = 0.0;

	this->InitSimulation();
	
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

		NewEvent->ProcessEvent(this, false);
		
		delete NewEvent;
	}
}

void Simulation::RunSimulationSlot(double preempt_time)  throw (EDLUTException){
    bool real_time_restriction;
#if defined(_WIN32) || defined(_WIN64)
    LARGE_INTEGER slot_start_time,slot_end_time;
    unsigned long num_events_per_time_measurement;
    if(this->MaxSlotConsumedTime != 0UL){
        num_events_per_time_measurement=0UL;
        QueryPerformanceCounter(&slot_start_time);
        }
#endif
    real_time_restriction=false;
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

                NewEvent->ProcessEvent(this, real_time_restriction);

		delete NewEvent;
		
#if defined(_WIN32) || defined(_WIN64)
        if(this->MaxSlotConsumedTime != 0UL){
            unsigned long slot_elapsed_time;
            if(num_events_per_time_measurement > NUM_EVENTS_PER_TIME_SLOT_CHECK){
                num_events_per_time_measurement=0UL;
                QueryPerformanceCounter(&slot_end_time);
                slot_elapsed_time=(unsigned long)(slot_end_time.QuadPart-slot_start_time.QuadPart);
                if(slot_elapsed_time > this->MaxSlotConsumedTime){
                    real_time_restriction=true;
                }
            }
            else
                num_events_per_time_measurement++;
        }
#endif
    }
}

void Simulation::WriteSpike(const Spike * spike){
	Neuron * neuron=spike->GetSource();  // source of the spike
    
    if(neuron->IsMonitored()){
		for (list<OutputSpikeDriver *>::iterator it=this->MonitorSpike.begin(); it!=this->MonitorSpike.end(); ++it){
			if (!(*it)->IsWritePotentialCapable()){
				(*it)->WriteSpike(spike);
			}
		}
	}
	   	
	if(neuron->IsOutput()){
		for (list<OutputSpikeDriver *>::iterator it=this->OutputSpike.begin(); it!=this->OutputSpike.end(); ++it){
			(*it)->WriteSpike(spike);
		}
	}		
}

void Simulation::WriteState(float time, Neuron * neuron){
	for (list<OutputSpikeDriver *>::iterator it=this->MonitorSpike.begin(); it!=this->MonitorSpike.end(); ++it){
		if ((*it)->IsWritePotentialCapable()){
			(*it)->WriteState(time, neuron);
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
//	cout << "Sending outputs in time " << this->CurrentSimulationTime << endl;
	for (list<OutputSpikeDriver *>::iterator it=this->OutputSpike.begin(); it!=this->OutputSpike.end(); ++it){
		if ((*it)->IsBuffered()){
			(*it)->FlushBuffers();	
		}
	}
}

void Simulation::GetInput(){
//	cout << "Getting inputs in time " << this->CurrentSimulationTime << endl;
	for (list<InputSpikeDriver *>::iterator it=this->InputSpike.begin(); it!=this->InputSpike.end(); ++it){
		if (!(*it)->IsFinished()){
			(*it)->LoadInputs(this->Queue, this->Net);
		}
	}
}

long Simulation::GetTotalSpikeCounter(){
	return this->TotalSpikeCounter;
}

void Simulation::SetTotalSpikeCounter(long int value) {
	this->TotalSpikeCounter = value;
}

void Simulation::IncrementTotalSpikeCounter() {
	this->TotalSpikeCounter++;
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

InputSpikeDriver *Simulation::GetInputSpikeDriver(unsigned int ElementPosition){
    InputSpikeDriver *list_element;
    if(this->InputSpike.size() > ElementPosition){
        list<InputSpikeDriver *>::iterator list_it = this->InputSpike.begin();
        advance(list_it, ElementPosition);
        list_element=*list_it;
    } else
        list_element=NULL;
    return list_element;
}

void Simulation::RemoveInputSpikeDriver(InputSpikeDriver * NewInput){
	this->InputSpike.remove(NewInput);
}
		 
void Simulation::AddOutputSpikeDriver(OutputSpikeDriver * NewOutput){
	this->OutputSpike.push_back(NewOutput);	
}

OutputSpikeDriver *Simulation::GetOutputSpikeDriver(unsigned int ElementPosition){
    OutputSpikeDriver *list_element;
    if(this->OutputSpike.size() > ElementPosition){
        list<OutputSpikeDriver *>::iterator list_it = this->OutputSpike.begin();
        advance(list_it, ElementPosition);
        list_element=*list_it;
    } else
        list_element=NULL;
    return list_element;
}

void Simulation::RemoveOutputSpikeDriver(OutputSpikeDriver * NewOutput){
	this->OutputSpike.remove(NewOutput);
}

void Simulation::AddMonitorActivityDriver(OutputSpikeDriver * NewMonitor){
	this->MonitorSpike.push_back(NewMonitor);
}

OutputSpikeDriver *Simulation::GetMonitorActivityDriver(unsigned int ElementPosition){
    OutputSpikeDriver *list_element;
    if(this->MonitorSpike.size() > ElementPosition){
        list<OutputSpikeDriver *>::iterator list_it = this->MonitorSpike.begin();
        advance(list_it, ElementPosition);
        list_element=*list_it;
    } else
        list_element=NULL;
    return list_element;
}
		
void Simulation::RemoveMonitorActivityDriver(OutputSpikeDriver * NewMonitor){
	this->MonitorSpike.remove(NewMonitor);
}

void Simulation::AddOutputWeightDriver(OutputWeightDriver * NewOutput){
	this->OutputWeight.push_back(NewOutput);	
}

OutputWeightDriver *Simulation::GetOutputWeightDriver(unsigned int ElementPosition){
    OutputWeightDriver *list_element;
    if(this->OutputWeight.size() > ElementPosition){
        list<OutputWeightDriver *>::iterator list_it = this->OutputWeight.begin();
        advance(list_it, ElementPosition);
        list_element=*list_it;
    } else
        list_element=NULL;
    return list_element;
}

void Simulation::RemoveOutputWeightDriver(OutputWeightDriver * NewOutput){
	this->OutputWeight.remove(NewOutput);
}

ostream & Simulation::PrintInfo(ostream & out) {
	out << "- Simulation:" << endl;

	out << "  * End simulation time: " << this->GetTotalSimulationTime() << " s."<< endl;

	out << "  * Saving weight step time: " << this->GetSaveStep() << " s." << endl;

	out << "  * Communication step time: " << this->GetSimulationStep() << " s." << endl;

	out << "  * Total simulation time: " << this->GetTotalSimulationTime() << " s." << endl;

	this->GetNetwork()->PrintInfo(out);

	out << "  * Input spike channels: " << this->InputSpike.size() << endl;

	for (list<InputSpikeDriver *>::iterator it=this->InputSpike.begin(); it!=this->InputSpike.end(); ++it){
		(*it)->PrintInfo(out);
	}

	out << "  * Output spike channels: " << this->OutputSpike.size() << endl;

	for (list<OutputSpikeDriver *>::iterator it=this->OutputSpike.begin(); it!=this->OutputSpike.end(); ++it){
		(*it)->PrintInfo(out);
	}

	out << "  * Monitor spike channels: " << this->MonitorSpike.size() << endl;

	for (list<OutputSpikeDriver *>::iterator it=this->MonitorSpike.begin(); it!=this->MonitorSpike.end(); ++it){
		(*it)->PrintInfo(out);
	}

	out << "  * Saving weight channels: " << this->OutputWeight.size() << endl;

	for (list<OutputWeightDriver *>::iterator it=this->OutputWeight.begin(); it!=this->OutputWeight.end(); ++it){
		(*it)->PrintInfo(out);
	}

	out << "  * Network description:" << endl;
	this->GetNetwork()->PrintInfo(out);

	return out;
}
