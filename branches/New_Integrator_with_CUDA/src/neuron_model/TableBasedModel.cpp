/***************************************************************************
 *                           TableBasedModel.cpp                           *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#include "../../include/neuron_model/TableBasedModel.h"
#include "../../include/neuron_model/NeuronModelTable.h"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/Interconnection.h"
#include "../../include/spike/PropagatedSpike.h"

#include "../../include/simulation/Utils.h"

#include <string>

void TableBasedModel::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
	if(fh){
		Currentline=1L;
		skip_comments(fh,Currentline);
		if(fscanf(fh,"%i",&this->NumStateVar)==1){
			unsigned int nv;

			// Initialize all vectors.
			this->StateVarTable = (NeuronModelTable **) new NeuronModelTable * [this->NumStateVar];
			this->StateVarOrder = (unsigned int *) new unsigned int [this->NumStateVar];

			// Auxiliary table index
			unsigned int * TablesIndex = (unsigned int *) new unsigned int [this->NumStateVar];

			skip_comments(fh,Currentline);
			for(nv=0;nv<this->NumStateVar;nv++){
				if(fscanf(fh,"%i",&TablesIndex[nv])!=1){
					throw EDLUTFileException(13,41,3,1,Currentline);
					delete [] TablesIndex;
				}
			}

			skip_comments(fh,Currentline);

			float InitValue;
			InitValues=new float[NumStateVar+1]();

			// Create a new initial state
       		this->InitialState = (VectorNeuronState *) new VectorNeuronState(this->NumStateVar+1, false);
//       		this->InitialState->SetLastUpdateTime(0);
//       		this->InitialState->SetNextPredictedSpikeTime(NO_SPIKE_PREDICTED);
//      		this->InitialState->SetStateVariableAt(0,0);

       		for(nv=0;nv<this->NumStateVar;nv++){
       			if(fscanf(fh,"%f",&InitValue)!=1){
       				throw EDLUTFileException(13,42,3,1,Currentline);
					delete [] TablesIndex;
       			} else {
					InitValues[nv+1]=InitValue;
//       			this->InitialState->SetStateVariableAt(nv+1,InitValue);
       			}
       		}

			// Allocate temporal state vars
			this->TempStateVars = (float *) new float [this->InitialState->GetNumberOfVariables()];
			
   			skip_comments(fh,Currentline);
   			unsigned int FiringIndex, FiringEndIndex;
   			if(fscanf(fh,"%i",&FiringIndex)==1){
   				skip_comments(fh,Currentline);
   				if(fscanf(fh,"%i",&FiringEndIndex)==1){
   					skip_comments(fh,Currentline);
   					if(fscanf(fh,"%i",&this->NumSynapticVar)==1){
               			skip_comments(fh,Currentline);

               			this->SynapticVar = (unsigned int *) new unsigned int [this->NumSynapticVar];
               			for(nv=0;nv<this->NumSynapticVar;nv++){
                  			if(fscanf(fh,"%i",&this->SynapticVar[nv])!=1){
                  				throw EDLUTFileException(13,40,3,1,Currentline);
								delete [] TablesIndex;
                  			}
                  		}

              			skip_comments(fh,Currentline);
              			if(fscanf(fh,"%i",&this->NumTables)==1){
              				unsigned int nt;
              				int tdeptables[MAXSTATEVARS];
              				int tstatevarpos,ntstatevarpos;

              				this->Tables = (NeuronModelTable *) new NeuronModelTable [this->NumTables];

              				// Update table links
              				for(nv=0;nv<this->NumStateVar;nv++){
								this->StateVarTable[nv] = this->Tables+TablesIndex[nv];
							}
              				this->FiringTable = this->Tables+FiringIndex;
              				this->EndFiringTable = this->Tables+FiringEndIndex;

              				for(nt=0;nt<this->NumTables;nt++){
              					this->Tables[nt].LoadTableDescription(fh, Currentline);
                   			}

              				this->NumTimeDependentStateVar = 0;
                 			for(nt=0;nt<this->NumStateVar;nt++){
         						for(nv=0;nv<this->StateVarTable[nt]->GetDimensionNumber() && this->StateVarTable[nt]->GetDimensionAt(nv)->statevar != 0;nv++);
            					if(nv<this->StateVarTable[nt]->GetDimensionNumber()){
            						tdeptables[nt]=1;
            						this->NumTimeDependentStateVar++;
            					}else{
               						tdeptables[nt]=0;
            					}
            				}

         					tstatevarpos=0;
         					ntstatevarpos=this->NumTimeDependentStateVar; // we place non-t-depentent variables in the end, so that they are evaluated afterwards
         					for(nt=0;nt<this->NumStateVar;nt++){
            					this->StateVarOrder[(tdeptables[nt])?tstatevarpos++:ntstatevarpos++]=nt;
         					}
              			}else{
         					throw EDLUTFileException(13,37,3,1,Currentline);
							delete [] TablesIndex;
      					}
      				}else{
       					throw EDLUTFileException(13,36,3,1,Currentline);
						delete [] TablesIndex;
          			}
				}else{
    				throw EDLUTFileException(13,43,3,1,Currentline);
					delete [] TablesIndex;
          		}
			}else{
 				throw EDLUTFileException(13,35,3,1,Currentline);
				delete [] TablesIndex;
			}

			delete [] TablesIndex;
		}else{
			throw EDLUTFileException(13,34,3,1,Currentline);
		}
	}else{
		throw EDLUTFileException(13,25,13,0,Currentline);
	}
}

void TableBasedModel::LoadTables(string TableFile) throw (EDLUTException){
	FILE *fd;
	unsigned int i;
	NeuronModelTable * tab;
	fd=fopen(TableFile.c_str(),"rb");
	if(fd){
		for(i=0;i<this->NumTables;i++){
			tab=&this->Tables[i];
			tab->LoadTable(fd);
		}
		fclose(fd);
	}else{
		throw EDLUTException(10,24,13,0);
	}
}

TableBasedModel::TableBasedModel(string NeuronTypeID, string NeuronModelID): EventDrivenNeuronModel(NeuronTypeID, NeuronModelID),
		NumStateVar(0), NumTimeDependentStateVar(0), NumSynapticVar(0), SynapticVar(0),
		StateVarOrder(0), StateVarTable(0), FiringTable(0), EndFiringTable(0),
		NumTables(0), Tables(0), TempStateVars(0) {

}

TableBasedModel::~TableBasedModel() {
	
	if (this->StateVarOrder!=0) {
		delete [] this->StateVarOrder;
	}

	if (this->StateVarTable!=0) {
		delete [] this->StateVarTable;
	}

	if (this->SynapticVar!=0) {
		delete [] this->SynapticVar;
	}

	if (this->Tables!=0) {
		delete [] this->Tables;
	}

	if (this->TempStateVars!=0) {
		delete [] this->TempStateVars;
	}

	if(this->InitValues!=0){
		delete [] InitValues;
	}
}

void TableBasedModel::LoadNeuronModel() throw (EDLUTFileException){

	this->LoadNeuronModel(this->GetModelID()+".cfg");

	this->LoadTables(this->GetModelID()+".dat");

}

VectorNeuronState * TableBasedModel::InitializeState(){
	//return (NeuronState *) new NeuronState(*(this->InitialState));
	return InitialState;
}

InternalSpike * TableBasedModel::GenerateInitialActivity(Neuron *  Cell){
	double Predicted = this->NextFiringPrediction(Cell->GetIndex_VectorNeuronState(),Cell->GetVectorNeuronState());

	InternalSpike * spike = 0;

	if(Predicted != NO_SPIKE_PREDICTED){
		Predicted += Cell->GetVectorNeuronState()->GetLastUpdateTime(Cell->GetIndex_VectorNeuronState());

		spike = new InternalSpike(Predicted,Cell);
	}

	return spike;
}

void TableBasedModel::UpdateState(int index, VectorNeuronState * State, double CurrentTime){
	unsigned int ivar,orderedvar;

	State->SetStateVariableAt(index,0,CurrentTime-State->GetLastUpdateTime(index));

	for(ivar=0;ivar<this->NumTimeDependentStateVar;ivar++){
		orderedvar=this->StateVarOrder[ivar];
		this->TempStateVars[orderedvar]=this->StateVarTable[orderedvar]->TableAccess(index,State);
	}

	for(ivar=0;ivar<this->NumTimeDependentStateVar;ivar++){
		orderedvar=this->StateVarOrder[ivar];
		State->SetStateVariableAt(index,orderedvar+1,this->TempStateVars[orderedvar]);
	}

	for(ivar=this->NumTimeDependentStateVar;ivar<this->NumStateVar;ivar++){
		orderedvar=this->StateVarOrder[ivar];
		State->SetStateVariableAt(index,orderedvar+1,this->StateVarTable[orderedvar]->TableAccess(index,State));
	}

	State->SetLastUpdateTime(index,CurrentTime);

}

void TableBasedModel::SynapsisEffect(int index, VectorNeuronState * State, Interconnection * InputConnection){
	State->IncrementStateVariableAtCPU(index, this->SynapticVar[InputConnection->GetType()]+1, InputConnection->GetWeight()*WEIGHTSCALE);
}

double TableBasedModel::NextFiringPrediction(int index, VectorNeuronState * State){
	return this->FiringTable->TableAccess(index, State);
}

double TableBasedModel::EndRefractoryPeriod(int index, VectorNeuronState * State){
	return this->EndFiringTable->TableAccess(index, State);
}


InternalSpike * TableBasedModel::ProcessInputSpike(PropagatedSpike *  InputSpike){

	Interconnection * inter = InputSpike->GetSource()->GetOutputConnectionAt(InputSpike->GetTarget());

	int TargetIndex=inter->GetTarget()->GetIndex_VectorNeuronState();

	Neuron * TargetCell = inter->GetTarget();

	VectorNeuronState * CurrentState = TargetCell->GetVectorNeuronState();

	// Update the neuron state until the current time
	this->UpdateState(TargetIndex,CurrentState,InputSpike->GetTime());

	// Add the effect of the input spike
	this->SynapsisEffect(TargetIndex,CurrentState,inter);

	InternalSpike * GeneratedSpike = 0;

	// Check if an spike will be fired
	double NextSpike = this->NextFiringPrediction(TargetIndex,CurrentState);
	if (NextSpike != NO_SPIKE_PREDICTED){
		NextSpike += CurrentState->GetLastUpdateTime(TargetIndex);

		if (NextSpike>CurrentState->GetEndRefractoryPeriod(TargetIndex)){
			GeneratedSpike = new InternalSpike(NextSpike,TargetCell);
		} else { // Only for neurons which never stop firing
			// The generated spike was at refractory period -> Check after refractoriness

			VectorNeuronState newState(*CurrentState, TargetIndex);

			this->UpdateState(0,&newState,newState.GetEndRefractoryPeriod(0));

			NextSpike = this->NextFiringPrediction(0,&newState);

			if(NextSpike != NO_SPIKE_PREDICTED){
				NextSpike += CurrentState->GetEndRefractoryPeriod(TargetIndex);

				GeneratedSpike = new InternalSpike(NextSpike,TargetCell);
			}
		}
	}

	CurrentState->SetNextPredictedSpikeTime(TargetIndex,NextSpike);

	return GeneratedSpike;
}


InternalSpike * TableBasedModel::ProcessInputSpike(Interconnection * inter, Neuron * target, double time){

	int TargetIndex=target->GetIndex_VectorNeuronState();


	VectorNeuronState * CurrentState = target->GetVectorNeuronState();

	// Update the neuron state until the current time
	this->UpdateState(TargetIndex,CurrentState,time);

	// Add the effect of the input spike
	this->SynapsisEffect(TargetIndex,CurrentState,inter);

	InternalSpike * GeneratedSpike = 0;

	// Check if an spike will be fired
	double NextSpike = this->NextFiringPrediction(TargetIndex,CurrentState);
	if (NextSpike != NO_SPIKE_PREDICTED){
		NextSpike += CurrentState->GetLastUpdateTime(TargetIndex);

		if (NextSpike>CurrentState->GetEndRefractoryPeriod(TargetIndex)){
			GeneratedSpike = new InternalSpike(NextSpike,target);
		} else { // Only for neurons which never stop firing
			// The generated spike was at refractory period -> Check after refractoriness

			VectorNeuronState newState(*CurrentState, TargetIndex);

			this->UpdateState(0,&newState,newState.GetEndRefractoryPeriod(0));

			NextSpike = this->NextFiringPrediction(0,&newState);

			if(NextSpike != NO_SPIKE_PREDICTED){
				NextSpike += CurrentState->GetEndRefractoryPeriod(TargetIndex);

				GeneratedSpike = new InternalSpike(NextSpike,target);
			}
		}
	}

	CurrentState->SetNextPredictedSpikeTime(TargetIndex,NextSpike);

	return GeneratedSpike;
}


InternalSpike * TableBasedModel::GenerateNextSpike(InternalSpike *  OutputSpike){

	Neuron * SourceCell = OutputSpike->GetSource();

	int SourceIndex=SourceCell->GetIndex_VectorNeuronState();

	VectorNeuronState * CurrentState = SourceCell->GetVectorNeuronState();

	InternalSpike * NextSpike = 0;

	this->UpdateState(SourceIndex,CurrentState,OutputSpike->GetTime());

	double EndRefractory = this->EndRefractoryPeriod(SourceIndex,CurrentState);

	if(EndRefractory != NO_SPIKE_PREDICTED){
		EndRefractory += OutputSpike->GetTime();
	}else{
		EndRefractory = OutputSpike->GetTime()+DEF_REF_PERIOD;
		cerr << "Warning: firing table and firing-end table discrepance (using default ref period)" << endl;
	}

	CurrentState->SetEndRefractoryPeriod(SourceIndex,EndRefractory);

	// Check if some auto-activity is generated after the refractory period
	VectorNeuronState PostFiringState (*CurrentState, SourceIndex);

	this->UpdateState(0,&PostFiringState,CurrentState->GetEndRefractoryPeriod(SourceIndex));

	double PredictedSpike = this->NextFiringPrediction(0,&PostFiringState);

	if(PredictedSpike != NO_SPIKE_PREDICTED){
		PredictedSpike += CurrentState->GetEndRefractoryPeriod(SourceIndex);

		NextSpike = new InternalSpike(PredictedSpike,SourceCell);
	}

	CurrentState->SetNextPredictedSpikeTime(SourceIndex,PredictedSpike);

	return NextSpike;
}


bool TableBasedModel::DiscardSpike(InternalSpike *  OutputSpike){
	return (OutputSpike->GetSource()->GetVectorNeuronState()->GetNextPredictedSpikeTime(OutputSpike->GetSource()->GetIndex_VectorNeuronState())!=OutputSpike->GetTime());
}

ostream & TableBasedModel::PrintInfo(ostream & out) {
	out << "- Table-Based Model: " << this->GetModelID() << endl;

	for(unsigned int itab=0;itab<this->NumTables;itab++){
		out << this->Tables[itab].GetDimensionNumber() << " " << this->Tables[itab].GetInterpolation() << " (" << this->Tables[itab].GetFirstInterpolation() << ")\t";

		for(unsigned int idim=0;idim<this->Tables[itab].GetDimensionNumber();idim++){
			out << this->Tables[itab].GetDimensionAt(idim)->statevar << " " << this->Tables[itab].GetDimensionAt(idim)->interp << " (" << this->Tables[itab].GetDimensionAt(idim)->nextintdim << ")\t";
		}
	}

	out << endl;

	return out;
}


void TableBasedModel::InitializeStates(int N_neurons){
	InitialState->InitializeStates(N_neurons, InitValues);
}