/***************************************************************************
 *                           Network_GPU.cpp                               *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido, Richard Carrillo and  *
 *						: Francisco Naveros                                *
 * email                : jgarrido@atc.ugr.es, fnaveros@ugr.es             *
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
 * \note: this file Network_GPU.cpp must be used instead of file Network.cpp to
 * implement a CPU-GPU hybrid architecture.
*/


#include "../../include/spike/Network.h"
#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"

#include "../../include/learning_rules/SimetricCosSTDPWeightChange.h"
#include "../../include/learning_rules/SimetricCosSinSTDPWeightChange.h"
#include "../../include/learning_rules/LearningRuleFactory.h"

#include "../../include/neuron_model/NeuronModelFactory.h"
#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/EventDrivenNeuronModel.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/simulation/EventQueue.h"
#include "../../include/simulation/Utils.h"
#include "../../include/simulation/Configuration.h"
#include "../../include/simulation/RandomGenerator.h"

#include "../../include/simulation/NetworkDescription.h"

#include "../../include/cudaError.h"

#include "../../include/openmp/openmp.h"


int qsort_inters(const void *e1, const void *e2){
	int ord;
	double ordf;

	//Calculate source index
	ord=((Interconnection *)e1)->GetTarget()->get_OpenMP_queue_index() - ((Interconnection *)e2)->GetTarget()->get_OpenMP_queue_index();
	if(!ord){
		//the same source index-> calculate target OpenMP index
		ord=((Interconnection *)e1)->GetSource()->GetIndex() - ((Interconnection *)e2)->GetSource()->GetIndex();
		if(!ord){
			ordf=((Interconnection *)e1)->GetDelay() - ((Interconnection *)e2)->GetDelay();
			if(ordf<0.0){
				ord=-1;
			}else if(ordf>0.0){
				ord=1;
			}//The same propagation time-> calculate targe index
			else if(ordf==0){
				ord=((Interconnection *)e1)->GetTarget()->GetIndex() - ((Interconnection *)e2)->GetTarget()->GetIndex();
			}
		}
	}

	return(ord);
}

void Network::FindOutConnections(){
	// Change the ordenation
   	qsort(inters,ninters,sizeof(Interconnection),qsort_inters);
	if(ninters>0){
		// Calculate the number of input connections with learning for each cell
		unsigned long ** NumberOfOutputs = (unsigned long **) new unsigned long *[this->nneurons];
		unsigned long ** OutputsLeft = (unsigned long **) new unsigned long *[this->nneurons];
		for (unsigned long i=0; i<this->nneurons; i++){
			NumberOfOutputs[i]=(unsigned long *) new unsigned long [this->GetNumberOfQueues()]();
			OutputsLeft[i]=(unsigned long *) new unsigned long [this->GetNumberOfQueues()];
		}



		for (unsigned long con= 0; con<this->ninters; ++con){
			NumberOfOutputs[this->inters[con].GetSource()->GetIndex()][this->inters[con].GetTarget()->get_OpenMP_queue_index()]++;
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			for(int i=0; i<this->GetNumberOfQueues(); i++){
				OutputsLeft[neu][i] = NumberOfOutputs[neu][i];
			}
		}

		Interconnection **** OutputConnections = (Interconnection ****) new Interconnection *** [this->nneurons];
		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			OutputConnections[neu] = (Interconnection ***) new Interconnection ** [this->GetNumberOfQueues()];
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			for(int i=0; i<this->GetNumberOfQueues(); i++){
				if (NumberOfOutputs[neu][i]>0){
					OutputConnections[neu][i] = (Interconnection **) new Interconnection * [NumberOfOutputs[neu][i]];
				} else {
					OutputConnections[neu][i] = 0;
				}
			}
		}

		for (unsigned long con= this->ninters-1; con<this->ninters; --con){
			unsigned long SourceCell = this->inters[con].GetSource()->GetIndex();
			int OpenMP_index=this->inters[con].GetTarget()->get_OpenMP_queue_index();
			OutputConnections[SourceCell][OpenMP_index][--OutputsLeft[SourceCell][OpenMP_index]] = this->inters+con;
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			this->neurons[neu].SetOutputConnections(OutputConnections[neu],NumberOfOutputs[neu]);
		}

		delete [] OutputConnections;
		delete [] NumberOfOutputs;
		delete [] OutputsLeft;
	}
}

void Network::SetWeightOrdination(){
	if (ninters>0){
		for (int ninter=0; ninter<ninters;ninter++){
			int index = inters[ninter].GetIndex();
			this->wordination[index] = &(this->inters[ninter]);
		}
	}
}

void Network::FindInConnections(){
	if(this->ninters>0){

		// Calculate the number of input connections with learning for each cell
		unsigned int ** NumberOfInputsWithPostSynapticLearning = (unsigned int **) new unsigned int * [this->nneurons]();
		unsigned int ** InputsLeftWithPostSynapticLearning = (unsigned int **) new unsigned int * [this->nneurons];

		unsigned int ** NumberOfInputsWithTriggerSynapticLearning = (unsigned int **) new unsigned int * [this->nneurons]();
		unsigned int ** InputsLeftWithTriggerSynapticLearning = (unsigned int **) new unsigned int * [this->nneurons];

		unsigned int ** NumberOfInputsWithPostAndTriggerSynapticLearning = (unsigned int **) new unsigned int * [this->nneurons]();
		unsigned int ** InputsLeftWithPostAndTriggerSynapticLearning = (unsigned int **) new unsigned int * [this->nneurons];

		for (unsigned long i=0; i<this->nneurons; ++i){
			NumberOfInputsWithPostSynapticLearning[i] = (unsigned int *) new unsigned int [this->nwchanges]();
			InputsLeftWithPostSynapticLearning[i] = (unsigned int *) new unsigned int [this->nwchanges]();
			NumberOfInputsWithTriggerSynapticLearning[i] = (unsigned int *) new unsigned int [this->nwchanges]();
			InputsLeftWithTriggerSynapticLearning[i] = (unsigned int *) new unsigned int [this->nwchanges]();
			NumberOfInputsWithPostAndTriggerSynapticLearning[i] = (unsigned int *) new unsigned int [this->nwchanges]();
			InputsLeftWithPostAndTriggerSynapticLearning[i] = (unsigned int *) new unsigned int [this->nwchanges]();
		}

		for (unsigned long con= 0; con<this->ninters; ++con){
			if(this->inters[con].GetWeightChange_withPost()!=0){
				unsigned int n_rule = this->inters[con].GetWeightChange_withPost()->GetLearningRuleIndex();
				NumberOfInputsWithPostSynapticLearning[this->inters[con].GetTarget()->GetIndex()][n_rule]++;
			}
			if(this->inters[con].GetWeightChange_withTrigger()!=0){
				unsigned int n_rule = this->inters[con].GetWeightChange_withTrigger()->GetLearningRuleIndex();
				NumberOfInputsWithTriggerSynapticLearning[this->inters[con].GetTarget()->GetIndex()][n_rule]++;
			}
			if(this->inters[con].GetWeightChange_withPostAndTrigger()!=0){
				unsigned int n_rule = this->inters[con].GetWeightChange_withPostAndTrigger()->GetLearningRuleIndex();
				NumberOfInputsWithPostAndTriggerSynapticLearning[this->inters[con].GetTarget()->GetIndex()][n_rule]++;
			}
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			for (unsigned int wcindex = 0; wcindex<this->nwchanges; ++wcindex){
				InputsLeftWithPostSynapticLearning[neu][wcindex] = NumberOfInputsWithPostSynapticLearning[neu][wcindex];
				InputsLeftWithTriggerSynapticLearning[neu][wcindex] = NumberOfInputsWithTriggerSynapticLearning[neu][wcindex];
				InputsLeftWithPostAndTriggerSynapticLearning[neu][wcindex] = NumberOfInputsWithPostAndTriggerSynapticLearning[neu][wcindex];
			}
		}

		Interconnection **** InputConnectionsWithPostSynapticLearning = (Interconnection ****) new Interconnection *** [this->nneurons];
		Interconnection **** InputConnectionsWithTriggerSynapticLearning = (Interconnection ****) new Interconnection *** [this->nneurons];
		Interconnection **** InputConnectionsWithPostAndTriggerSynapticLearning = (Interconnection ****) new Interconnection *** [this->nneurons];

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			InputConnectionsWithPostSynapticLearning[neu] = (Interconnection ***) new Interconnection ** [this->nwchanges];
			InputConnectionsWithTriggerSynapticLearning[neu] = (Interconnection ***) new Interconnection ** [this->nwchanges];
			InputConnectionsWithPostAndTriggerSynapticLearning[neu] = (Interconnection ***) new Interconnection ** [this->nwchanges];

			for (unsigned int wcindex=0; wcindex<this->nwchanges; ++wcindex){
				InputConnectionsWithPostSynapticLearning[neu][wcindex] = 0;

				if (NumberOfInputsWithPostSynapticLearning[neu][wcindex]>0){
					InputConnectionsWithPostSynapticLearning[neu][wcindex] = (Interconnection **) new Interconnection * [NumberOfInputsWithPostSynapticLearning[neu][wcindex]];
				}

				InputConnectionsWithTriggerSynapticLearning[neu][wcindex] = 0;

				if (NumberOfInputsWithTriggerSynapticLearning[neu][wcindex]>0){
					InputConnectionsWithTriggerSynapticLearning[neu][wcindex] = (Interconnection **) new Interconnection * [NumberOfInputsWithTriggerSynapticLearning[neu][wcindex]];
				}

				InputConnectionsWithPostAndTriggerSynapticLearning[neu][wcindex] = 0;

				if (NumberOfInputsWithPostAndTriggerSynapticLearning[neu][wcindex]>0){
					InputConnectionsWithPostAndTriggerSynapticLearning[neu][wcindex] = (Interconnection **) new Interconnection * [NumberOfInputsWithPostAndTriggerSynapticLearning[neu][wcindex]];
				}
			}
		}

		for (unsigned long con= this->ninters-1; con<this->ninters; --con){
			if (this->inters[con].GetWeightChange_withPost()!=0){
				unsigned long TargetCell = this->inters[con].GetTarget()->GetIndex();
				unsigned int wcindex = this->inters[con].GetWeightChange_withPost()->GetLearningRuleIndex();
				InputConnectionsWithPostSynapticLearning[TargetCell][wcindex][--InputsLeftWithPostSynapticLearning[TargetCell][wcindex]] = this->inters+con;
			}
			if (this->inters[con].GetWeightChange_withTrigger()!=0){
				unsigned long TargetCell = this->inters[con].GetTarget()->GetIndex();
				unsigned int wcindex = this->inters[con].GetWeightChange_withTrigger()->GetLearningRuleIndex();
				InputConnectionsWithTriggerSynapticLearning[TargetCell][wcindex][--InputsLeftWithTriggerSynapticLearning[TargetCell][wcindex]] = this->inters+con;
			}
			if (this->inters[con].GetWeightChange_withPostAndTrigger()!=0){
				unsigned long TargetCell = this->inters[con].GetTarget()->GetIndex();
				unsigned int wcindex = this->inters[con].GetWeightChange_withPostAndTrigger()->GetLearningRuleIndex();
				InputConnectionsWithPostAndTriggerSynapticLearning[TargetCell][wcindex][--InputsLeftWithPostAndTriggerSynapticLearning[TargetCell][wcindex]] = this->inters+con;
			}
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			this->neurons[neu].SetInputConnectionsWithPostSynapticLearning(InputConnectionsWithPostSynapticLearning[neu],NumberOfInputsWithPostSynapticLearning[neu],this->nwchanges);
			this->neurons[neu].SetInputConnectionsWithTriggerSynapticLearning(InputConnectionsWithTriggerSynapticLearning[neu],NumberOfInputsWithTriggerSynapticLearning[neu],this->nwchanges);
			this->neurons[neu].SetInputConnectionsWithPostAndTriggerSynapticLearning(InputConnectionsWithPostAndTriggerSynapticLearning[neu],NumberOfInputsWithPostAndTriggerSynapticLearning[neu],this->nwchanges);


			//learning rule index are ordered by input synapses index in order to improve trigger learning.
			for (unsigned int wcindex=0; wcindex<this->nwchanges; ++wcindex){
				for (unsigned long aux = 0; aux < NumberOfInputsWithTriggerSynapticLearning[neu][wcindex]; aux++){
					InputConnectionsWithTriggerSynapticLearning[neu][wcindex][aux]->SetLearningRuleIndex_withTrigger(InputConnectionsWithTriggerSynapticLearning[neu][wcindex][aux]->GetWeightChange_withTrigger()->counter);
					InputConnectionsWithTriggerSynapticLearning[neu][wcindex][aux]->GetWeightChange_withTrigger()->counter++;
				}
			}

			//learning rule index are ordered by input synapses index in order to improve postsynaptic learning.
			for (unsigned int wcindex=0; wcindex<this->nwchanges; ++wcindex){
				for (unsigned long aux = 0; aux < NumberOfInputsWithPostSynapticLearning[neu][wcindex]; aux++){
					unsigned int index = InputConnectionsWithPostSynapticLearning[neu][wcindex][aux]->GetWeightChange_withPost()->counter;
					InputConnectionsWithPostSynapticLearning[neu][wcindex][aux]->SetLearningRuleIndex_withPost(index);
					InputConnectionsWithPostSynapticLearning[neu][wcindex][aux]->GetWeightChange_withPost()->counter++;
				}
			}

			//learning rule index are ordered by input synapses index in order to improve postsynaptic learning.
			for (unsigned int wcindex=0; wcindex<this->nwchanges; ++wcindex){
				for (unsigned long aux = 0; aux < NumberOfInputsWithPostAndTriggerSynapticLearning[neu][wcindex]; aux++){
					unsigned int index = InputConnectionsWithPostAndTriggerSynapticLearning[neu][wcindex][aux]->GetWeightChange_withPostAndTrigger()->counter;
					InputConnectionsWithPostAndTriggerSynapticLearning[neu][wcindex][aux]->SetLearningRuleIndex_withPostAndTrigger(index);
					InputConnectionsWithPostAndTriggerSynapticLearning[neu][wcindex][aux]->GetWeightChange_withPostAndTrigger()->counter++;
				}
			}

			//Initialize the learning rule index in aech neuron in order to improve cache friendly.
			this->neurons[neu].initializeLearningRuleIndex();

		}

		delete [] InputsLeftWithPostSynapticLearning;
		delete [] InputsLeftWithTriggerSynapticLearning;
		delete [] InputsLeftWithPostAndTriggerSynapticLearning;
		delete [] NumberOfInputsWithPostSynapticLearning;
		delete [] NumberOfInputsWithTriggerSynapticLearning;
		delete [] NumberOfInputsWithPostAndTriggerSynapticLearning;

	}
}

void Network::InitializeStates(int ** N_neurons){
	for( int z=0; z< this->nneutypes; z++){
		for(int j=0; j<this->GetNumberOfQueues(); j++){
			if(N_neurons[z][j]>0){
				neutypes[z][j]->InitializeStates(N_neurons[z][j], j);
			}else{
				neutypes[z][j]->InitializeStates(1,j);
			}
		}
	}
}



void Network::InitNetPredictions(EventQueue * Queue){
	int nneu;
	for(nneu=0;nneu<nneurons;nneu++){
		if (neurons[nneu].GetNeuronModel()->GetModelSimulationMethod()==EVENT_DRIVEN_MODEL && neurons[nneu].GetNeuronModel()->GetModelType()==NEURAL_LAYER){
			EventDrivenNeuronModel * Model = (EventDrivenNeuronModel *) neurons[nneu].GetNeuronModel();
			InternalSpike * spike = Model->GenerateInitialActivity(neurons+nneu);
			if (spike!=0){
				Queue->InsertEvent(spike->GetSource()->get_OpenMP_queue_index(),spike);
			}
		}
	}

}

Network::Network(const std::list<NeuronLayerDescription> & neuron_layer_list,
				 const std::list<ModelDescription> & learning_rule_list,
				 const std::list<SynapticLayerDescription> & synaptic_layer_list,
				 EventQueue * Queue, int numberOfQueues) noexcept(false) :
        inters(0), ninters(0), neutypes(0), nneutypes(0), neurons(0), nneurons(0), timedrivenneurons(0),
        ntimedrivenneurons(0), wchanges(0), nwchanges(0), wordination(0), NumberOfQueues(numberOfQueues),
        minpropagationdelay(0.0001), invminpropagationdelay(1.0/0.0001){
  	this->LoadNet(neuron_layer_list, learning_rule_list, synaptic_layer_list);
	this->InitNetPredictions(Queue);
	this->CalculaElectricalCouplingDepedencies();

}

Network::~Network(){
	if (inters!=0) {
		delete [] inters;
	}

	if (neutypes!=0) {
		for (int i=0; i<this->nneutypes; ++i){
			if (this->neutypes[i]!=0){
				for( int j=0; j<this->GetNumberOfQueues(); j++){
					if (this->neutypes[i][j]!=0){
						if(ntimedrivenneurons_GPU[i][0]>0){
							HANDLE_ERROR(cudaSetDevice(GPUsIndex[j % NumberOfGPUs]));
						}
						delete this->neutypes[i][j];
					}
				}
				delete [] this->neutypes[i];
			}
		}
		delete [] neutypes;
	}


	if (neurons!=0) {
		delete [] neurons;
	}

	if (timedrivenneurons!=0) {
		for (int z=0; z<this->nneutypes; z++){
			if(timedrivenneurons[z]!=0){
				for(int j=0; j<this->GetNumberOfQueues(); j++){
					delete [] timedrivenneurons[z][j];
				}
				delete [] timedrivenneurons[z];
			}
		}
		delete [] timedrivenneurons;
	}

	if(ntimedrivenneurons!=0){
		for(int i=0; i<this->nneutypes; i++){
			if(ntimedrivenneurons[i]!=0){
				delete [] ntimedrivenneurons[i];
			}
		}
		delete [] ntimedrivenneurons;
	}

	if (timedrivenneurons_GPU!=0) {
		for (int z=0; z<this->nneutypes; z++){
			if(timedrivenneurons_GPU[z]!=0){
				for(int j=0; j<this->GetNumberOfQueues(); j++){
					delete [] timedrivenneurons_GPU[z][j];
				}
				delete [] timedrivenneurons_GPU[z];
			}
		}
		delete [] timedrivenneurons_GPU;
	}

	if(ntimedrivenneurons_GPU!=0){
		for(int i=0; i<this->nneutypes; i++){
			if(ntimedrivenneurons_GPU[i]!=0){
				delete [] ntimedrivenneurons_GPU[i];
			}
		}
		delete [] ntimedrivenneurons_GPU;
	}

	if (wchanges!=0) {
		for (int i=0; i<this->nwchanges; ++i){
			delete this->wchanges[i];
		}
		delete [] wchanges;
	}
	if (wordination!=0) delete [] wordination;
}

Neuron * Network::GetNeuronAt(int index) const{
	return &(this->neurons[index]);
}

int Network::GetNeuronNumber() const{
	return this->nneurons;
}

Neuron ** Network::GetTimeDrivenNeuronAt(int index0, int index1) const{
	return this->timedrivenneurons[index0][index1];
}

Neuron * Network::GetTimeDrivenNeuronAt(int index0,int index1, int index2) const{
	return this->timedrivenneurons[index0][index1][index2];
}

Neuron ** Network::GetTimeDrivenNeuronGPUAt(int index0, int index1) const{
	return this->timedrivenneurons_GPU[index0][index1];
}

Neuron * Network::GetTimeDrivenNeuronGPUAt(int index0,int index1, int index2) const{
	return this->timedrivenneurons_GPU[index0][index1][index2];
}

int ** Network::GetTimeDrivenNeuronNumber() const{
	return this->ntimedrivenneurons;
}

int Network::GetNneutypes() const{
	return this->nneutypes;
}
int ** Network::GetTimeDrivenNeuronNumberGPU() const{
	return this->ntimedrivenneurons_GPU;
}

NeuronModel ** Network::GetNeuronModelAt(int index) const{
	return this->neutypes[index];
}

NeuronModel * Network::GetNeuronModelAt(int index1, int index2) const{
	return this->neutypes[index1][index2];
}

LearningRule * Network::GetLearningRuleAt(int index) const{
	return this->wchanges[index];
}

int Network::GetLearningRuleNumber() const{
	return this->nwchanges;
}

void Network::ParseNet(const char *netfile, const char *wfile,
							  std::list<NeuronLayerDescription> & neuron_layer_list,
							  std::list<ModelDescription> & learning_rule_list,
							  std::list<SynapticLayerDescription> & synaptic_layer_list) noexcept(false){

	// Clear all the description lists passed as argument
	neuron_layer_list.clear();
	learning_rule_list.clear();
	synaptic_layer_list.clear();

	// Open the file
	FILE *fh;
	long Currentline=1L;
	fh=fopen(netfile,"rt");
	if(!fh) {
		throw EDLUTFileException(TASK_NETWORK_LOAD, ERROR_NETWORK_OPEN, REPAIR_NETWORK_OPEN, Currentline, netfile);
	}

	// Load neuron models
	skip_comments(fh, Currentline);
	// Load neuron models
	int lnneutypes = 0;
	if (fscanf(fh, "%i", &lnneutypes) != 1 || lnneutypes <= 0) {
		throw EDLUTFileException(TASK_NETWORK_LOAD, ERROR_NETWORK_NEURON_MODEL_LOAD_NUMBER,
								 REPAIR_NETWORK_NEURON_MODEL_LOAD_NUMBER, Currentline, netfile);
	}

	skip_comments(fh, Currentline);
	int lnneurons = 0;
	if (fscanf(fh, "%i", &lnneurons) != 1 || lnneurons <= 0) {
		throw EDLUTFileException(TASK_NETWORK_LOAD, ERROR_NETWORK_READ_NUMBER_OF_NEURONS,
								 REPAIR_NETWORK_READ_NUMBER_OF_NEURONS, Currentline, netfile);
	}

	// Load each neuron layer row
	int nn = 0;
	for(int tind=0;tind<lnneurons;tind+=nn) {
		skip_comments(fh, Currentline);
		int outn, monit;
		char ident[MAXIDSIZE + 1];
		char ident_type[MAXIDSIZE + 1];
		if (fscanf(fh, "%i", &nn) != 1 || nn <= 0 ||
//			fscanf(fh, " %"MAXIDSIZEC"[^ ]%*[^ ]", ident_type) != 1 ||
			fscanf(fh, "%64s", ident_type) != 1 ||
//			fscanf(fh, " %"MAXIDSIZEC"[^ ]%*[^ ]", ident) != 1 ||
			fscanf(fh, "%64s", ident) != 1 ||
			fscanf(fh, "%i", &outn) != 1 ||
			fscanf(fh, "%i", &monit) != 1) {
			throw EDLUTFileException(TASK_NETWORK_LOAD, ERROR_NETWORK_NEURON_PARAMETERS,
									 REPAIR_NETWORK_NEURON_PARAMETERS, Currentline, netfile);
		}

		// Check the number of neurons and the total
		if (tind + nn > lnneurons) {
			throw EDLUTFileException(TASK_NETWORK_LOAD, ERROR_NETWORK_NUMBER_OF_NEURONS,
									 REPAIR_NETWORK_NUMBER_OF_NEURONS, Currentline, netfile);
		}

		// Add the new neuron layer description struct to the list
		ModelDescription nmodel = NeuronModelFactory::ParseNeuronModel(ident_type, ident);
		nmodel.model_name = ident_type;
		NeuronLayerDescription nlayer;
		nlayer.neuron_model = nmodel;
		nlayer.output_activity = outn;
		nlayer.log_activity = monit;
		nlayer.num_neurons = nn;
		neuron_layer_list.push_back(nlayer);
	}

	// Load the learning rules
	skip_comments(fh,Currentline);
	int lnwchanges = 0;
	if (fscanf(fh, "%i", &lnwchanges) != 1 || lnwchanges < 0) {
		throw EDLUTFileException(TASK_NETWORK_LOAD_LEARNING_RULES, ERROR_NETWORK_LEARNING_RULE_NUMBER, REPAIR_NETWORK_LEARNING_RULE_NUMBER, Currentline, netfile);
	}
	for(int wcind=0;wcind<lnwchanges;wcind++) {
		char ident_type[MAXIDSIZE + 1];
		skip_comments(fh, Currentline);
		string LearningModel;
//		if (fscanf(fh, " %"MAXIDSIZEC"[^ \n]%*[^ ]", ident_type) != 1) {
		if (fscanf(fh, "%64s", ident_type) != 1) {
			throw EDLUTFileException(TASK_NETWORK_LOAD_LEARNING_RULES, ERROR_NETWORK_LEARNING_RULE_LOAD,
									 REPAIR_NETWORK_LEARNING_RULE_LOAD, Currentline, netfile);
		}

		ModelDescription lrule;
		// Load the parameters of the new learning rule
		try {
			 lrule = LearningRuleFactory::ParseLearningRule(ident_type, fh);
		} catch (EDLUTException exc) {
			throw EDLUTFileException(exc, Currentline, netfile);
		}

		learning_rule_list.push_back(lrule);
	}

	// Load the synapses from the file
	skip_comments(fh,Currentline);
	long int lninters = 0;

	if (fscanf(fh, "%li", &lninters) != 1 || lninters <= 0) {
		throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_LOAD_NUMBER, REPAIR_NETWORK_SYNAPSES_LOAD_NUMBER, Currentline, netfile);
	}

	int source,nsources,target,ntargets,nreps;
	int intwchange1, intwchange2;
	bool trigger1, trigger2;
	float delay,delayinc,maxweight;
	int type;
	int iind;

	// Create an only synapsis layer for all the synapses
	SynapticLayerDescription synaptic_layer;
	synaptic_layer.source_neuron_list.resize(lninters);
	synaptic_layer.target_neuron_list.resize(lninters);
	std::vector<float> delay_list(lninters,0.001);
	std::vector<int> type_list(lninters,0);
	std::vector<float> weight_list(lninters,1.0);
	std::vector<float> max_weight_list(lninters,1.0);
	std::vector<int> triggerwchange_list(lninters, -1);
	std::vector<int> wchange_list(lninters, -1);

	// Iterate over all the synapses to be created
	for(iind=0;iind<lninters;iind+=nsources*ntargets*nreps) {
		skip_comments(fh, Currentline);
		if (fscanf(fh, "%i", &source) != 1 ||
			fscanf(fh, "%i", &nsources) != 1 ||
			fscanf(fh, "%i", &target) != 1 ||
			fscanf(fh, "%i", &ntargets) != 1 ||
			fscanf(fh, "%i", &nreps) != 1 ||
			fscanf(fh, "%f", &delay) != 1 ||
			fscanf(fh, "%f", &delayinc) != 1 ||
			fscanf(fh, "%i", &type) != 1 ||
			fscanf(fh, "%f", &maxweight) != 1) {
			throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_LOAD,
									 REPAIR_NETWORK_SYNAPSES_LOAD, Currentline, netfile);
		}

		//Load the first learning rule index
		trigger1 = false;
		//Check if the synapse implement a trigger learning rule
		if (fscanf(fh, " t%d", &intwchange1) == 1) {
			trigger1 = true;
		} else {
			//Check if the synapse implement a non trigger learning rule
			if (fscanf(fh, "%d", &intwchange1) == 1) {
				trigger1 = false;
			} else {
				throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_FIRST_LEARNING_RULE_LOAD,
										 REPAIR_NETWORK_SYNAPSES_LEARNING_RULE_INDEX, Currentline, netfile);
			}
		}

		if (intwchange1 >= int(learning_rule_list.size())) {
			throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_FIRST_LEARNING_RULE_INDEX,REPAIR_NETWORK_SYNAPSES_LEARNING_RULE_INDEX, Currentline, netfile);
		}

		//Load the second learning rule index
		intwchange2 = -1;
		trigger2 = false;
		if (intwchange1 >= 0) {
			//Check if there is a second learning rule defined for this synapse.
			if (is_end_line(fh, Currentline) == false) {
				//Check if the synapse implement a trigger learning rule
				if (fscanf(fh, " t%d", &intwchange2) == 1) {
					trigger2 = true;
				} else {
					//Check if the synapse implement a non trigger learning rule
					if (fscanf(fh, "%d", &intwchange2) == 1) {
						trigger2 = false;
					} else {
						throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES,
												 ERROR_NETWORK_SYNAPSES_SECOND_LEARNING_RULE_LOAD,
												 REPAIR_NETWORK_SYNAPSES_LEARNING_RULE_INDEX, Currentline, netfile);
					}
				}

				if (intwchange2 >= int(learning_rule_list.size())) {
					throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES,
											 ERROR_NETWORK_SYNAPSES_SECOND_LEARNING_RULE_INDEX,
											 REPAIR_NETWORK_SYNAPSES_LEARNING_RULE_INDEX, Currentline, netfile);
				}
			}
		}

		//Check if the number of synapses do not exceed the total number of synapses.
		if (iind + nsources * ntargets * nreps > lninters) {
			throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_NUMBER,
									 REPAIR_NETWORK_SYNAPSES_NUMBER, Currentline, netfile);
		}

		// Check if there are 2 non-trigger learning rules
		if (trigger1 == false && (intwchange2 >= 0 && trigger2 == false)) {
			throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_LEARNING_RULE_NON_TRIGGER,
									 REPAIR_NETWORK_SYNAPSES_LEARNING_RULE, Currentline, netfile);
		}

		// Check if there are 2 trigger learning rules
		if (trigger1 == true && trigger2 == true) {
			throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_LEARNING_RULE_TRIGGER,
									 REPAIR_NETWORK_SYNAPSES_LEARNING_RULE, Currentline, netfile);
		}

		// Fill the positions of the synaptic layers
		for (int rind = 0; rind < nreps; rind++) {
			for (int sind = 0; sind < nsources; sind++) {
				for (int tind = 0; tind < ntargets; tind++) {
					int posc = iind + rind * nsources * ntargets + sind * ntargets + tind;
					synaptic_layer.source_neuron_list[posc] = source + rind * nsources + sind;
					synaptic_layer.target_neuron_list[posc] = target + rind * ntargets + tind;
					delay_list[posc] = delay + delayinc * tind;
					type_list[posc] = type;
					max_weight_list[posc] = maxweight;
					if (trigger1 && intwchange1 >= 0) {
						triggerwchange_list[posc] = intwchange1;
					} else if (intwchange1 >= 0) {
						wchange_list[posc] = intwchange1;
					}
					if (trigger2 && intwchange2 >= 0) {
						triggerwchange_list[posc] = intwchange2;
					} else if (intwchange2 >= 0) {
						wchange_list[posc] = intwchange2;
					}
				}
			}
		}
	}

	fclose(fh);

	// Load the weights from the corresponding file
    int connind;
    Currentline=1L;
    fh=fopen(wfile,"rt");
    if(!fh) {
        throw EDLUTFileException(TASK_WEIGHTS_LOAD, ERROR_WEIGHTS_OPEN, REPAIR_WEIGHTS_OPEN, Currentline, wfile);
    }
    int nweights,weind;
    float weight;
	//Random generator
	RandomGenerator randomGenerator;

    skip_comments(fh,Currentline);
    for(connind=0;connind<synaptic_layer.source_neuron_list.size();connind+=nweights){
        skip_comments(fh, Currentline);
        if(fscanf(fh,"%i",&nweights)!=1 || fscanf(fh,"%f",&weight)!=1) {
            throw EDLUTFileException(TASK_WEIGHTS_LOAD, ERROR_WEIGHTS_READ, REPAIR_WEIGHTS_READ, Currentline, wfile);
        }

        if(nweights < 0 || nweights + connind > synaptic_layer.source_neuron_list.size()){
            throw EDLUTFileException(TASK_WEIGHTS_LOAD, ERROR_WEIGHTS_NUMBER, REPAIR_WEIGHTS_NUMBER, Currentline, wfile);
        }

        if(nweights == 0){
            nweights=lninters-connind;
        }

        for(weind=0;weind<nweights;weind++){
            double calc_weight = 0.0;
            if (weight<0.0){
                calc_weight=randomGenerator.frand()*max_weight_list[connind+weind];
            } else if (weight>max_weight_list[connind+weind]) {
                calc_weight=max_weight_list[connind+weind];
            } else {
                calc_weight = weight;
            }
            weight_list[connind+weind]=calc_weight;
        }
    }
    fclose(fh);

	// Include all the vectors in the param map
	synaptic_layer.param_map["delay"] = boost::any(delay_list);
	synaptic_layer.param_map["type"] = boost::any(type_list);
	synaptic_layer.param_map["weight"] = boost::any(weight_list);
	synaptic_layer.param_map["max_weight"] = boost::any(max_weight_list);
	synaptic_layer.param_map["trigger_wchange"] = boost::any(triggerwchange_list);
	synaptic_layer.param_map["wchange"] = boost::any(wchange_list);

	synaptic_layer_list.push_back(synaptic_layer);

	return;
}


void Network::LoadNet(const std::list<NeuronLayerDescription> & neuron_layer_list,
			 const std::list<ModelDescription> & learning_rule_list,
			 const std::list<SynapticLayerDescription> & synaptic_layer_list) noexcept(false){
	std::list<NeuronLayerDescription>::const_iterator it;

	// -----------------------------------------------------------------------------
	// Initialize neuron layers and models
	// -----------------------------------------------------------------------------

	this->CreateNeuronLayers(neuron_layer_list);

	// -----------------------------------------------------------------------------
	// Initialize learning rules
	// -----------------------------------------------------------------------------

	std::vector<int> N_ConectionWithLearning = this->CreateWeightChanges(learning_rule_list);

	// ------------------------------------------------------------------------------------------
	// Initialize network synapses
	// ------------------------------------------------------------------------------------------
	this->CreateConnections(synaptic_layer_list, N_ConectionWithLearning);

	// Initialize connection states
	this->InitializeLearningRuleState(N_ConectionWithLearning);

	// Once all the synapses have been created, sort them as convenient
	this->FindOutConnections();
	this->SetWeightOrdination(); // must be before find_in_c() and after find_out_c()
	this->FindInConnections();


	//Synaptic weight initialization in learning rules with dopamine
	for (int i = 0; i < this->ninters; i++){
		if (inters[i].GetWeightChange_withPost() != 0){
			ConnectionState * state = inters[i].GetWeightChange_withPost()->GetConnectionState();
			if (state !=0){
				state->SetWeight(inters[i].GetLearningRuleIndex_withPost(), inters[i].GetWeight(), inters[i].GetMaxWeight());
			}
		}
		if (inters[i].GetWeightChange_withTrigger() != 0){
			ConnectionState * state = inters[i].GetWeightChange_withTrigger()->GetConnectionState();
			if (state !=0){
				state->SetWeight(inters[i].GetLearningRuleIndex_withTrigger(), inters[i].GetWeight(), inters[i].GetMaxWeight());
			}
		}
		if (inters[i].GetWeightChange_withPostAndTrigger() != 0){
			ConnectionState * state = inters[i].GetWeightChange_withPostAndTrigger()->GetConnectionState();
			if (state !=0){
				state->SetWeight(inters[i].GetLearningRuleIndex_withPostAndTrigger(), inters[i].GetWeight(), inters[i].GetMaxWeight());
			}
		}
	}


	//Calculate the output delay structure of each neuron. This structure is used by PropagatepSpikeGroup event to group several
	//PropagatedSpike events in just one.
	for(int i=0; i<this->GetNeuronNumber(); i++){
		this->GetNeuronAt(i)->CalculateOutputDelayStructure();
	}
	for(int i=0; i<this->GetNeuronNumber(); i++){
		this->GetNeuronAt(i)->CalculateOutputDelayIndex();
	}


	//Initialize the Input Current Sypases Structure in each neuron model if it implements this kind of input synapses.
	for (int z = 0; z < this->nneutypes; z++){
		for (int j = 0; j < this->GetNumberOfQueues(); j++){
			neutypes[z][j]->InitializeInputCurrentSynapseStructure();
		}
	}

	return;

}

void Network::InitializeLearningRuleState(const vector<int> &N_ConectionWithLearning) const {
	for(int t=0; t < nwchanges; t++){
		if(N_ConectionWithLearning[t]>0){
			wchanges[t]->InitializeConnectionState(N_ConectionWithLearning[t], this->GetNeuronNumber());
		}
	}
}

void Network::CreateConnections(const list<SynapticLayerDescription> &synaptic_layer_list,
											std::vector<int> & NConnectionsPerLearning) {// Check the number of synapses
	std::list<SynapticLayerDescription>::const_iterator it4;
	ninters = 0;
	for (it4=synaptic_layer_list.begin(); it4!=synaptic_layer_list.end(); ++it4) {
		ninters += it4->source_neuron_list.size();
	}

	int iind,posc;

	// Allocate memory for synapse storage
	inters =(Interconnection *) new Interconnection [ninters];
	wordination =(Interconnection **) new Interconnection * [ninters];

	// Create all the synapses
	posc = 0;
	for (it4=synaptic_layer_list.begin(),iind=0; it4!=synaptic_layer_list.end(); ++it4, iind+=it4->source_neuron_list.size()) {

		std::map<std::string, boost::any> param_map = it4->param_map;

    // Check the number of source and target neurons
    if (it4->source_neuron_list.size()!=it4->target_neuron_list.size()) {
      throw EDLUTException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_NUMBER, REPAIR_NETWORK_SYNAPSES_NUMBER);
    }

    // Establish synapses default values
    std::vector<float> delay(it4->source_neuron_list.size(),0.001);
    std::vector<int> type(it4->source_neuron_list.size(), 0);
    std::vector<float> weight(it4->source_neuron_list.size(), 1.0);
		std::vector<float> max_weight(it4->source_neuron_list.size(), 1.0);
		std::vector<int> inttchange(it4->source_neuron_list.size(), -1);
		std::vector<int> intwchange(it4->source_neuron_list.size(), -1);


		// Set the delay (if it exists)
		std::map<std::string, boost::any>::iterator it5 = param_map.find("delay");
		if (it5!=param_map.end()) {
			if (param_map["delay"].type() == typeid(float)){
				float lr = boost::any_cast<float>(param_map["delay"]);
				for (std::vector<float>::iterator itv=delay.begin(); itv!=delay.end(); ++itv){
					*itv = lr;
				}
			} else if (param_map["delay"].type() == typeid(std::vector<float>)) {
				std::vector<float> lr = boost::any_cast<std::vector<float> >(param_map["delay"]);
				for (std::vector<float>::iterator itv=delay.begin(), itv2=lr.begin(); itv!=delay.end() && itv2!=lr.end(); ++itv, ++itv2){
					*itv = *itv2;
				}
			}
			param_map.erase(it5);
		}

		// Set the synaptic type (if it exists)
		it5 = param_map.find("type");
		if (it5!=param_map.end()) {
			if (param_map["type"].type() == typeid(int)){
				int lr = boost::any_cast<int>(param_map["type"]);
				for (std::vector<int>::iterator itv=type.begin(); itv!=type.end(); ++itv){
					*itv = lr;
				}
			} else if (param_map["type"].type() == typeid(std::vector<int>)) {
				std::vector<int> lr = boost::any_cast<std::vector<int> >(param_map["type"]);
				for (std::vector<int>::iterator itv=type.begin(), itv2=lr.begin(); itv!=type.end() && itv2!=lr.end(); ++itv, ++itv2){
					*itv = *itv2;
				}
			}
			param_map.erase(it5);
		}

		// Set the weight (if it exists)
		it5 = param_map.find("weight");
		if (it5!=param_map.end()) {
			if (param_map["weight"].type() == typeid(float)){
				float lr = boost::any_cast<float>(param_map["weight"]);
				for (std::vector<float>::iterator itv=weight.begin(); itv!=weight.end(); ++itv){
					*itv = lr;
				}
			} else if (param_map["weight"].type() == typeid(std::vector<float>)) {
				std::vector<float> lr = boost::any_cast<std::vector<float> >(param_map["weight"]);
				for (std::vector<float>::iterator itv=weight.begin(), itv2=lr.begin(); itv!=weight.end() && itv2!=lr.end(); ++itv, ++itv2){
					*itv = *itv2;
				}
			}
			param_map.erase(it5);
		}

		// Set the max_weight (if it exists)
		it5 = param_map.find("max_weight");
		if (it5!=param_map.end()) {
			if (param_map["max_weight"].type() == typeid(float)){
				float lr = boost::any_cast<float>(param_map["max_weight"]);
				for (std::vector<float>::iterator itv=max_weight.begin(); itv!=max_weight.end(); ++itv){
					*itv = lr;
				}
			} else if (param_map["max_weight"].type() == typeid(std::vector<float>)) {
				std::vector<float> lr = boost::any_cast<std::vector<float> >(param_map["max_weight"]);
				for (std::vector<float>::iterator itv=max_weight.begin(), itv2=lr.begin(); itv!=max_weight.end() && itv2!=lr.end(); ++itv, ++itv2){
					*itv = *itv2;
				}
			}
			param_map.erase(it5);
		}

		// Check the trigger weight change (if it exists)
		it5 = param_map.find("trigger_wchange");
		if (it5 != param_map.end()) {
			if (param_map["trigger_wchange"].type() == typeid(int)){
				int lr = boost::any_cast<int>(param_map["trigger_wchange"]);
				for (std::vector<int>::iterator itv = inttchange.begin(); itv != inttchange.end(); ++itv){
					*itv = lr;
				}
			}
			else if (param_map["trigger_wchange"].type() == typeid(std::vector<int>)) {
				std::vector<int> lr = boost::any_cast<std::vector<int> >(param_map["trigger_wchange"]);
				for (std::vector<int>::iterator itv = inttchange.begin(), itv2 = lr.begin(); itv != inttchange.end() && itv2 != lr.end(); ++itv, ++itv2){
					*itv = *itv2;
				}
			}
			param_map.erase(it5);
		}

		// Check the weight change (if it exists)
		it5 = param_map.find("wchange");
		if (it5 != param_map.end()) {
			if (param_map["wchange"].type() == typeid(int)){
				int lr = boost::any_cast<int>(param_map["wchange"]);
				for (std::vector<int>::iterator itv = intwchange.begin(); itv != intwchange.end(); ++itv){
					*itv = lr;
				}
			}
			else if (param_map["wchange"].type() == typeid(std::vector<int>)) {
				std::vector<int> lr = boost::any_cast<std::vector<int> >(param_map["wchange"]);
				for (std::vector<int>::iterator itv = intwchange.begin(), itv2 = lr.begin(); itv != intwchange.end() && itv2 != lr.end(); ++itv, ++itv2){
					*itv = *itv2;
				}
			}
			param_map.erase(it5);
		}

		//Check if the dictionary is empty
		if (!param_map.empty()){
			it5 = param_map.begin();
			cout << "UNKNOW PARAMETERS: ";
			while (it5 != param_map.end()){
				cout << it5->first << " ";
				it5++;
			}
			cout << endl;
			throw EDLUTException(TASK_NETWORK_LOAD_SYNAPSES_FROM_DICTIONARY, ERROR_NETWORK_LOAD_FROM_DICTIONARY, REPAIR_NETWORK_LOAD_FROM_DICTIONARY);
		}


		// Check the index of source and target neurons
    for (unsigned int id = 0; id<it4->source_neuron_list.size(); ++id, ++posc) {
      if (it4->source_neuron_list[id] >= nneurons) {
        throw EDLUTException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_NEURON_INDEX, REPAIR_NETWORK_SYNAPSES_NEURON_INDEX);
      }

      if (it4->target_neuron_list[id] >= nneurons) {
        throw EDLUTException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_NEURON_INDEX, REPAIR_NETWORK_SYNAPSES_NEURON_INDEX);
      }

      inters[posc].SetIndex(posc);
      inters[posc].SetSource(&(neurons[it4->source_neuron_list[id]]));
      inters[posc].SetTarget(&(neurons[it4->target_neuron_list[id]]));
			inters[posc].SetTargetNeuronModel(neurons[it4->target_neuron_list[id]].GetNeuronModel());
			inters[posc].SetTargetNeuronModelIndex(neurons[it4->target_neuron_list[id]].GetIndex_VectorNeuronState());
      inters[posc].SetDelay(RoundPropagationDelay(delay[id]));
			inters[posc].SetType(type[id]);

			if (inters[posc].GetTarget()->GetNeuronModel()->CheckSynapseType(&inters[posc]) == false){
        throw EDLUTException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_TYPE, REPAIR_NETWORK_SYNAPSES_TYPE);
      }

			inters[posc].SetMaxWeight(max_weight[id]);
      inters[posc].SetWeight(weight[id]);

      inters[posc].SetWeightChange_withPost(0);
      inters[posc].SetWeightChange_withTrigger(0);
			inters[posc].SetWeightChange_withPostAndTrigger(0);


			//load the learning rule
			int wchange_id = intwchange[id];
			if(wchange_id >= 0){
        if (wchange_id >= nwchanges){
          throw EDLUTException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_FIRST_LEARNING_RULE_INDEX, REPAIR_NETWORK_SYNAPSES_LEARNING_RULE_INDEX);
        }

        //Set the new learning rule
        if(wchanges[wchange_id]->ImplementPostSynaptic() == true){
					if (wchanges[wchange_id]->ImplementTriggerSynaptic() == true) {
						inters[posc].SetWeightChange_withPostAndTrigger(wchanges[wchange_id]);
					}else{
						inters[posc].SetWeightChange_withPost(wchanges[wchange_id]);
					}
        }else{
          inters[posc].SetWeightChange_withTrigger(wchanges[wchange_id]);
        }
        NConnectionsPerLearning[wchange_id]++;
      }

      //load the trigger learning rule
			int twchange_id = inttchange[id];
      if(twchange_id >= 0) {
        if (twchange_id >= nwchanges){
          throw EDLUTException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_SECOND_LEARNING_RULE_INDEX, REPAIR_NETWORK_SYNAPSES_LEARNING_RULE_INDEX);
        }

        //Set the new learning rule
        if (wchanges[twchange_id]->ImplementTriggerSynaptic() == true) {
					if (wchanges[twchange_id]->ImplementPostSynaptic() == false){
	       		inters[posc].SetWeightChange_withTrigger(wchanges[twchange_id]);
					}else{
						inters[posc].SetWeightChange_withPostAndTrigger(wchanges[twchange_id]);
					}
					inters[posc].SetTriggerConnection();
        }
				else{
					throw EDLUTException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_POSTSYNAPTIC_TRIGGER, REPAIR_NETWORK_SYNAPSES_LEARNING_RULE_INDEX);
				}
        NConnectionsPerLearning[twchange_id]++;
      }
    }
	}
	return;
}

void Network::CreateNeuronLayers(const std::list<NeuronLayerDescription> &neuron_layer_list) {// Check the number of neuron models
	std::list<NeuronLayerDescription>::const_iterator it, it2;
	std::vector<NeuronModel *> TempNeuronModels;
    std::vector<NeuronModel *>::const_iterator it3;
	std::vector<int> NeuronTypeIndex;

	// Load the neuron models and obtain those which are distinct
    for (it=neuron_layer_list.begin(); it!=neuron_layer_list.end(); ++it) {
        NeuronModel * type = NeuronModelFactory::CreateNeuronModel(it->neuron_model);

        bool found = false;
        int type_index;
        for (it3=TempNeuronModels.begin(), type_index=0; it3!=TempNeuronModels.end() && !found; ++it3){
            found = type->compare(*it3);
            if (!found) type_index++;
        }
        if (!found){
            TempNeuronModels.push_back(type);
        }
        NeuronTypeIndex.push_back(type_index);
    }

    unsigned int distinct_models = TempNeuronModels.size();

	// Allocate memory for the number of neuron models
	nneutypes = distinct_models;
	neutypes =(NeuronModel ***) new NeuronModel ** [nneutypes];
	// Allocate memory for the number of neuron models
	for (int ni = 0; ni < nneutypes; ni++) {
		neutypes[ni] = (NeuronModel **) new NeuronModel *[GetNumberOfQueues()];
		for (int n = 0; n < GetNumberOfQueues(); n++) {
			ModelDescription nmodel;
			nmodel.param_map = (*TempNeuronModels[ni]).GetParameters();
			nmodel.model_name = boost::any_cast<std::string>(nmodel.param_map["name"]);
			neutypes[ni][n] = NeuronModelFactory::CreateNeuronModel(nmodel);
		}
		// Delete temporal neuron model
		delete TempNeuronModels[ni];
		TempNeuronModels[ni] = 0;
	}

	// Count the total number of neurons
	nneurons = 0;
	for (it=neuron_layer_list.begin(); it!=neuron_layer_list.end(); ++it) {
		nneurons += it->num_neurons;
	}

	// Allocate memory for the requested number of neuron
	int tind,nind;
	NeuronModel ** type;
	neurons =(Neuron *) new Neuron [nneurons];

	ntimedrivenneurons = (int**) new int* [nneutypes]();
	int *** time_driven_index = (int ***) new int **[nneutypes];

	ntimedrivenneurons_GPU= (int **) new int* [this->nneutypes]();
	int *** time_driven_index_GPU=(int ***) new int **[this->nneutypes];

	int ** N_neurons= (int **) new int *[nneutypes]();

	for (int z=0; z < nneutypes; z++){
		ntimedrivenneurons[z]=new int [GetNumberOfQueues()]();
		ntimedrivenneurons_GPU[z]=new int [this->GetNumberOfQueues()]();

		time_driven_index[z]=(int**)new int* [GetNumberOfQueues()];
		time_driven_index_GPU[z]=(int**)new int* [this->GetNumberOfQueues()];
		for(int j=0; j < GetNumberOfQueues(); j++){
			time_driven_index[z][j]=new int [nneurons]();
			time_driven_index_GPU[z][j]=new int [this->nneurons]();
		}

		N_neurons[z]=new int [GetNumberOfQueues()]();
	}

	// Load each neuron model type
    std::vector<int>::const_iterator it4;
	for (it=neuron_layer_list.begin(), tind=0, it4=NeuronTypeIndex.begin();
	        it!=neuron_layer_list.end() && it4!=NeuronTypeIndex.end();
		 	tind+=it->num_neurons, ++it, ++it4) {
		type = this->neutypes[*it4];

		int blockSize= (it->num_neurons + NumberOfQueues - 1) / NumberOfQueues;
		int blockIndex;
		for(nind=0;nind<it->num_neurons;nind++){
			blockIndex=nind/blockSize;

			// Initialize the neurons
			neurons[nind + tind].InitNeuron(nind + tind, N_neurons[*it4][blockIndex], type[blockIndex],
											it->log_activity, it->output_activity, blockIndex);

			N_neurons[*it4][blockIndex]++;

			// Set monitoring neurons if it is not an input device
			if (it->log_activity && type[0]->GetModelType() != INPUT_DEVICE){
				for(int n=0; n < GetNumberOfQueues(); n++){
					type[n]->GetVectorNeuronState()->Set_Is_Monitored(true);
				}
				this->monitore_neurons = true;
			}

			// Set time driven neurons
			if (type[0]->GetModelSimulationMethod() == TIME_DRIVEN_MODEL_CPU){
				time_driven_index[*it4][blockIndex][ntimedrivenneurons[*it4][blockIndex]] = nind + tind;
				ntimedrivenneurons[*it4][blockIndex]++;
			}

			if (type[0]->GetModelSimulationMethod()==TIME_DRIVEN_MODEL_GPU){
				time_driven_index_GPU[*it4][blockIndex][ntimedrivenneurons_GPU[*it4][blockIndex]] = nind + tind;
				ntimedrivenneurons_GPU[*it4][blockIndex]++;
			}
		}
	}

	// Create the time-driven cell array
	timedrivenneurons =(Neuron ****) new Neuron *** [nneutypes]();
	for (int z=0; z < nneutypes; z++){
		if (ntimedrivenneurons[z][0] > 0){
			timedrivenneurons[z]=(Neuron ***) new Neuron ** [GetNumberOfQueues()];
			for(int j=0; j < GetNumberOfQueues(); j++){
				timedrivenneurons[z][j]=(Neuron **) new Neuron * [ntimedrivenneurons[z][j]];
				for (int i=0; i < ntimedrivenneurons[z][j]; ++i){
					timedrivenneurons[z][j][i] = &(neurons[time_driven_index[z][j][i]]);
				}
			}
		}
	}

	// Create the time-driven cell array for GPU
	timedrivenneurons_GPU=(Neuron ****) new Neuron *** [this->nneutypes]();
	for (int z=0; z<this->nneutypes; z++){
		if (this->ntimedrivenneurons_GPU[z][0]>0){
			this->timedrivenneurons_GPU[z]=(Neuron ***) new Neuron ** [this->GetNumberOfQueues()];
			for(int j=0; j<this->GetNumberOfQueues(); j++){
				this->timedrivenneurons_GPU[z][j]=(Neuron **) new Neuron * [this->ntimedrivenneurons_GPU[z][j]];
				for (int i=0; i<this->ntimedrivenneurons_GPU[z][j]; ++i){
					this->timedrivenneurons_GPU[z][j][i] = &(this->neurons[time_driven_index_GPU[z][j][i]]);
				}
			}
		}
	}


	// Initialize states.
	InitializeStates(N_neurons);

	// Release temporal arrays
	for (int z=0; z < nneutypes; z++){
		for(int j=0; j < GetNumberOfQueues(); j++){
			delete [] time_driven_index[z][j];
			delete [] time_driven_index_GPU[z][j];
		}
		delete [] time_driven_index[z];
		delete [] time_driven_index_GPU[z];
		delete [] N_neurons[z];
	}
	delete [] time_driven_index;
	delete [] time_driven_index_GPU;
	delete [] N_neurons;
}

std::vector<int> Network::CreateWeightChanges(const std::list<ModelDescription> &learning_rule_list) {
	std::list<ModelDescription>::const_iterator it3;
	nwchanges = learning_rule_list.size();

	// Initialize a vector with the number of synapses per learning rule
	std::vector<int> NConnectionsPerLearning = std::vector<int>(this->nwchanges,0);

	unsigned int wcind;
	this->wchanges =new LearningRule * [nwchanges];
	for(wcind=0, it3=learning_rule_list.begin(); wcind < nwchanges; wcind++,++it3) {
		this->wchanges[wcind] = LearningRuleFactory::CreateLearningRule(*it3);
		this->wchanges[wcind]->SetLearningRuleIndex(wcind);
	}

	return NConnectionsPerLearning;
}

void Network::SaveWeights(const char *wfile) noexcept(false){
	FILE *fh;
	int connind;
	fh=fopen(wfile,"wt");
	if(fh){
		float weight,antweight;
		int nantw;
		nantw=0;
		antweight=0.0;
		weight=0.0; // just to avoid compiler warning messages

		for(connind=0;connind<=this->ninters;connind++){
			if(connind < this->ninters){
				weight=this->wordination[connind]->GetWeight();
			}

			if(antweight != weight || connind == this->ninters){
				if(nantw > 0){
					if(fprintf(fh,"%i %g\n",nantw,antweight) <= 0){
						throw EDLUTException(TASK_WEIGHTS_SAVE, ERROR_WEIGHTS_SAVE, REPAIR_WEIGHTS_SAVE);
					}
				}

				antweight=weight;
				nantw=1;
			}else{
				nantw++;
			}
		}

		// fprintf(fh,"// end of written data\n");

		fclose(fh);
	}else{
		throw EDLUTException(TASK_WEIGHTS_SAVE, ERROR_WEIGHTS_SAVE_OPEN, REPAIR_WEIGHTS_SAVE);
	}

}

void Network::GetCompressedWeights(std::vector<int> & N_equal_weights, std::vector<float> & equal_weights){
	int connind;

	float weight, antweight;
	int nantw;
	nantw = 0;
	antweight = 0.0;
	weight = 0.0; // just to avoid compiler warning messages

	for (connind = 0; connind <= this->ninters; connind++){
		if (connind < this->ninters){
			weight = this->wordination[connind]->GetWeight();
		}

		if (antweight != weight || connind == this->ninters){
			if (nantw > 0){
				N_equal_weights.push_back(nantw);
				equal_weights.push_back(antweight);
			}

			antweight = weight;
			nantw = 1;
		}
		else{
			nantw++;
		}
	}
}

std::vector<float> Network::GetWeights(){
	int connind;
	std::vector<float> outputweights(this->ninters);
	std::vector<float>::iterator itoutputweights = outputweights.begin();
	for (connind = 0; connind < this->ninters; connind++, itoutputweights++){
		*itoutputweights = this->wordination[connind]->GetWeight();
	}
	return outputweights;
}

std::vector<float> Network::GetSelectedWeights(std::vector<int> synaptic_indexes){
	std::vector<float> outputweights(synaptic_indexes.size(), -1.0f);

	std::vector<int>::iterator itsynaptic_indexes = synaptic_indexes.begin();
	std::vector<float>::iterator itoutputweights = outputweights.begin();
	for (; itsynaptic_indexes != synaptic_indexes.end(); itsynaptic_indexes++, itoutputweights++){
		if (*itsynaptic_indexes >= 0 && *itsynaptic_indexes < this->ninters){
			*itoutputweights = this->wordination[*itsynaptic_indexes]->GetWeight();
		}
	}
	return outputweights;
}


ostream & Network::PrintInfo(ostream & out) {
	int ind;

	out << "- Neuron types:" << endl;

	for(ind=0;ind<this->nneutypes;ind++){
		out << "\tType: " << ind << endl;

		this->neutypes[ind][0]->PrintInfo(out);
	}

	out << "- Neurons:" << endl;

   	for(ind=0;ind<this->nneurons;ind++){
		this->neurons[ind].PrintInfo(out);
	}

   	out << "- Weight change types:" << endl;

	for(ind=0;ind<this->nwchanges;ind++){
		out << "\tChange: " << ind << endl;
		this->wchanges[ind]->PrintInfo(out);
	}

	out << "- Interconnections:" << endl;

	for(ind=0; ind<this->ninters; ind++){
		out << "\tConnection: " << ind << endl;
		this->inters[ind].PrintInfo(out);
	}

	return out;
}

int Network::GetNumberOfQueues(){
	return NumberOfQueues;
}


double Network::GetMinInterpropagationTime(){
	double time = 10000000;
	for(int i=0; i<this->ninters; i++){
		if(inters[i].GetSource()->get_OpenMP_queue_index()!=inters[i].GetTarget()->get_OpenMP_queue_index() && inters[i].GetDelay()<time){
			time=inters[i].GetDelay();
		}
	}

	if(time == 10000000){
		time=0;
	}
	return time;
}


double Network::RoundPropagationDelay(double time){
	double result=floor(time*invminpropagationdelay + minpropagationdelay*0.5)*minpropagationdelay;
	if(result<=0){
		return minpropagationdelay;
	}
	return result;
}




void Network::CalculaElectricalCouplingDepedencies(){
	//Calculate the electrical coupling dependencies for each neuron model
	for (int i = 0; i < this->ninters; i++){
		(this->inters + i)->GetTarget()->GetNeuronModel()->CalculateElectricalCouplingSynapseNumber((this->inters + i));
	}

	for (int i = 0; i < this->GetNneutypes(); i++){
		for (int j = 0; j < this->GetNumberOfQueues(); j++){
			this->GetNeuronModelAt(i, j)->InitializeElectricalCouplingSynapseDependencies();
		}
	}
	for (int i = 0; i < this->ninters; i++){
		(this->inters + i)->GetTarget()->GetNeuronModel()->CalculateElectricalCouplingSynapseDependencies((this->inters + i));
	}
}
