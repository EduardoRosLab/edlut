/***************************************************************************
 *                           Network.h                                     *
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

#ifndef NETWORK_H_
#define NETWORK_H_

/*!
 * \file Network.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a spiking neural network.
 */

#include <string>
#include <cstdlib>
#include <cstring>

#include "./EDLUTFileException.h"

#include "../simulation/PrintableObject.h"

class Interconnection;
class NeuronModel;
class Neuron;
class WeightChange;
class EventQueue;

/*!
 * \class Network
 *
 * \brief Spiking neural network
 *
 * This class abstract the behaviour of a spiking neural network.
 * It is composed by interconnections, neurons, neuron types and weight changes.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class Network : public PrintableObject{
	private:
	
		/*!
		 * \brief Network interconnections.
		 */
		Interconnection *inters;
   
   		/*!
   		 * \brief Number of interconnections.
   		 */
   		long int ninters;
   
   		/*!
   		 * \brief Neuron types.
   		 */
   		NeuronModel ** neutypes;
   
   		/*!
   		 * \brief Neuron types number.
   		 */
   		int nneutypes;
   
   		/*!
   		 * \brief Neuron array.
   		 */
   		Neuron *neurons;
   
   		/*!
   		 * \brief Number of neurons.
   		 */
   		int nneurons;
   		
   		/*!
   		 * \brief Learning rules.
   		 */
   		WeightChange ** wchanges;
   
   		/*!
   		 * \brief Number of learning rules.
   		 */
   		int nwchanges;
   
   		/*!
   		 * \brief Initial connection ordenation.
   		 */
   		Interconnection ** wordination;
   		
   		/*!
   		 * \brief It sorts the connections by the source neuron and the delay and add the output connections
   		 * 
   		 * It sorts the connections by the source neuron (from the lowest to the highest index) and by the connection
   		 * delay. It adds the connections to the output connections of the source neuron.
   		 * 
   		 * \post The connections will be sorted by source neuron and delay.
   		 */
   		void FindOutConnections();
   		
   		/*!
   		 * \brief It adds the input connection to the target neuron.
   		 * 
   		 * It adds the connections to the input connections of the target neuron.
   		 */   		
   		void FindInConnections();
   		
   		/*!
   		 * \brief It sorts the connections by the connection index.
   		 * 
   		 * It sorts the connections by the connection index. This ordination
   		 * is the ordenation needed for the weight load. The ordination will be
   		 * in wordination field.
   		 */
   		void SetWeightOrdination();
   		
   		/*!
  		 * \brief It prints information about load tables.
  		 * 
  		 * It prints information about load tables.
  		 * 
  		 */
  		void TablesInfo();
  		
  		/*!
  		 * \brief It prints information about load types.
  		 * 
  		 * It prints information about load types.
  		 * 
  		 */
  		void TypesInfo();  		
  		
   		/*!
   		 * \brief It loads the neuron type characteristics.
   		 * 
   		 * It checks if the neuron type has been loaded, and in other case,
   		 * it loads the characteristics from the neuron type files.
   		 * 
   		 * \param ident_type Type of the neuron model. At this moment, only "SRMTimeDriven" and "TableBasedModel" are implemented.
   		 * \param neutype The name of the neuron type to load.
   		 * 
   		 * \return The loaded (or existing) neuron type.
   		 * \throw EDLUTException If the neuron model file hasn't been able to be correctly readed. 
   		 */
   		NeuronModel * LoadNetTypes(string ident_type, string neutype) throw (EDLUTException);
   		
   		/*!
   		 * \brief It inits the spikes predictions of every neuron in the network.
   		 * 
   		 * It adds all the spike predictions in the network in the initial conditions.
   		 * 
   		 * \param Queue The event queue where the spikes will be added.
   		 */
   		void InitNetPredictions(EventQueue * Queue);
   		
   		/*!
   		 * \brief It loads the network configuration from a file.
   		 * 
   		 * It loads a new network from a file.
   		 * 
   		 * \param netfile The file name of the network configuration file.
   		 * 
   		 * \throw EDLUTFileException If the network configuration file hasn't been able to be correctly readed.
   		 */
   		void LoadNet(const char *netfile) throw (EDLUTException);
   		
   		/*!
   		 * \brief It loads the connection synaptic weights from a file.
   		 * 
   		 * It loads the connection synaptic weights from a file.
   		 * 
   		 * \param wfile The file name of the weights file.
   		 * 
   		 * \pre The network connections are sorted by the index in the field wordenation.
   		 * 
   		 * \see SetWeightOrdenation()
   		 * 
   		 * \throw EDLUTFileException If the weights file hasn't been able to be correctly readed.
   		 */
   		void LoadWeights(const char *wfile) throw (EDLUTFileException);
   		   		
   	public:
   	
   		/*!
   		 * \brief It creates a new network object by loading the configuration and the
   		 * weights from files.
   		 * 
   		 * It creates a new network object. The network is loaded from the configuration file,
   		 * and the weights are loaded from his file. Finally, it initializes the event queue
   		 * with the initial spikes.
   		 * 
   		 * \param netfile The network configuration file name.
   		 * \param wfile The weight file name.
   		 * \param Queue The event queue where the events will be inserted.
   		 * 
   		 * \throw EDLUTException If some error has happened.
   		 */
   		Network(const char * netfile, const char * wfile, EventQueue * Queue) throw (EDLUTException);
   		
   		/*!
   		 * \brief Default destructor.
   		 * 
   		 * It destroies a network object and frees the memory.
   		 */
   		~Network();
   		
   		/*!
   		 * \brief It gets a neuron by the index.
   		 * 
   		 * It returns a neuron from the index.
   		 * 
   		 * \param index The index of the neuron to get.
   		 * 
   		 * \return The neuron whose index is the parameter.
   		 */
   		Neuron * GetNeuronAt(int index) const;
   		
   		/*!
   		 * \brief It gets the number of neurons in the network.
   		 * 
   		 * It gets the number of neurons in the network.
   		 * 
   		 * \return The number of neurons.
   		 */
   		int GetNeuronNumber() const;
   		
   		/*!
   		 * \brief It saves the weights in a file.
   		 * 
   		 * It saves the network weights in a file.
   		 * 
   		 * \param wfile The file name where we save the weights.
   		 * 
   		 * \throw EDLUTException If some error happends.
   		 */
   		void SaveWeights(const char *wfile) throw (EDLUTException);
   		
   		/*!
   		 * \brief It prints the network info.
   		 * 
   		 * It prints the current network characteristics (number of neurons, types, connections...).
   		 * 
   		 * \param out The stream where it prints the information.
   		 * 
   		 * \return The stream after the printer.
   		 */
   		virtual ostream & PrintInfo(ostream & out);
   		
};

/*!
 * \brief It sorts two connections by the source neuron and the delay.
 * 
 * This functions sorts two connections by the source neuron and the delay.
 * 
 * \param e1 The firs connection.
 * \param e2 The second connection.
 * 
 * \return 0 if the two connections have the same index of the source neuron and the same delay. <0 if
 * the second connection have a higher index of the source neuron or the same index and lower delay.
 * >0 if the first connection have a higher index of the source neuron or the same index and lower delay.
 */
int qsort_inters(const void *e1, const void *e2);


#endif /*NETWORK_H_*/
