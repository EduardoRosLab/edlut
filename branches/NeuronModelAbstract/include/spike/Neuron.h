/***************************************************************************
 *                           Neuron.h                                      *
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

#ifndef NEURON_H_
#define NEURON_H_

/*!
 * \file Neuron.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a spiking neuron behaviour.
 */
#include <math.h>
#include <vector>

#include "../simulation/Configuration.h"
 
using namespace std;

class NeuronModel;
class NeuronState;
class Interconnection;
class EventQueue;
class InternalSpike;
class PropagatedSpike;
class Spike;


/*!
 * \class Neuron
 *
 * \brief Spiking neuron
 *
 * This class abstract the behaviour of a neuron in a spiking neural network.
 * It includes network characteristics as index in the network or input and output 
 * connections and spiking characteristics as the neural model, the current state,
 * the last updated time, or the spike prediction.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class Neuron{
	private:
	
		/*!
		 * \brief Neuron associated model.
		 */
		NeuronModel * type;
		
		/*!
		 * \brief Neuron index into the network neurons.
		 */
		long int index;
   
   		/*!
   		 * \brief Neuron state variables.
   		 */
   		//float statevars[MAXSTATEVARS];
   		NeuronState * state;
   		
   		/*!
   		 * Output connections.
   		 */
   		vector<Interconnection*> OutputConnections;
   		
   		/*!
   		 * Input connections.
   		 */
   		vector<Interconnection*> InputConnections;
   		
   		/*!
   		 * It tells if neuron activity will be registered.
   		 */
   		bool monitored;
   		
   		/*!
   		 * It tells if neuron is output neuron
   		 */
   		bool isOutput;
   		
   	public:
   		/*!
		 * \brief Default constructor without parameters.
		 *
		 * It generates a new default neuron object without input connections, output connections
		 * or neuron model. The neuron will be initialized with the default values.
		 */
   		Neuron();

		/*!
		 * \brief Neuron constructor with parameters.
		 *
		 * It generates a new neuron with neuron model Type and neuron index NewIndex.
		 * Moreover, it initializes the neuron variables with the model initial values.
		 *
		 * \param NewIndex The neuron index into the network order.
		 * \param Type The neuron type. It can't be null.
		 * \param Monitored If true, the neuron activity will be registered.
		 * \param IsOutput If true, the neuron activity will be send to output driver
		 * \sa InitNeuron()
		 */   	
   		Neuron(int NewIndex, NeuronModel * Type, bool Monitored, bool IsOutput);
   		
   		/*!
		 * \brief It initializes the neuron values.
		 *
		 * It initializes a neuron object with neuron model Type and neuron index NewIndex.
		 * Moreover, it initializes the neuron variables with the model initial values.
		 *
		 * \param NewIndex The neuron index into the network order.
		 * \param Type The neuron type. It can't be null.
		 * \param Monitored If true, the neuron activity will be registered.
		 * \param IsOutput If true, the neuron activity will be send to output driver
		 */
   		void InitNeuron(int NewIndex, NeuronModel * Type, bool Monitored, bool IsOutput);
   		
   		/*!
		 * \brief It gets the neuron index into the network.
		 * 
		 * It returns the saved neuron index into the network's neurons.
		 * 
		 * \return The neuron index.
		 */
		long int GetIndex() const;
   		
   		/*!
   		 * \brief It gets the current neuron state.
   		 * 
   		 * It gets the current neuron state.
   		 * 
   		 * \return The current neuron state.
   		 */
   		NeuronState * GetNeuronState() const;
   		
   		/*!
   		 * \brief It gets the number of inputs to the current neuron.
   		 * 
   		 * It returns the number of input connections to the current neuron.
   		 * 
   		 * \return The number of input connections to the current neuron.
   		 */
   		int GetInputNumber() const;
   		
   		/*!
   		 * \brief It gets the number of output from the current neuron.
   		 * 
   		 * It returns the number of output connections from the current neuron.
   		 * 
   		 * \return The number of output connections from the current neuron.
   		 */
   		int GetOutputNumber() const;
   		
   		/*!
   		 * \brief It gets the input connection at an specified index.
   		 * 
   		 * It returns the input connection at index index.
   		 * 
   		 * \param index The index of the input connection what we want to get.
   		 * \return The input connection of index index.
   		 */
   		Interconnection * GetInputConnectionAt(int index) const;
   		
   		/*!
   		 * \brief It adds an input connection to the neuron.
   		 * 
   		 * It adds the parameter connection as an input connection to the neuron.
   		 * 
   		 * \param Connection The input connection to add.
   		 */
   		void AddInputConnection(Interconnection * Connection);
   		
   		/*!
   		 * \brief It checks if the neuron has some input connection.
   		 * 
   		 * It checks if the number of input connections is greater than 0.
   		 * 
   		 * \return True if the neuron has some input connection. False in other case.
   		 */
   		bool IsInputConnected() const;
   		
   		/*!
   		 * \brief It gets the output connection at an specified index.
   		 * 
   		 * It returns the output connection at index index.
   		 * 
   		 * \param index The index of the output connection what we want to get.
   		 * \return The output connection of index index.
   		 */
   		Interconnection * GetOutputConnectionAt(int index) const;
   		
   		/*!
   		 * \brief It adds an output connection to the neuron.
   		 * 
   		 * It adds the parameter connection as an output connection to the neuron.
   		 * 
   		 * \param Connection The output connection to add.
   		 */
   		void AddOutputConnection(Interconnection * Connection);
   		
   		/*!
   		 * \brief It checks if the neuron has some output connection.
   		 * 
   		 * It checks if the number of output connections is greater than 0.
   		 * 
   		 * \return True if the neuron has some output connection. False in other case.
   		 */
   		bool IsOutputConnected() const;
   		
   		/*!
   		 * \brief It checks if the neuron is monitored.
   		 * 
   		 * It checks if the neuron activity will be registered.
   		 * 
   		 * \return True if the neuron is monitored. False in other case.
   		 */
		bool IsMonitored() const;
		
		/*!
   		 * \brief It checks if the neuron is output.
   		 * 
   		 * It checks if the neuron activity is output.
   		 * 
   		 * \return True if the neuron is output. False in other case.
   		 */
		bool IsOutput() const;
		
		/*!
   		 * \brief It gets the neuron type.
   		 * 
   		 * It returns the neuron type of the current object.
   		 * 
   		 * \return The neuron type of the current object.
   		 */
		NeuronModel * GetNeuronModel() const;
   		  		
};
  
#endif /*NEURON_H_*/
