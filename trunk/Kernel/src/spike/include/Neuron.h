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

#include "../../simulation/include/Configuration.h"
 
using namespace std;

class NeuronType;
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
		NeuronType * type;
		
		/*!
		 * \brief Neuron index into the network neurons.
		 */
		long int index;
   
   		/*!
   		 * \brief Neuron state variables.
   		 */
   		float statevars[MAXSTATEVARS];
   		
   		/*!
   		 * \brief Last update time
   		 */
   		double lastupdate;
   		
   		/*!
   		 * Next spike predicted time.
   		 */
   		double predictedspike;
   		
   		/*!
   		 * End of the event prediction.
   		 */
   		double predictionend;
   		
   		
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
   		
   		/*!
		 * \brief It updates the neuron state.
		 *
		 * It updates the neuron state variables for the elapsed time 
		 *
		 * \param ElapsedTime Elapsed time since the time of the last update.
		 */
   		void NeuronUpdate(double ElapsedTime);
   		
   		/*!
   		 * \brief It calculates the effect of an input spike in the neuron.
   		 * 
   		 * It calculates the effect of an input spike belong the interconnection inter.
   		 * It updates the neuron membrane potential according the connection synaptic weight.
   		 * 
   		 * \param inter Input connection where the spike fire.
   		 * \pre inter must be an input connection to the current neuron.
   		 */
   		void SynapsisEffect(Interconnection *inter);
   		
   		/*!
   		 * \brief It calculates the spike firing time.
   		 * 
   		 * It calculates the time to the next spike fire with the current neural state.
   		 * It queries the firing prediction table.
   		 * 
   		 * \return The time to the next fire. -1.0 if spike won't be fired with the current state.
   		 */   		 
   		double FiringPrediction();
   		
   		/*!
   		 * \brief It calculates the time when the next spike will finish.
   		 * 
   		 * It calculates the time to the next spike finish with the current neural state.
   		 * It queries the firing end prediction table.
   		 * 
   		 * \return The time to the next fire ends.
   		 */
   		double FiringEndPrediction();
   		
   		
   		
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
   		Neuron(int NewIndex, NeuronType * Type, bool Monitored, bool IsOutput);
   		
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
   		void InitNeuron(int NewIndex, NeuronType * Type, bool Monitored, bool IsOutput);
   		
   		/*!
		 * \brief It generates the initial neuron activity.
		 *
		 * It generates the initial events in the neuron. It gets the time when the
		 * first spike will be fired (without external activity) and inserts a new event
		 * in SpikeQueue. 
		 *
		 * \param eventQueue The event queue where the event will be inserted.
		 */
   		void InitNeuronPrediction(EventQueue * eventQueue);
   		
   		/*!
   		 * \brief It generates the next spike in the neuron without input activity.
   		 * 
   		 * It checks if the neuron will fire a new spike without input activity and
   		 * inserts its in the spike queue.
   		 * 
   		 * \note It's only necessary when the neuron can fire periodically without input activity.
   		 * 
   		 * \param eventQueue The event queue where the event will be inserted.
   		 */
   		void GenerateAutoActivity(EventQueue * eventQueue);
   		
   		/*!
   		 * \brief It updates the neural state after an input spike.
   		 * 
   		 * It updates the neural state variables after an input spike. 
   		 * 
   		 * \param InputSpike Input spike who arrives the current object.
   		 * \pre InputSpike.GetSource() must be this.
   		 * 
   		 * \note It's only necessary when InputSpike.GetTarget()==-2
   		 */
   		void ProcessInputActivity(InternalSpike * InputSpike);
   		
   		/*!
   		 * \brief It updates the neural state after a synaptic input spike (target neuron).
   		 * 
   		 * It updates the neural state variables of a target neuron after an spike. 
   		 * 
   		 * \param InputSpike Input spike who arrives the current object.
   		 * \pre InputSpike.GetTarget() must be greater or equal than 0.
   		 */
   		void ProcessInputSynapticActivity(PropagatedSpike * InputSpike);
   		
   		/*!
   		 * \brief It generates the first output spike.
   		 * 
   		 * It produces the first output spike in the first output connection of the current neuron.
   		 * This spike will be propagated.
   		 * 
   		 * \param InputSpike Current spike which generates the activity.
   		 * \param eventQueue The event queue where the event will be inserted.
   		 * \note This function will be called when we process a spike with target lower than 0.
   		 */
   		void GenerateOutputActivity(Spike * InputSpike, EventQueue * eventQueue);
   		
   		/*!
   		 * \brief It propagates an output spike.
   		 * 
   		 * It produces the next output spike in the next output connection.
   		 * 
   		 * \param LastSpike The last processed spike.
   		 * \pre LastSpike.GetSource() must be equal to this.
   		 * \param eventQueue The event queue where the event will be inserted.
   		 * \note This function will be called when we process a spike with target greater or equal than 0.
   		 */
   		void PropagateOutputSpike(PropagatedSpike * LastSpike, EventQueue * eventQueue);
   		
   		/*!
   		 * \brief It checks if a new spike will be fired.
   		 * 
   		 * It updates the neuron next spike time and inserts the spike in the event queue.
   		 * Moreover, it checks if the neuron generates autoactivity.
   		 * 
   		 * \param eventQueue The event queue where the event will be inserted.
   		 * \note This function will be called after we process a propagated spike.
   		 */
   		void GenerateInputActivity(EventQueue * eventQueue);
		
		/*!
		 * \brief It gets the neuron index into the network.
		 * 
		 * It returns the saved neuron index into the network's neurons.
		 * 
		 * \return The neuron index.
		 */
		long int GetIndex() const;
   		
   		/*!
   		 * \brief It gets the last update time.
   		 * 
   		 * It returns the last update time in that neuron.
   		 * 
   		 * \return The last update time.
   		 */
   		double GetLastUpdate() const;
   		
   		/*!
   		 * \brief It gets the time of the next predicted spike.
   		 * 
   		 * It returns the time of the next predicted spike.
   		 * 
   		 * \return The time of the next predicted spike.
   		 */
   		double GetPredictedSpike() const;
   		
   		/*!
   		 * \brief It gets the time of the end of the next predicted spike.
   		 * 
   		 * It returns the time of the end of the next predicted spike.
   		 * 
   		 * \return The time of the end of the next predicted spike.
   		 */
   		double GetPredictionEnd() const;
   		
   		//void SetPredictionEnd(float PredictionEnd);
   		
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
   		 * \brief It gets the state variable at an specified index.
   		 * 
   		 * It returns the state variable at index index.
   		 * 
   		 * \param index The index of the state variable what we want to get.
   		 * \return The state variable of index index.
   		 */
   		float GetStateVarAt(int index) const;
   		
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
		NeuronType * GetNeuronType() const;
   		  		
};
  
#endif /*NEURON_H_*/
