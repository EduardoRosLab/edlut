/***************************************************************************
 *                           Simulation.h                                  *
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

#ifndef SIMULATION_H_
#define SIMULATION_H_

/*!
 * \file Simulation.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a simulation of an spiking neural network.
 */
 
#include <list>

#include "../spike/EDLUTException.h"

#include "./PrintableObject.h"

class Network;
class EventQueue;
class InputSpikeDriver;
class OutputSpikeDriver;
class OutputWeightDriver;
class Spike;
class Neuron;


/*!
 * \class Simulation
 *
 * \brief Spiking neural network simulation.
 *
 * This class abstract the behaviour of a spiking neural network simulation. It loads the network,
 * gets the input spikes and finally produces the output spikes. Moreover, this class generates some
 * simulation statistics.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class Simulation : public PrintableObject{
	
	private:
		/*!
		 * Neural network to simulate.
		 */
		Network * Net;
		
		/*!
		 * Event queue used in the events.
		 */
		EventQueue * Queue;
		
		/*!
		 * Input of activity.
		 */
		list<InputSpikeDriver *> InputSpike;
		
		/*!
		 * Output of activity.
		 */
		list<OutputSpikeDriver *> OutputSpike;
		
		/*!
		 * Output of activity.
		 */
		list<OutputWeightDriver *> OutputWeight;
		
		/*!
		 * Monitor of activity.
		 */
		list<OutputSpikeDriver *> MonitorSpike;
		
		/*!
		 * Total time of simulation.
		 */
		double Totsimtime;
		
		/*!
		 * Simulation step
		 */
		double SimulationStep;
		
		/*!
		 * Save weight step
		 */
		double SaveWeightStep;
		
		/*!
		 * Current simulation Time
		 */
		double CurrentSimulationTime;
		
		/*!
		 * End of simulation
		 */
		bool EndOfSimulation;
		
		/*!
		 * Stop of simulation
		 */
		bool StopOfSimulation;

		/*!
		 * Number of realized updates.
		 */
		long long Updates;
		
		/*!
		 * Sumatory of heap size.
		 */
		long long Heapoc;

		/*!
		 * Total spike count.
		 */
		long TotalSpikeCounter;

	protected:
		
		/*!
		 * \brief It realizes the main process of simulation.
		 * 
		 * It propagates the spikes after the input is loaded. This propagation ends when the
		 * simulation time is higher than the total simulation time or when no more activity is
		 * produced.
		 * 
		 * \pre The activity has been previously loaded.
		 * 
		 * \throw EDLUTException If something wrong happends in the spike propagation process.
		 */
		void RunSimulationStep() throw (EDLUTException);
				
	public:
	
		/*Simulation(Network * Net, InputDriver * NewInput, OutputDriver * NewOutput, long SimulationTime);*/
		
		/*!
		 * \brief Class constructor with parameters.
		 * 
		 * It creates a new object an initilizes its elements.
		 * 
		 * \param NetworkFile Network description file name. The network will be loaded from this file.
		 * \param WeightsFile Weights description file name. The network synaptic weights will be loaded from this file.
		 * \param SimulationTime Simulation total time.
		 * \param NewSimulationStep Simulation step time.
		 * 
		 * throw EDLUTException If something wrong happens.
		 */
		Simulation(const char * NetworkFile, const char * WeightsFile, double SimulationTime, double NewSimulationStep=0.00) throw (EDLUTException);
		
		/*!
		 * \brief Copy constructor of the class.
		 * 
		 * It creates a new object copied of the parameter.
		 * 
		 * \param ant Current simulation object (copy source).
		 */
		Simulation(const Simulation & ant);
		
		/*!
		 * \brief Object destructor.
		 * 
		 * It destroys an object of Simulation class.
		 */
		~Simulation();
		
		/*!
		 * \brief It sets the saving weights step time.
		 * 
		 * It sets the saving weights step time.
		 * 
		 * \param NewSaveStep The saving step time (in seconds). 0 values don't save the weights.
		 */
		void SetSaveStep(float NewSaveStep);
		
		/*!
		 * \brief It gets the saving weights step time.
		 * 
		 * It gets the saving weights step time.
		 * 
		 * \return The saving step time (in seconds). 0 values don't save the weights.
		 */
		double GetSaveStep();
		
		/*!
		 * \brief It sets the simulation step time.
		 * 
		 * It sets the saving step time.
		 * 
		 * \param NewSimulationStep The simulation step time (in seconds). 0 values don't simulate by steps.
		 */
		void SetSimulationStep(double NewSimulationStep);
		
		/*!
		 * \brief It gets the simulation step time.
		 * 
		 * It gets the simulation step time.
		 * 
		 * \return The simulation step time (in seconds). 0 values don't simulate by step.
		 */
		double GetSimulationStep();
		
		/*!
		 * \brief It ends the simulation before the next event.
		 * 
		 * It ends the simulation before the next event.
		 */
		void EndSimulation();

		/*!
		 * \brief It stops the simulation before the next event.
		 *
		 * It stops the simulation before the next event.
		 */
		void StopSimulation();
				 
				
		/*!
		 * \brief It adds a new input driver to the input driver list.
		 * 
		 * It adds a new input driver to the input driver list.
		 * 
		 * \param NewInput The input driver to add.
		 */
		 void AddInputSpikeDriver(InputSpikeDriver * NewInput);
		 
		/*!
		 * \brief It removes an input driver of the input driver list.
		 * 
		 * It removes an input driver of the input driver list.
		 * 
		 * \param NewInput The input driver to remove.
		 */
		 void RemoveInputSpikeDriver(InputSpikeDriver * NewInput);
		 
		/*!
		 * \brief It adds a new output driver to the output driver list.
		 * 
		 * It adds a new output driver to the output driver list.
		 * 
		 * \param NewOutput The Output driver to add.
		 */
		 void AddOutputSpikeDriver(OutputSpikeDriver * NewOutput);
		
		/*!
		 * \brief It removes an output driver of the output driver list.
		 * 
		 * It removes an output driver of the output driver list.
		 * 
		 * \param NewOutput The output driver to remove.
		 */
		 void RemoveOutputSpikeDriver(OutputSpikeDriver * NewOutput);
		 
		 /*!
		 * \brief It adds a new output driver to the monitor activity list.
		 * 
		 * It adds a new output driver to the monitor activity list.
		 * 
		 * \param NewMonitor The Output driver to add.
		 */
		 void AddMonitorActivityDriver(OutputSpikeDriver * NewMonitor);
		
		/*!
		 * \brief It removes an output driver of the output monitor list.
		 * 
		 * It removes an output driver of the output monitor list.
		 * 
		 * \param NewMonitor The output driver to remove.
		 */
		 void RemoveMonitorActivityDriver(OutputSpikeDriver * NewMonitor);
		 
		 /*!
		 * \brief It adds a new output weight driver to the output driver list.
		 * 
		 * It adds a new output driver to the output driver list.
		 * 
		 * \param NewOutput The Output driver to add.
		 */
		 void AddOutputWeightDriver(OutputWeightDriver * NewOutput);
		
		/*!
		 * \brief It removes an output weight driver of the output driver list.
		 * 
		 * It removes an output driver of the output driver list.
		 * 
		 * \param NewOutput The output driver to remove.
		 */
		 void RemoveOutputWeightDriver(OutputWeightDriver * NewOutput);
		
		/*!
		 * \brief It loads the simulation.
		 *
		 * It loads the simulation (inputs from file, saving weight events, communication events...).
		 *
		 * \throw EDLUTException If something wrong happens.
		 */
		void InitSimulation() throw (EDLUTException);

		/*!
		 * \brief It runs the simulation.
		 * 
		 * It runs the simulation.
		 * 
		 * \throw EDLUTException If something wrong happens.
		 */
		void RunSimulation() throw (EDLUTException);
		
		/*!
		 * \brief It runs the simulation until simulation time > preempt_time.
		 *
		 * It runs the simulation until simulation time > preempt_time.
		 *
		 * \throw EDLUTException If something wrong happens.
		 *
		 * \param preempt_time is the ending time of the time slot.
		 */
		void RunSimulationSlot(double preempt_time) throw (EDLUTException);


		/*!
		 * \brief It writes a spike in the activity log and in the outputs.
		 * 
		 * It writes a spike in the activity log and in the outputs.
		 * 
		 * \param spike The spike to be wrotten.
		 */
		void WriteSpike(const Spike * spike);
		
		/*!
		 * \brief It writes a neuron potential.
		 * 
		 * It writes a neuron potential.
		 * 
		 * \param time The event time.
		 * \param neuron The neuron of the event.
		 */
		void WriteState(float time, Neuron * neuron);
		
		/*!
		 * \brief It saves the current synaptic weights in the output weights driver.
		 * 
		 * It saves the current synaptic weights in the output weights driver.
		 */
		void SaveWeights();
		
		/*!
		 * \brief It sends the happened events.
		 * 
		 * It sends the happened events.
		 */
		void SendOutput();
		
		/*!
		 * \brief Set total spike count.
		 *
		 * Sets the total spike counter.
		 * \param value is value to set the counter to.
		 */
		void SetTotalSpikeCounter(int value);

		/*!
		 * \brief Get total spike count.
		 *
		 * Gets the total spike counter.
		 * \returns the counter value.
		 */
		long GetTotalSpikeCounter();

		/*!
		 * \brief It gets the input activity.
		 * 
		 * It gets the input activity.
		 */		
		void GetInput();
		
		/*!
		 * \brief It gets the simulated network.
		 * 
		 * It gets the simulated network.
		 * 
		 * \return The simulated network.
		 */
		Network * GetNetwork() const;
		
		/*!
		 * \brief It gets the event queue.
		 * 
		 * It gets the event queue.
		 * 
		 * \return The simulated network.
		 */
		EventQueue * GetQueue() const;
		
		/*!
		 * \brief It gets the stablished simulation time.
		 * 
		 * It gets the total simulation time.
		 * 
		 * \return The total simulation time.
		 */
		double GetTotalSimulationTime() const;
		
		/*!
		 * \brief It gets the number of simulation updates realized.
		 * 
		 * It gets the number of simulation updates realized.
		 * 
		 * \return The number of simulation updates.
		 */
		long long GetSimulationUpdates() const;
		
		/*!
		 * \brief It gets the sumatory of event queue sizes.
		 * 
		 * It gets the sumatory of event queue sizes.
		 * 
		 * \return The number of acumulated event queue sizes.
		 */
		long long GetHeapAcumSize() const;		

		/*!
		 * \brief It prints the information of the object.
		 *
		 * It prints the information of the object.
		 *
		 * \param out The output stream where it prints the object to.
		 * \return The output stream.
		 */
		virtual ostream & PrintInfo(ostream & out);
	
};

#endif /*SIMULATION_H_*/
