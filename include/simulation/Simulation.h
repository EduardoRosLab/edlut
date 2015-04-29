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
#include "./RealTimeRestriction.h"

#include "./PrintableObject.h"

/*!
 * This constant defines how many events are processed before the RunSimulationSlot method
 * checks that the specified MaxSlotConsumedTime is not violated. A higher number increases
 * the computation overhead but reduces the excess time that this method execution could
 * consume.
 */
#define NUM_EVENTS_PER_TIME_SLOT_CHECK 5UL

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
		 * Vector of event queue used in the events for each OpenMP thread.
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
		 * Vector or current simulation Time for each OpenMP thread
		 */
		double * CurrentSimulationTime;
		
		
		/*!
		 * Vector of end of simulation for each OpenMP thread
		 */
		bool * EndOfSimulation;
		
		/*!
		 * Vector of stop of simulation for each OpenMP thread
		 */
		bool * StopOfSimulation;

		/*!
		 * Vector of syncronize events of simulation for each OpenMP thread
		 */
		bool * SynchronizeThread;

		/*!
		 * Vector of number of realized updates for each OpenMP thread
		 */
		long long * Updates;
		
		/*!
		 * Vector of sumatory of heap size for each OpenMP thread
		 */
		long long * Heapoc;

		/*!
		 * Vector of total spike count for each OpenMP thread
		 */
		long long * TotalSpikeCounter;

		/*!
		 * Vector of total propated spike count for each OpenMP thread
		 */
		long long * TotalPropagateCounter;

		/*!
		 * Vector of total propated spike event count for each OpenMP thread
		 */
		long long * TotalPropagateEventCounter;

		/*!
 		 * Number of OpenMP thread. 
 		 */
		int NumberOfThreads;

		/*!
 		 * Number of OpenMP queues. 
 		 */
		int NumberOfQueues;

		/*!
 		 * Min interpropagation time between neuron of differents OpenMP queues. This value is used
		 * to create the SynchronizeActivityEvent that insert the buffer of events inside each queue.
 		 */
		double MinInterpropagationTime;

	public:

		/*!
 		 * Real time restriction object used to control a real robot. 
 		 */
		RealTimeRestriction * RealTimeRestrictionObject;

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
		Simulation(const char * NetworkFile, const char * WeightsFile, double SimulationTime, double NewSimulationStep=0.00, int NewNumberOfQueues=1, int NewNumberOfThreads=1) throw (EDLUTException);
		
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
		 * \brief It ends the simulation before the next event for one OpenMP thread.
		 * 
		 * It ends the simulation before the next event for one OpenMP thread.
		 * 
		 * \param indexThread index of the OpenMP thread.
		 */
		void EndSimulation(int indexThread);

		/*!
		 * \brief It stops the simulation before the next event for one OpenMP thread.
		 *
		 * It stops the simulation before the next event for one OpenMP thread.
		 		 * 
		 * \param indexThread index of the OpenMP thread.
		 */
		void StopSimulation(int indexThread);

		void SetSynchronizeSimulationEvent(int indexThread);

		void ResetSynchronizeSimulationEvent(int indexThread);

		bool GetSynchronizeSimulationEvent(int indexThread);

				 
		void SynchronizeThreads();

		void SynchronizeThreadsForSlot();

		/*!
		 * \brief It adds a new input driver to the input driver list.
		 * 
		 * It adds a new input driver to the input driver list.
		 * 
		 * \param NewInput The input driver to add.
		 */
		 void AddInputSpikeDriver(InputSpikeDriver * NewInput);
		 
		/*!
		 * \brief It return the specified element of the InputSpikeDriver list.
		 * 
		 * It returns the InputSpikeDriver pointer in the specified position of the Simulation's
		 * InputSpikeDriver list.
		 * 
		 * \param ElementPosition The position of the list. First element is in position 0.
		 * \return The InputSpikeDriver in the specified position of the list.
		 */
		 InputSpikeDriver *GetInputSpikeDriver(unsigned int ElementPosition);
		 
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
		 * \brief It return the specified element of the OutputSpikeDriver list.
		 * 
		 * It returns the OutputSpikeDriver pointer in the specified position of
		 * the Simulation's InputSpikeDriver list.
		 * 
		 * \param ElementPosition The position of the list. First element is in position 0.
		 * \return The OutputSpikeDriver in the specified position of the list.
		 */
		 OutputSpikeDriver *GetOutputSpikeDriver(unsigned int ElementPosition);

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
		 * \brief It return the specified element of the MonitorActivityDriver list.
		 * 
		 * It returns the OutputSpikeDriver pointer in the specified position of
		 * the Simulation's MonitorActivityDriver list.
		 * 
		 * \param ElementPosition The position of the list. First element is in position 0.
		 * \return The OutputSpikeDriver in the specified position of the list.
		 */
		 OutputSpikeDriver *GetMonitorActivityDriver(unsigned int ElementPosition);

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
		 * \brief It return the specified element of the OutputWeightDriver list.
		 * 
		 * It returns the OutputWeightDriver pointer in the specified position of
		 * the Simulation's OutputWeightDriver list.
		 * 
		 * \param ElementPosition The position of the list. First element is in position 0.
		 * \return The OutputWeightDriver in the specified position of the list.
		 */
		 OutputWeightDriver *GetOutputWeightDriver(unsigned int ElementPosition);

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
		void SetTotalSpikeCounter(int indexThread, long long value);

		/*!
		 * \brief Set total spike count.
		 *
		 * Sets the total spike counter.
		 * \param value is value to set the counter to.
		 */
		void IncrementTotalSpikeCounter(int indexThread);

		/*!
		 * \brief Get total spike count.
		 *
		 * Gets the total spike counter.
		 * \returns the counter value.
		 */
		long long GetTotalSpikeCounter(int indexThread);

		long long GetTotalSpikeCounter();

		long long GetTotalPropagateCounter(int indexThread);

		long long GetTotalPropagateCounter();

		long long GetTotalPropagateEventCounter(int indexThread);

		long long GetTotalPropagateEventCounter();

		void SetTotalPropagateCounter(int indexThread, long long value); 

		void SetTotalPropagateEventCounter(int indexThread, long long value); 

		void IncrementTotalPropagateCounter(int indexThread); 

		void IncrementTotalPropagateCounter(int indexThread, int increment); 

		void IncrementTotalPropagateEventCounter(int indexThread); 

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
		long long GetSimulationUpdates(int indexThread) const;

		long long GetSimulationUpdates() const;
		
		/*!
		 * \brief It gets the sumatory of event queue sizes.
		 * 
		 * It gets the sumatory of event queue sizes.
		 * 
		 * \return The number of acumulated event queue sizes.
		 */
		long long GetHeapAcumSize(int indexThread) const;

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

		/*!
		 * \brief It gets the number of OpenMP thread.
		 * 
		 * It gets the number of OpenMP thread.
		 * 
		 * \return The number of OpenMP thread.
		 */
		int GetNumberOfThreads();

		int GetNumberOfQueues();

		double GetMinInterpropagationTime();
	

		void SelectGPU(int OpenMP_index);
};

#endif /*SIMULATION_H_*/
