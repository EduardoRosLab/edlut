/***************************************************************************
 *                           TimeDrivenPropagatedSpike.h                   *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TIMEDRIVENPROPAGATEDSPIKE_H_
#define TIMEDRIVENPROPAGATEDSPIKE_H_

/*!
 * \file PropagatedSpike.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a neural network propagated spike.
 */
 
#include <iostream>

#include "./Spike.h"

using namespace std;

class Neuron;
class Interconnection;

/*!
 * \class PropagatedSpike
 *
 * \brief Neural network propagated spike.
 *
 * A propagated spike is an event which generates a new spike in the next output connection.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class TimeDrivenPropagatedSpike: public Spike{
	
	protected: 

		const int MaxSize;
		int N_Elements;
		int * N_ConnectionsWithEqualDelay;
		Interconnection ** ConnectionsWithEqualDelay;


   		/*!
   		 * It say in which OpenMP queue is the target neuron of this propagated spike.
   		 */
		const int OpenMP_index;

   		
   	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new spike object.
		 * 
		 * \param NewOpenMP_index queue index of the target neuron.
		 */
   		TimeDrivenPropagatedSpike(double NewTime, int NewOpenMP_index, int NewMaxSize);
   	
    		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~TimeDrivenPropagatedSpike();

		int GetN_Elementes();
		int GetMaxSize();

		bool IncludeNewSource(int NewN_Connections, Interconnection * NewConnections);
   	
 		

   		/*!
   		 * \brief It process an event in the simulation with the option of real time available.
   		 * 
   		 * It process an event in the simulation with the option of real time available.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
		 * \param RealTimeRestriction watchdog variable executed in a parallel OpenMP thread that
		 * control the consumed time in each slot.
   		 */
   		virtual void ProcessEvent(Simulation * CurrentSimulation,  int RealTimeRestriction);

		/*!
   		 * \brief It process an event in the simulation without the option of real time available.
   		 * 
   		 * It process an event in the simulation without the option of real time available.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
   		 */
		virtual void ProcessEvent(Simulation * CurrentSimulation);


   		/*!
   		 * \brief It return the OpenMP_index.
   		 * 
   		 * It return the OpenMP_index.
		 * 
   		 * \return the OpenMP_index.
   		 */
		int GetOpenMP_index() const;

   		/*!
   		 * \brief this method print the event type.
   		 * 
   		 * This method print the event type..
		 */
		virtual void PrintType();
   		
};

#endif /*TIMEDRIVENPROPAGATEDSPIKE_H_*/
