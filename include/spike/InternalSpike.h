/***************************************************************************
 *                           InternalSpike.h                               *
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

#ifndef INTERNALSPIKE_H_
#define INTERNALSPIKE_H_

/*!
 * \file InternalSpike.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a neural network internal spike.
 */
 
#include <iostream>

#include "./Spike.h"

using namespace std;

class Neuron;
class Simulation;

/*!
 * \class InternalSpike
 *
 * \brief Neural network internal spike.
 *
 * This class abstract the concept of spike. An internal spike is an event generated by the own cell
 * without external excitation.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date November 2008
 */
class InternalSpike: public Spike{
	
	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new spike object.
   		 */
   		InternalSpike();
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new spike with the parameters.
   		 * 
   		 * \param NewTime Time of the new spike.
   		 * \param NewSource Source neuron of the spike.
   		 */
   		InternalSpike(double NewTime, Neuron * NewSource);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~InternalSpike();
   	

		/*!
   		 * \brief It process an event in the simulation.
   		 * 
   		 * It process the event in the simulation.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
		 * \param RealTimeRestriction This variable indicates whether we are making a 
		 * real-time simulation and the watchdog is enabled.
   		 */
   		void ProcessEvent(Simulation * CurrentSimulation, bool RealTimeRestriction);
   		 
   		
};

#endif /*SPIKE_H_*/
