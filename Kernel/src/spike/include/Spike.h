#ifndef SPIKE_H_
#define SPIKE_H_

/*!
 * \file Spike.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a neural network spike.
 */
 
#include <iostream>

#include "../../simulation/include/Event.h"

using namespace std;

class Neuron;
class Simulation;

/*!
 * \class Spike
 *
 * \brief Neural network spike.
 *
 * This class abstract the concept of spike. A spike is an event which generates a neuron
 * state update.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class Spike: public Event{
	
	protected: 
   		/*!
   		 * Source neuron of the spike.
   		 */
   		Neuron * source;
     		
   	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new spike object.
   		 */
   		Spike();
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new spike with the parameters.
   		 * 
   		 * \param NewTime Time of the new spike.
   		 * \param NewSource Source neuron of the spike.
   		 */
   		Spike(double NewTime, Neuron * NewSource);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~Spike();
   	
   		/*!
   		 * \brief It gets the spike source neuron.
   		 * 
   		 * It gets the spike source neuron.
   		 * 
   		 * \return The spike source neuron.
   		 */
   		Neuron * GetSource () const;
   		
   		/*!
   		 * \brief It sets the spike source neuron.
   		 * 
   		 * It sets the spike source neuron.
   		 * 
   		 * \param NewSource The new spike source neuron.
   		 */
   		void SetSource (Neuron * NewSource);
   		
   		/*!
   		 * \brief It process an event in the simulation.
   		 * 
   		 * It process the event in the simulation.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
   		 */
   		virtual void ProcessEvent(Simulation * CurrentSimulation) = 0;
   		
   		friend ostream & operator<< (ostream & out, Spike * spike);
   	
};

/*!
 * \brief It prints an spike in the output.
 * 
 * It prints an spike in the output.
 * 
 * \param out The output stream.
 * \param spike The spike for printing.
 */
ostream & operator<< (ostream & out, Spike * spike);

#endif /*SPIKE_H_*/
