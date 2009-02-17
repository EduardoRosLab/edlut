#ifndef PROPAGATEDSPIKE_H_
#define PROPAGATEDSPIKE_H_

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
class PropagatedSpike: public Spike{
	
	protected: 
   		/*!
   		 * >0: Interneurons spike.
   		 */
   		int target;
   		
   	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new spike object.
   		 */
   		PropagatedSpike();
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new spike with the parameters.
   		 * 
   		 * \param NewTime Time of the new spike.
   		 * \param NewSource Source neuron of the spike.
   		 * \param NewTarget >0->Interneurons spike.
   		 */
   		PropagatedSpike(double NewTime, Neuron * NewSource, int NewTarget);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~PropagatedSpike();
   	
   		/*!
   		 * \brief It gets the spike source type.
   		 * 
   		 * It gets the spike source type.
   		 * 
   		 * \return The spike source type: -1->Input spike, -2->Internal spike and >0->Interneurons spike.
   		 */
   		int GetTarget () const;
   		
   		/*!
   		 * \brief It sets the spike source type.
   		 * 
   		 * It sets the spike source type.
   		 * 
   		 * \param NewTarget The new spike source type: -1->Input spike, -2->Internal spike and >0->Interneurons spike.
   		 */
   		void SetTarget (int NewTarget);
   		
   		/*!
   		 * \brief It process an event in the simulation.
   		 * 
   		 * It process the event in the simulation.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
   		 */
   		virtual void ProcessEvent(Simulation * CurrentSimulation);
   		
};

#endif /*SPIKE_H_*/
