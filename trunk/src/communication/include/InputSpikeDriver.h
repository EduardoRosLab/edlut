#ifndef INPUTSPIKEDRIVER_H_
#define INPUTSPIKEDRIVER_H_

/*!
 * \file InputSpikeDriver.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class for getting external input spikes.
 */
#include "../../spike/include/EDLUTException.h"
 
class EventQueue;
class Network;

/*!
 * \class InputSpikeDriver
 *
 * \brief Class for getting input spikes. 
 *
 * This class abstract methods for getting the input spikes to the network. Its subclasses
 * implements the input source and methods.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class InputSpikeDriver {
	
	protected:
	
		/*!
		 * Is the input proccess finished?
		 */
		bool Finished;

	public:
	
		/*!
		 * \brief Default destructor.
		 * 
		 * Default destructor.
		 */
		virtual ~InputSpikeDriver();
	
		/*!
		 * \brief It introduces the input activity in the simulation event queue.
		 * 
		 * This method introduces the cumulated input activity in the simulation event queue.
		 * 
		 * \param Queue The event queue where the input spikes are inserted.
		 * \param Net The network associated to the input spikes.
		 * 
		 * \throw EDLUTException If something wrong happens in the input process.
		 */
		virtual void LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTException) = 0;
		
		/*!
		 * \brief It checks if the input process is finished.
		 * 
		 * It checks if the input process is finished.
		 * 
		 * \return True if the inputs have been finished. False in other case.
		 */
		bool IsFinished() const;
};


#endif /*INPUTDRIVER_H_*/
