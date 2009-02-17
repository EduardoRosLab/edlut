#ifndef OUTPUTWEIGHTDRIVER_H_
#define OUTPUTWEIGHTDRIVER_H_

/*!
 * \file OutputWeightDriver.h
 *
 * \author Jesus Garrido
 * \date November 2008
 *
 * This file declares a class for communicate network synaptic weights.
 */
 
#include "../../spike/include/EDLUTException.h" 

class Network;

/*!
 * \class OutputWeightDriver
 *
 * \brief Class for communicate network synaptic weights. 
 *
 * This class abstract methods for saving the network synaptic weights. Its subclasses
 * implements the output target and methods.
 *
 * \author Jesus Garrido
 * \date November 2008
 */
class OutputWeightDriver {

	public:
	
		/*!
		 * \brief Default destructor.
		 * 
		 * Default destructor.
		 */
		virtual ~OutputWeightDriver();
	
		/*!
		 * \brief It communicates the output activity to the external system.
		 * 
		 * This method introduces the network synaptic weights to the output target (file, tcpip system...).
		 * 
		 * \param Net The network to save the weights.
		 * \param SimulationTime The current simulation time.
		 * 
		 * \throw EDLUTException If something wrong happens in the output process.
		 */
		virtual void WriteWeights(Network * Net, float SimulationTime) throw (EDLUTException) = 0;

		/*!
		 * \brief It communicates the output activity to the external system.
		 * 
		 * This method introduces the network synaptic weights to the output target (file, tcpip system...).
		 * 
		 * \param Net The network to save the weights.
		 * 
		 * \throw EDLUTException If something wrong happens in the output process.
		 */
		virtual void WriteWeights(Network * Net) throw (EDLUTException) = 0;

};

#endif /*OUTPUTDRIVER_H_*/
