/***************************************************************************
 *                           SimetricCosSTDPWeightChange.h                 *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros and Niceto Luque   *
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


#ifndef SIMETRICCOSSTDPWEIGHTCHANGE_H_
#define SIMETRICCOSSTDPWEIGHTCHANGE_H_

#include "./WithPostSynaptic.h"

/*!
 * \file SimetricCosSTDPWeightChange.h
 *
 * \author Francisco Naveros
 * \date May 2014
 *
 * This file declares a class which abstracts a STDP learning rule.
 */

class Interconnection;

/*!
 * \class SimetricCosSTDPWeightChange
 *
 * \brief Learning rule.
 *
 * This class abstract the behaviour of a STDP learning rule.
 *
 * \author Francisco Naveros
 * \date May 2014
 */
class SimetricCosSTDPWeightChange: public WithPostSynaptic {
	private:
		/*!
		 * \brief Kernel amplitude in second.
		 */
		float tau;

		/*!
		 * \brief Exponent
		 */
		float exponent;

		/*!
		 * \brief Maximum weight change for LTP
		 */
		float a1pre;

		/*!
		 * \brief Maximum weight change LTD
		 */
		float a2prepre;

	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new learning rule with its index.
		 *
		 * \param NewLearningRuleIndex learning rule index.
		 */ 
		SimetricCosSTDPWeightChange(int NewLearningRuleIndex);

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~SimetricCosSTDPWeightChange();


		/*!
		 * \brief It initialize the state associated to the learning rule for all the synapses.
		 *
		 * It initialize the state associated to the learning rule for all the synapses.
		 *
		 * \param NumberOfSynapses the number of synapses that implement this learning rule.
		 */
		virtual void InitializeConnectionState(unsigned int NumberOfSynapsesAndNeurons);


		/*!
		 * \brief It loads the learning rule properties.
		 *
		 * It loads the learning rule properties.
		 *
		 * \param fh A file handler placed where the Learning rule properties are defined.
		 * \param Currentline The file line where the handler is placed.
		 *
		 * \throw EDLUTFileException If something wrong happens in reading the learning rule properties.
		 */
		virtual void LoadLearningRule(FILE * fh, long & Currentline) throw (EDLUTFileException);

   		/*!
   		 * \brief It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime);

   		/*!
		 * \brief It applies the weight change function when a postsynaptic spike arrives.
		 *
		 * It applies the weight change function when a postsynaptic spike arrives.
		 *
		 * \param Connection The connection where the learning rule happens.
		 * \param SpikeTime The spike time of the postsynaptic spike.
		 */
		virtual void ApplyPostSynapticSpike(Interconnection * Connection,double SpikeTime);

		/*!
		 * \brief It prints the learning rule info.
		 *
		 * It prints the current learning rule characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out);


};

#endif /* SIMETRICCOSSTDPWEIGHTCHANGE_H_ */
