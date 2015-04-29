/***************************************************************************
 *                           CosWeightChange.h                             *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros and Niceto Luque   *
 * email                : fnaveros@ugr.es nluque@ugr.es                    *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef COSWEIGHTCHANGE_H_
#define COSWEIGHTCHANGE_H_

/*!
 * \file CosWeightChange.h
 *
 * \author Francisco Naveros
 * \author Niceto Luque
 * \date May 2014
 *
 * This file declares a class which abstracts the behaviour of a exponential-cosinusoidal additive learning rule. When a spike arrive 
 * for a non-trigger synapse, a LTP method with value a1pre is applied. When a spike arrive for a trigger synapse, a LTD method with kernel 
 * a2prepre*exp(exponent*t/tau)*cos^2((pi/2)*t/tau), for all non-trigger synapses. This kernel is applied only to the previous activity
 * to the trigger synapse.
 */
 
#include "./WithoutPostSynaptic.h"
 
/*!
 * \class CosWeightChange
 *
 * \brief Cosinusoidal learning rule.
 *
 * This class abstract the behaviour of a exponential-cosinusoidal additive learning rule. When a spike arrive for a non-trigger synapse,
 * a LTP method with value a1pre is applied. When a spike arrive for a trigger synapse, a LTD method with kernel 
 * a2prepre*exp(exponent*t/tau)*cos^2((pi/2)*t/tau), for all non-trigger synapses. This kernel is applied only to the previous activity
 * to the trigger synapse.
 *
 * \author Francisco Naveros
 * \author Niceto Luque
 * \date May 2014
 */ 
class CosWeightChange: public WithoutPostSynaptic{
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
		CosWeightChange(int NewLearningRuleIndex);

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~CosWeightChange();

		/*!
		 * \brief It initialize the state associated to the learning rule for all the synapses.
		 *
		 * It initialize the state associated to the learning rule for all the synapses.
		 *
		 * \param NumberOfSynapses the number of synapses that implement this learning rule.
		 */
		void InitializeConnectionState(unsigned int NumberOfSynapses);

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

#endif /*COSWEIGHTCHANGE_H_*/
