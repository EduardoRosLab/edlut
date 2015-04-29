/***************************************************************************
 *                           SimetricCosSinWeightChange.h                  *
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

#ifndef SIMETRICCOSSINWEIGHTCHANGE_H_
#define SIMETRICCOSSINWEIGHTCHANGE_H_

/*!
 * \file CosWeightChange.h
 *
 * \author Francisco Naveros
 * \author Niceto Luque
 * \date May 2014
 *
 * This file declares a class which abstracts a exponential-cosinusoidal additive learning rule. This learning rule only uses the previous and future
 * spikes to the trigger spike to update the weight. 
 */
 
#include "./WithoutPostSynaptic.h"
 
/*!
 * \class SimetricCosWeightChange
 *
 * \brief Cosinusoidal learning rule.
 *
 * This class abstract the behaviour of a exponential-cosinusoidal additive learning rule.
 *
 * \author Francisco Naveros
 * \author Niceto Luque
 * \date May 2014
 */ 

class SimetricCosSinWeightChange: public WithoutPostSynaptic{
	private:


		/*!
		 * Distance un second between the maximun and minimun value
		 */
		float MaxMinDistance;

		/*!
		 * Max increment in nanosiemen for the central lobule
		 */
		float CentralAmplitudeFactor;

		/*!
		 * Max increment in nanosiemen for the lateral lobules
		 */
		float LateralAmplitudeFactor;



		
	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new learning rule with its index.
		 *
		 * \param NewLearningRuleIndex learning rule index.
		 */ 
		SimetricCosSinWeightChange(int NewLearningRuleIndex);

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~SimetricCosSinWeightChange();

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

#endif /*SIMETRICCOSSINWEIGHTCHANGE_H_*/
