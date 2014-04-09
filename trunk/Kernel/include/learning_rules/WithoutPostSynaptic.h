/***************************************************************************
 *                           WithoutPostSynaptic.h                         *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
 * email                : fnaveros@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef WITHOUTPOSTSYNAPTIC_H_
#define WITHOUTPOSTSYNAPTIC_H_

#include "./LearningRule.h"

/*!
 * \file WithoutPostSynaptic.h
 *
 * \author Francisco Naveros
 * \date November 2013
 *
 * This file declares a class which abstracts a learning rule that does not implement postsynaptic learning.
 */

/*!
 * \class WithLearningRule
 *
 * \brief Learning rule.
 *
 * This class abstract the behaviour of a learning rule that does not implement postsynaptic learning.
 *
 * \author Francisco Naveros
 * \date November 2013
 */
class WithoutPostSynaptic : public LearningRule {

	public:

		/*!
		 * \brief It initialize the state associated to the learning rule for all the synapses.
		 *
		 * It initialize the state associated to the learning rule for all the synapses.
		 *
		 * \param NumberOfSynapses the number of synapses that implement this learning rule.
		 */
		virtual void InitializeConnectionState(unsigned int NumberOfSynapses) = 0;


		/*!
		 * \brief Default constructor.
		 * 
		 * It creates a new WithoutPostSynaptic object.
		 */ 
		WithoutPostSynaptic();

		/*!
		 * \brief Object destructor.
		 *
		 * It remove a WithoutPostSynaptic object.
		 */
		virtual ~WithoutPostSynaptic();

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
		virtual void LoadLearningRule(FILE * fh, long & Currentline) throw (EDLUTFileException)= 0;

   		/*!
   		 * \brief It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime) = 0;

   		/*!
		 * \brief It applies the weight change function when a postsynaptic spike arrives.
		 *
		 * It applies the weight change function when a postsynaptic spike arrives.
		 *
		 * \param Connection The connection where the learning rule happens.
		 * \param SpikeTime The spike time of the postsynaptic spike.
		 */
		void ApplyPostSynapticSpike(Interconnection * Connection, double SpikeTime);

   		/*!
		 * \brief It prints the learning rule info.
		 *
		 * It prints the current learning rule characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out) = 0;

   		/*!
		 * \brief It returns if this learning rule implements postsynaptic learning.
		 *
		 * It returns if this learning rule implements postsynaptic learning.
		 *
		 * \returns if this learning rule implements postsynaptic learning
		 */
		bool ImplementPostSynaptic();

};

#endif /* WITHOUTPOSTSYNAPTIC_H_ */
