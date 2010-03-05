/***************************************************************************
 *                           AdditiveKernelChange.h                        *
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

#ifndef ADDITIVEKERNELCHANGE_H_
#define ADDITIVEKERNELCHANGE_H_

/*!
 * \file AdditiveKernelChange.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date March 2010
 *
 * This file declares a class which abstracts an additive learning rule.
 */

#include "./LearningRule.h"

/*!
 * \class AdditiveKernelChange
 *
 * \brief Additive learning rule with kernel.
 *
 * This class abstract the behaviour of a additive learning rule with kernel.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date March 2010
 */
class AdditiveKernelChange : public LearningRule {
	protected:
		/*!
		 * Maximum time of the learning rule.
		 */
		float maxpos;

		/*!
		 * Number of activity registers.
		 */
		int numexps;

		/*!
		 * This weight change is a trigger.
		 */
		int trigger;

		/*!
		 * Learning rule parameter 1.
		 */
		float a1pre;

		/*!
		 * Learning rule parameter 2.
		 */
		float a2prepre;

		/*!
		 * \brief It updates the activity state.
		 *
		 * It updates the activity state.
		 *
		 * \param time The current simulation time.
		 * \param Connection The connection to be updated.
		 * \param spike If an spike has just happened in current connection.
		 */
		virtual void update_activity(double time,Interconnection * Connection,bool spike);


	public:
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

		/*!
		 * \brief It gets the number of state variables that this learning rule needs.
		 *
		 * It gets the number of state variables that this learning rule needs.
		 *
		 * \return The number of state variables that this learning rule needs.
		 */
		virtual int GetNumberOfVar() const;
};



#endif /* ADDITIVEKERNELCHANGE_H_ */
