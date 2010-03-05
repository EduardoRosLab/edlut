/***************************************************************************
 *                           STDPWeightChange.h                            *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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


#ifndef STDPWEIGHTCHANGE_H_
#define STDPWEIGHTCHANGE_H_

#include "./LearningRule.h"

/*!
 * \file STDPWeightChange.h
 *
 * \author Jesus Garrido
 * \date August 2010
 *
 * This file declares a class which abstracts a STDP learning rule.
 */

class Interconnection;

/*!
 * \class STDPWeightChange
 *
 * \brief Learning rule.
 *
 * This class abstract the behaviour of a STDP learning rule.
 *
 * \author Jesus Garrido
 * \date March 2010
 */
class STDPWeightChange: public LearningRule {
	private:
		/*!
		 * \brief Decay parameter
		 */
		float tau;

		/*!
		 * \brief Maximum weight change
		 */
		float MaxChange;

	protected:
		/*!
		 * \brief It gets the weight difference after the last spike.
		 *
		 * It gets the weight difference after the last spike.
		 *
		 * \param Connection The connection where the spike happened.
		 * \param CurrentTime Current simulation time.
		 * \return The weight difference.
		 */
		virtual double GetWeightChange(Interconnection * Connection, double CurrentTime);


	public:
		/*!
		 * \brief It gets the number of state variables that this learning rule needs.
		 *
		 * It gets the number of state variables that this learning rule needs.
		 *
		 * \return The number of state variables that this learning rule needs.
		 */
		virtual int GetNumberOfVar() const;

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

#endif /* STDPWEIGHTCHANGE_H_ */
