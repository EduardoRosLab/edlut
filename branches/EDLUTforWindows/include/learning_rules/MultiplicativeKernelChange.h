/***************************************************************************
 *                           MultiplicativeKernelChange.h                  *
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


#ifndef MULTIPLICATIVEKERNELCHANGE_H_
#define MULTIPLICATIVEKERNELCHANGE_H_

#include "./LearningRule.h"

class MultiplicativeKernelChange: public LearningRule {
	private:
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
		 * Activity register.
		 */
		float lpar[3];

		/*!
		 * Activity register.
		 */
		float cpar[3];

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
		 * \brief It gets the Lpar parameter of the learning rule.
		 *
		 * It gets the Lpar parameter of the learning rule.
		 *
		 * \param index Parameter index.
		 *
		 * \return Lpar parameter at indexth position.
		 */
		float GetLparAt(int index) const;

		/*!
		 * \brief It sets the Lpar parameter of the learning rule.
		 *
		 * It sets the Lpar parameter of the learning rule.
		 *
		 * \param index Parameter index.
		 * \param NewLpar parameter at indexth position.
		 */
		void SetLparAt(int index, float NewLpar);

		/*!
		 * \brief It gets the Cpar parameter of the learning rule.
		 *
		 * It gets the Cpar parameter of the learning rule.
		 *
		 * \param index Parameter index.
		 *
		 * \return Cpar parameter at indexth position.
		 */
		float GetCparAt(int index) const;

		/*!
		 * \brief It sets the Cpar parameter of the learning rule.
		 *
		 * It sets the Cpar parameter of the learning rule.
		 *
		 * \param index Parameter index.
		 * \param NewCpar parameter at indexth position.
		 */
		void SetCparAt(int index, float NewCpar);

		/*!
		 * \brief It gets the number of state variables that this learning rule needs.
		 *
		 * It gets the number of state variables that this learning rule needs.
		 *
		 * \return The number of state variables that this learning rule needs.
		 */
		virtual int GetNumberOfVar() const;

};

#endif /* STDPWEIGHTCHANGE_H_ */
