#ifndef ADDITIVEWEIGHTCHANGE_H_
#define ADDITIVEWEIGHTCHANGE_H_

/*!
 * \file MultiplicativeWeightChange.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a multiplicative learning rule.
 */
 
#include "./WeightChange.h"
 
/*!
 * \class AdditiveWeightChange
 *
 * \brief Additive learning rule.
 *
 * This class abstract the behaviour of a additive learning rule.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */ 
class AdditiveWeightChange: public WeightChange{
	private:
		/*!
   		 * \brief It updates the previous activity in the connection.
   		 * 
   		 * It updates the previous activity in the connection.
   		 * 
   		 * \param time The spike time.
   		 * \param Connection The connection to be modified.
   		 * \param spike True if an spike is produced.
   		 */
		void update_activity(double time,Interconnection * Connection,bool spike);
	
	public:
   	
   		/*!
   		 * \brief It applys the weight change function.
   		 * 
   		 * It applys the weight change function.
   		 * 
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyWeightChange(Interconnection * Connection,double SpikeTime);
   		
};

#endif /*ADDITIVEWEIGHTCHANGE_H_*/
