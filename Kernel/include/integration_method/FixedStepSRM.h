/***************************************************************************
 *                           FixedStepSRM.h                                *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
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

#ifndef FIXEDSTEPSRM_H_
#define FIXEDSTEPSRM_H_

/*!
 * \file IntegrationMethod.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement a fixed step integration method for SRM neuron model in CPU.
 * This class only store the value of the integration step size.
 */

#include "./FixedStep.h"

/*!
 * \class FixedStepSRM
 *
 * \brief Fixed step integration methods in CPU for SRM neuron model. This class only store the value of the
 * integration step size.
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class FixedStepSRM : public FixedStep {

	private:

		/*!
		  * \brief Default constructor of the class.
		 *
		 * It generates a new object.
		 */
		FixedStepSRM(){};
	public:


		/*!
		 * \brief Default constructor.
		 *
		 * It generates a new FixedStepSRM object.
		 *
		 */
		FixedStepSRM(TimeDrivenNeuronModel * NewModel);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~FixedStepSRM();

		
		/*!
		 * \brief It calculate the new neural state variables for a defined elapsed_time.
		 *
		 * It calculate the new neural state variables for a defined elapsed_time.
		 *
		 * \param index for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 * \param NeuronState neuron state variables of one neuron.
		 * \return Retrun if the neuron spike
		 */
		bool NextDifferentialEquationValues(int index, float * NeuronState) {
			return false;
		}

		/*!
		 * \brief It prints the integration method info.
		 *
		 * It prints the current integration method characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		 ostream & PrintInfo(ostream & out);


		/*!
		 * \brief It initialize the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It initialize the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param N_neuron number of neurons in the neuron model.
		 * \param inicialization vector with initial values.
		 */
		 void InitializeStates(int N_neurons, float * initialization){}


		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param index indicate which neuron must be reseted.
		 */
		void resetState(int index){}


		/*!
		 * \brief It calculates the conductance exponential values for time driven neuron models.
		 *
		 * It calculates the conductance exponential values for time driven neuron models.
		 */
		void Calculate_conductance_exp_values(){};

		/*!
		  * \brief It returns the integration method parameters.
	   *
	   * It returns the integration method parameters.
	   *
	   * \returns A dictionary with the integration method parameters
	   */
		virtual std::map<std::string,boost::any> GetParameters() const;

		/*!
		 * \brief It loads the integration method properties.
		 *
		 * It loads the integration method properties from parameter map.
		 *
		 * \param param_map The dictionary with the integration method parameters.
		 *
		 * \throw EDLUTFileException If it happens a mistake with the parameters in the dictionary.
		 */
		virtual void SetParameters(std::map<std::string, boost::any> param_map) noexcept(false);

		/*!
		 * \brief It returns the default parameters of the integration method.
		 *
		 * It returns the default parameters of the integration method. It may be used to obtained the parameters that can be
		 * set for this integration method.
		 *
		 * \returns A dictionary with the integration method parameters.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters();

		/*!
		 * \brief It loads the integration method description.
		 *
		 * It loads the integration method description.
		 *
		 * \param fh Filehandler of the file with the information about the integration method.
		 *
		 * \return An object with the parameters of the integration method.
		 */
		static ModelDescription ParseIntegrationMethod(FILE * fh) noexcept(false);

		/*!
		 * \brief It returns the name of the integration method
		 *
		 * It returns the name of the integration method
		 */
		static std::string GetName();

		/*!
		 * \brief It creates a new integration method object of this type.
		 *
		 * It creates a new integration method object of this type.
		 *
		 * \param param_map The integration method description object.
		 *
		 * \return A newly created integration method object.
		 */
		static IntegrationMethod* CreateIntegrationMethod(ModelDescription nmDescription, TimeDrivenNeuronModel *nmodel);

        /*!
         * \brief Comparison operator between integration methods.
         *
         * It compares two integration methods.
         *
         * \return True if the integration methods are of the same type and with the same parameters.
         */
        virtual bool compare(const IntegrationMethod * rhs) const{
            if (!FixedStep::compare(rhs)){
                return false;
            }
            const FixedStepSRM * e = dynamic_cast<const FixedStepSRM *> (rhs);
            if (e == 0) return false;
            return true;
        };

};

#endif /* FIXEDSTEPSRM_H_ */
