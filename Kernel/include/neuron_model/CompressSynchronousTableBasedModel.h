/***************************************************************************
 *                           CompressSynchronousTableBasedModel.h                   *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#ifndef COMPRESSSYNCHRONOUSTABLEBASEDMODEL_H_
#define COMPRESSSYNCHRONOUSTABLEBASEDMODEL_H_

/*!
 * \file CompressSynchronousTableBasedModel.h
 *
 * \author Francisco Naveros
 * \date July 2015
 *
 * This file is a modification of the CompressTableBasedModel. This file declares a class which 
 * implements a neuron model based in look-up tables. The main difference it is that when
 * a input spike arrive to this model, the neuron state variables are update, but instead of
 * make a predicction in that instant, an event of type SynchronousTableBasedModelEvent is created. All 
 * the spike that arrive in the same time are computed conjointly and only one predicction is made.
 * Finally the output spike time are restricted to a multiple of SpikeRestrictionTime that is
 * loaded from the end of the table configuration file. Multiple internal spike are preccesed
 * in only one event of type SynchronousTableBasedModelInternalSpike.
 * IMPORTANT: This method is better than the traditional CompressTableBasedModel when this one 
 * receives input activity in a synchronize way, because only one prediction is made. Conversely,
 * when the input activity does not arrive in a synchronize way, this method create an innecesary
 * overhead due to the creation of the SynchronousTableBasedModelEvent.
 * 
 */

#include "CompressTableBasedModel.h"

#include "../spike/EDLUTFileException.h"

class CompressNeuronModelTable;
class Interconnection;
class SynchronousTableBasedModelEvent;


/*!
 * \class CompressTableBasedModel
 *
 * \brief Spiking neuron model based in look-up tables
 *
 * This class implements the behavior of event-driven spiking neuron models using
 * precalculated look-up tables to "predict" the neuron behavior. This "CompressSynchronous"
 * version not only can merge several look-up tables with the same indexes in just one, such as
 * his parent class, but also can synchronise the output activity, minimizing the time
 * required to process the output spikes.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 *
 * \author Francisco Naveros
 * \date July 2015
 */
class CompressSynchronousTableBasedModel: public CompressTableBasedModel {
	protected:

		/*!
		 * \brief SynchronousTableBasedModelEvent used to minimize the number of spike prediction.
		 */
		SynchronousTableBasedModelEvent * synchronousTableBasedModelEvent;

		/*!
		 * \brief Time for the next SynchronousTableBasedModelEvent.
		 */
		double SynchronousTableBasedModelTime;

		/*!
		 * \brief Parameter used to synchronize the generation of the output activity.
		 */
		double SpikeRestrictionTime;
		double inv_SpikeRestrictionTime;


		/*!
		 * \brief It loads the neuron model description.
		 *
		 * It loads the neuron type description from the file .cfg.
		 *
		 * \param ConfigFile Name of the neuron description file (*.cfg).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		void LoadNeuronModel(string ConfigFile) noexcept(false);


	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object loading the configuration of
		 * the model and the look-up tables.
		 *
		 */
		CompressSynchronousTableBasedModel();

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~CompressSynchronousTableBasedModel();

		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		virtual void LoadNeuronModel() noexcept(false);

		/*!
		 * \brief It generates the first spike (if any) in a cell.
		 *
		 * It generates the first spike (if any) in a cell.
		 *
		 * \param Cell The cell to check if activity is generated.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * GenerateInitialActivity(Neuron *  Cell);

		/*!
		 * \brief It processes a propagated spike (input spike in the cell).
		 *
		 * It processes a propagated spike (input spike in the cell).
		 *
		 * \note This function doesn't generate the next propagated spike. It must be externally done.
		 *
		 * \param inter the interconection which propagate the spike
		 * \param time the time of the spike.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessInputSpike(Interconnection * inter, double time);


		/*!
		 * \brief It predicts if the neuron would generate a internalSpike aftern all the propagated spikes have arrived. 
		 *
		 * It predicts if the neuron would generate a internalSpike after all the propagated spikes have arrived. 
		 *
		 * \param target Neuron that must be updated.
		 * \param time time.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessActivityAndPredictSpike(Neuron * target, double time);


		/*!
		 * \brief It processes an internal spike and generates an end refractory period event.
		 *
		 * It processes an internal spike and generates an end refractory period event.
		 *
		 * \param OutputSpike The spike happened.
		 *
		 * \return A new end refractory period event.
		 */
		virtual EndRefractoryPeriodEvent * ProcessInternalSpike(InternalSpike * OutputSpike);

		/*!
		 * \brief It calculates if an internal spike must be generated at the end of the refractory period.
		 *
		 * It calculates if an internal spike must be generated at the end of the refractory period.
		 *
		 * \param time end of the refractory period.
		 * \param neuron source neuron.
		 *
		 * \return A new internal spike.
		 */
		virtual InternalSpike * GenerateNextSpike(double time, Neuron * neuron);



		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 * \param OpenMPQueueIndex openmp index
		 */
		virtual void InitializeStates(int N_neurons, int OpenMPQueueIndex);


		/*!
		 * \brief It gets time and calculates the upper multiple of SpikeRestrictionTime.
		 *
		 * It gets time and calculates the upper multiple of SpikeRestrictionTime.
		 *
		 * \param time Original internal spike time.
		 * \return virtual internal spike time multiple of SpikeRestrictionTime
		 */
		double GetSpikeRestrictionTimeMultiple(double time);


		/*!
		 * \brief It returns the neuron model parameters.
		 *
		 * It returns the neuron model parameters.
		 *
		 * \returns A dictionary with the neuron model parameters
		 */
		virtual std::map<std::string,boost::any> GetParameters() const;

		/*!
		* \brief It returns the neuron model parameters for a specific neuron once the neuron model has been initilized with the number of neurons.
		*
		* It returns the neuron model parameters for a specific neuron once the neuron model has been initilized with the number of neurons.
		*
		* \param index neuron index inside the neuron model.
		*
		* \returns A dictionary with the neuron model parameters
		*
		* NOTE: this function is accesible throgh the Simulatiob_API interface.
		*/
		virtual std::map<std::string, boost::any> GetSpecificNeuronParameters(int index) const noexcept(false);

		/*!
		 * \brief It loads the neuron model properties.
		 *
		 * It loads the neuron model properties from parameter map.
		 *
		 * \param param_map The dictionary with the neuron model parameters.
		 *
		 * \throw EDLUTException If it happens a mistake with the parameters in the dictionary.
		 */
		virtual void SetParameters(std::map<std::string, boost::any> param_map) noexcept(false);

		/*!
		 * \brief It returns the default parameters of the neuron model.
		 *
		 * It returns the default parameters of the neuron models. It may be used to obtained the parameters that can be
		 * set for this neuron model.
		 *
		 * \returns A dictionary with the neuron model default parameters.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters();

		/*!
		 * \brief It creates a new neuron model object of this type.
		 *
		 * It creates a new neuron model object of this type.
		 *
		 * \param param_map The neuron model description object.
		 *
		 * \return A newly created InputNeuronModel object.
		 */
		static NeuronModel* CreateNeuronModel(ModelDescription nmDescription);

		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 *
		 * \param FileName This parameter is not used. It is stub parameter for homegeneity with other neuron models.
		 *
		 * \return A neuron model description object with the parameters of the neuron model.
		 */
		static ModelDescription ParseNeuronModel(std::string FileName) noexcept(false);

		/*!
		 * \brief It returns the name of the neuron type
		 *
		 * It returns the name of the neuron type.
		 */
		static std::string GetName();

		/*!
		* \brief It returns the neuron model information, including its parameters.
		*
		* It returns the neuron model information, including its parameters.
		*
		*\return a map with the neuron model information, including its parameters.
		*/
		static std::map<std::string, std::string> GetNeuronModelInfo();

        /*!
         * \brief Comparison operator between neuron models.
         *
         * It compares two neuron models.
         *
         * \return True if the neuron models are of the same type and with the same parameters.
         */
        virtual bool compare(const NeuronModel * rhs) const{
            if (!CompressTableBasedModel::compare(rhs)){
                return false;
            }
            const CompressSynchronousTableBasedModel * e = dynamic_cast<const CompressSynchronousTableBasedModel *> (rhs);
            if (e == 0) return false;

			return this->SpikeRestrictionTime == e->SpikeRestrictionTime;
        };
};

#endif /* SYNCHRONOUSTABLEBASEDMODEL_H_ */
