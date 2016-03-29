# EDLUT: Brief description #

The EDLUT (Event-Driven simulator based on _[Look-Up-Tables](http://en.wikipedia.org/wiki/Lookup_table)_) is an advanced tool that allows splitting neural network simulations in two stages:
  * **Cell behavior characterization**. In this first stage, each cell is simulated reiteratively with different input stimuli and from different cell states. This allows scanning the cell behavior which is compiled into _look-up-tables_. Usually the cell model dynamics are defined with several differential equations that define the cell behavior. Therefore at this stage the tool uses advance numerical calculation methods (such as _Runge-Kutta_ model) to provide estimations of the cell states after receiving a specific stimulus. This represents a massive simulation load for each cell type but once it is done and the results are stored in well structured tables we can avoid numerical calculation during network simulations. Furthermore, simulations can be done adopting an event-driven simulation scheme which accelerates significantly the simulation speed.
  * **Network simulation towards system behavior characterization**. At this stage, we run multiple network simulations with different weight configuration or even including _Spike-Time Dependent (STD)_ learning rules. This stage does not require numerical calculation; the neural states in different times along the simulation are retrieved from the _look-up-tables_. This allows massive simulations only possible with this kind of advance simulation tool.


---


The individual neuron models can be simulated with different simulators ([NEURON](http://www.neuron.yale.edu/neuron/), [GENESIS](http://www.genesis-sim.org/GENESIS/), etc) at different levels of detail. Recently, the EDLUT simulator has been developed to facilitate medium/large-scale network simulations based on pre-compiled models and therefore avoiding numerical calculation on-the-loop. As indicated above, using EDLUT requires compiling previously the single cell behavior into tables. This is done by massive calculation to characterize how the cell state changes in response to an input spike (depending of its initial status). For this purpose, lookup tables (LUTs) are built compiling the characteristic cell status traces in response to input spikes. Once these tables are built we can run event-driven large-scale network simulations without redoing any numerical calculation. The neuron state at any instant in response to any input spike can be retrieved from these cell characterizing LUTs.

After building up cell models based on characterizing LUTs, we need to validate the model in two ways:
  1. Accuracy validation. The number of samples in each dimension of the table can be critical to the accuracy of the table-based cell approach. Therefore, we simulate the cell model with a classical numerical calculation method (for instance, Euler method with a very short time step) and we compare the output spike train obtained in response to different input spike trains with the results obtained using the EDLUT simulator. The comparison of the output spike trains obtained by the two methods is done for instance using the Van Rossum distance.
  1. Functional validation. Key cell features are kept. If we want to abstract a cell model that includes certain cell features that are considered relevant we also need to validate that the table-based model is able to reproduce the cell features under study.
<table><tr><td><a href='http://picasaweb.google.es/lh/photo/AyiNm600lGkyJ8aYO-uDGw?feat=embedwebsite'><img src='http://lh6.ggpht.com/__Vdn9nRnoYM/Se0I3eN38-I/AAAAAAAAABo/s6dsBo_oV1U/s400/Simulation.jpg' /></a></td></tr></table>
Fig. 1. An example of cell characterization validation can be seen is illustrated in the cell-model LUT development process.

The simulation engine is illustrated in the following illustration:
<table><tr><td><a href='http://picasaweb.google.es/lh/photo/9y9vcTtGTbg2bswf_xbt9g?feat=embedwebsite'><img src='http://lh5.ggpht.com/__Vdn9nRnoYM/Se0I3FB7-tI/AAAAAAAAABY/2fpH3ot1ff0/s400/Files_Scheme.jpg' /></a></td></tr></table>
Fig. 2. Simulation platform.

One example of network simulated with the EDLUT is provided in the next figure:
<table><tr><td><a href='http://picasaweb.google.es/lh/photo/13gQyVJc8zmeY7vggDq27Q?feat=embedwebsite'><img src='http://lh4.ggpht.com/__Vdn9nRnoYM/Se0I3JjagDI/AAAAAAAAABQ/A9U-27Yp_yw/s400/Cerebellum_Architecture.jpg' /></a></td></tr></table>
Fig. 3. Simplified cerebellum model architecture simulated with the EDLUT.

The computing performance of EDLUT using a standard PC is illustrated in Fig.4.
<table><tr><td><a href='http://picasaweb.google.es/lh/photo/oUyD5JgpqRD1CvNJgrojIA?feat=embedwebsite'><img src='http://lh6.ggpht.com/__Vdn9nRnoYM/Se0I3YBPeyI/AAAAAAAAABg/tL1AZs8S9tc/s400/Performances.jpg' /></a></td></tr></table>
Fig. 4. EDLUT computation performance.

EDLUT is being developed and used in the framework of cognitive systems research as indicated in Fig. 5. In [SENSOPAC](http://www.sensopac.org) project, we simulate biologically plausible networks to investigate their computational principles and study their potential use in artificial cognitive devices.
<table><tr><td><a href='http://picasaweb.google.es/lh/photo/MYCYePpuMcJivq-8IVdt_w?feat=embedwebsite'><img src='http://lh4.ggpht.com/__Vdn9nRnoYM/Se0I3C96QwI/AAAAAAAAABI/AmVzY1TphpI/s400/Biology_Cognitive.jpg' /></a></td></tr></table>
Fig. 5. EDLUT use in the framework of cognitive systems research ([SENSOPAC](http://www.sensopac.org)).