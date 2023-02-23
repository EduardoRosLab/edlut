from pyedlut import simulation_wrapper as pyedlut
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

simulation = pyedlut.PySimulation_API()


poisson_generator_params = {'frequency': 0.0}
input_spike_1 = simulation.AddNeuronLayer(num_neurons=1, model_name='PoissonGeneratorDeviceVector', param_dict=poisson_generator_params, log_activity=False, output_activity=True)
# input_spike_1 = simulation.AddNeuronLayer(num_neurons=1, model_name='InputSpikeNeuronModel', param_dict={}, log_activity=False, output_activity=False)
input_spike_2 = simulation.AddNeuronLayer(num_neurons=1, model_name='InputSpikeNeuronModel', param_dict={}, log_activity=False, output_activity=True)
input_spike_3 = simulation.AddNeuronLayer(num_neurons=1, model_name='InputSpikeNeuronModel', param_dict={}, log_activity=False, output_activity=True)
input_spike_4 = simulation.AddNeuronLayer(num_neurons=1, model_name='InputSpikeNeuronModel', param_dict={}, log_activity=False, output_activity=True)

input_current_1 = simulation.AddNeuronLayer(num_neurons=1, model_name='InputCurrentNeuronModel', param_dict={}, log_activity=False, output_activity=False)
input_current_2 = simulation.AddNeuronLayer(num_neurons=1, model_name='InputCurrentNeuronModel', param_dict={}, log_activity=False, output_activity=False)

tar_fir_rat = 10.0
tau_thr = 50.0
integration_method = pyedlut.PyModelDescription(model_name='Euler', params_dict={'step': 0.0001})
output_params = {
  'tau_ref': 1.0,'v_thr': -40.0,'e_exc': 0.0,'e_inh': -80.0,'e_leak': -65.0,'g_leak': 0.2,'c_m': 2.0,'tau_exc': 0.5,'tau_inh': 10.0,'tau_nmda': 15.0,
  'int_meth': integration_method,
  'tau_thr': tau_thr, 'tar_fir_rat': tar_fir_rat,
}
output_layer = simulation.AddNeuronLayer(num_neurons=1, model_name='ALIFTimeDrivenModel', param_dict=output_params, log_activity=False, output_activity=True)

lrule_params_1 = {'max_LTP': 0.02*1e-5, 'tau_LTP': 0.032, 'max_LTD': 0.16*1e-5, 'tau_LTD': 0.004}
lrule_params_2 = {'max_LTP': 0.02*1e-5, 'tau_LTP': 0.032, 'max_LTD': 0.04*1e-5, 'tau_LTD': 0.032}
STDP_rule_1 = simulation.AddLearningRule('STDP', lrule_params_1)
STDP_rule_2 = simulation.AddLearningRule('STDP', lrule_params_2)

src, tgt, con, w, wcha = [], [], [], [], []

src.append(input_spike_1[0])
tgt.append(output_layer[0])
con.append(0)
w.append(1.0)
wcha.append(-1)

src.append(input_current_1[0])
tgt.append(output_layer[0])
con.append(3)
w.append(0.0)
wcha.append(-1)

src.append(input_spike_2[0])
tgt.append(output_layer[0])
con.append(0)
w.append(1e-3)
wcha.append(STDP_rule_1)

src.append(input_spike_3[0])
tgt.append(output_layer[0])
con.append(0)
w.append(1e-3)
wcha.append(STDP_rule_2)

src.append(input_spike_4[0])
tgt.append(output_layer[0])
con.append(0)
w.append(1.0)
wcha.append(-1)

synaptic_params = {'weight': w,'max_weight':100.0,'type':con,'delay':0.001,'wchange':wcha,'trigger_wchange':-1,}
synaptic_layer = simulation.AddSynapticLayer(src, tgt, synaptic_params);


simulation.Initialize()


spk = {'t':[], 'n':[]}
spk['t'].append(0.0)
spk['n'].append(input_spike_1[0])
# simulation.AddExternalSpikeActivity(spk['t'], spk['n'])

cur = {'t':[], 'n':[], 'c':[]}

# for i in range(4):
#   cur['t'].append(i*25.0)
#   cur['n'].append(input_current_1[0])
#   cur['c'].append(5 + 0.1*i)
cur['t'].append(0.0)
cur['n'].append(input_current_1[0])
cur['c'].append(10.0)
# simulation.AddExternalCurrentActivity(cur['t'], cur['n'], cur['c'])


spk = {'t':[], 'n':[]}
t = np.arange(500.0,1000.0, 0.50).tolist()
spk['t'] += t
spk['n'] += [input_spike_4[0]] * len(t)
t = np.arange(500.0,750.0, 0.505).tolist()
spk['t'] += t
spk['n'] += [input_spike_2[0]] * len(t)
t = np.arange(750.0,1000.0, 0.505).tolist()
spk['t'] += t
spk['n'] += [input_spike_3[0]] * len(t)
simulation.AddExternalSpikeActivity(spk['t'], spk['n'])


total_simulation_time = 1000.0

ws = []
wtimes = np.linspace(total_simulation_time*0.5, total_simulation_time, num=500)

for t in np.linspace(0,total_simulation_time*0.25, num=5):
  simulation.SetSpecificNeuronParams(input_spike_1[0], {'frequency': t*0.2})
  simulation.RunSimulation(t)
simulation.SetSpecificNeuronParams(input_spike_1[0], {'frequency': 0.0})
for t in wtimes:
  ws.append(simulation.GetWeights())
  simulation.RunSimulation(t)


ws = np.array(ws)
plt.plot(wtimes, ws[:,2:4])
plt.title('Synaptic weights evolution (with different STDP kernels)')
plt.show()



output_times, output_index = simulation.GetSpikeActivity()
ot = np.array(output_times)
oi = np.array(output_index)

def smooth_spikes(t,dt, spike_times, sigma_s=200.0, resolution=1000):
  sub = (spike_times>=t) * (spike_times<=(t+dt))
  spike_times = spike_times[sub]
  hist, times = np.histogram(spike_times, bins=resolution, range=(t,t+dt))
  signal = gaussian_filter1d(hist.astype('float'), sigma_s/dt)*resolution/dt
  return times[:-1], signal


plt.figure()
plt.title('Adapting threshold with different input values')

sub = oi==output_layer[0]
x, out = smooth_spikes(0,total_simulation_time, ot[sub], sigma_s=1000.0)
plt.plot(x, out, label='Output neuron')

sub = oi==input_spike_1[0]
x, out = smooth_spikes(0,total_simulation_time, ot[sub], sigma_s=1000.0)
plt.plot(x, out, label='Poisson input')

sub = oi==input_spike_2[0]
x, out = smooth_spikes(0,total_simulation_time, ot[sub], sigma_s=1000.0)
plt.plot(x, out, label='Timed spike input 1')

sub = oi==input_spike_3[0]
x, out = smooth_spikes(0,total_simulation_time, ot[sub], sigma_s=1000.0)
plt.plot(x, out, label='Timed spike input 2')

plt.ylabel('Firing rate (Hz)', rotation=90)
plt.legend()
plt.show()
