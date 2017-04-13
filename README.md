# Hyperparameters for "Capacity and Trainability of Recurrent Neural Networks"
This repository contains a dataset of hyperparameters and final train, validation, and eval losses from tasks studied in the following paper:
  > Jasmine Collins, Jascha Sohl-Dickstein, David Sussillo, **"Capacity and Trainability of Recurrent Neural Networks**"
    In _Proceedings of the Fifth International Conference on Learning Representations (ICLR)_, 2017.
 
We trained a variety of RNNs (vanilla RNN, GRU, LSTM, etc.) on several different tasks (such as input memorization, next-character prediction, parentheses counting) to uncover any fundamental differences in terms of capacity and trainability between architectures. Within each experiment, we compared architectures of the same size (based on total number of parameters) and also looked at the consequences of varying stacking depth (1, 2, 4, and sometimes 8-layer models). We ran many many iterations of hyperparameter optimizations for different models, sizes, tasks, and levels of stacking depth. Overall, this dataset contains the hyperparameter sets and associated losses for 630,880 optimizations!
 
Link to paper: [https://arxiv.org/pdf/1611.09913.pdf](https://arxiv.org/pdf/1611.09913.pdf)
  
# Why study this data?
A large amount of computational resources went into getting results for this paper, and we only report data for the top performing hyperparameters. This means that we have many thousands of additional experiments (hyperparameter sets and their outcomes) that can be studied to discover:
  * Important insights about choosing hyperparameters for RNNs
  * How optimal hyperparameters vary across different tasks and different RNN architectures

# Structure of data
The data is generated in Python and is a collection of pickled structured numpy arrays, each corresponding to an experiment figure in the paper (ex: "fig4_arith.p"). Each numpy array has the following columns/dtypes:
  * "architecture" : `string`, name of the RNN cell
     * "rnn", "irnn", "ugrnn", "gru", "lstm", "+rnn"
  * "task_type" : `string`, training task
     * "per_cap", "mem_thru", "rcf", "text8", "parens_counting", “arithmetic" 
  * "hparams" : `dict`, dictionary of hyperparameters and values
  * "n_layers" : `int`, amount of stacking depth
     * 1, 2, 4, 8
  * "hyper_iter" : `int`, hyperparameter iteration
  * "max_params" : `int`, maximum parameter budget
  * "num_params" : `int`, actual number of parameters in model
  * "train_loss" : `float`, final training loss
  * "valid_loss" : `float`, final validation loss
  * "eval_loss" : `float`, final test loss
  
The "hparams" dictionary contains different sets of hyperparameters depending on the combination of "architecture" and "task_type". The hyperparameters are described in detail in the [paper appendix](https://arxiv.org/pdf/1611.09913.pdf#page=14) (page 14). Values can be strings, ints, or floats, depending on the hyperparameter.

# Sample usage

### Unpack the data
In the terminal, run:
```
tar -xzf rnn_hp_dataset.tar.gz
```

To load the data associated with text8 task in Python and print out the hyperparameters which gave the lowest test loss for the LSTM:
```python
import numpy as np

data = np.load('fig3_txt8rcf.p')

# print all available columns
print(data.dtype.names)
# ('architecture', 'eval_loss', 'hparams', 'max_params', 'n_layers', 'num_params', 'task_type', 'train_loss', 'valid_loss', 'hyper_iter')

# print the tasks and architectures included in this dataset
print(np.unique(data['task_type']))
# ['rcf' 'text8']
print(np.unique(data['architecture']))
# ['+rnn' 'gru' 'irnn' 'lstm' 'rnn' 'ugrnn']

# find the best HP set for the LSTM on text8 next-character prediction across all model sizes and levels of stacking depth
idxes = np.flatnonzero((data['task_type'] == "text8") & (data['architecture'] == "lstm"))
min_idx = np.argmin(data['eval_loss'][idxes])
loss = data['eval_loss'][idxes[min_idx]]
params = data['num_params'][idxes[min_idx]]
layers = data['n_layers'][idxes[min_idx]]
hps = data['hparams'][idxes[min_idx]]
print('best LSTM had final loss %f, was %d-layers, %d parameters, and had HPs:'%(loss, layers, params))
# best LSTM had final loss 1.549270, was 4-layers, 1268217 parameters, and had HPs:
for key in hps.keys():
  print('%s: %s'%(key, hps[key]))
# recurrent_bias_scale_o: 1.2967042923
# recurrent_bias_scale_i: -0.250855892897
# recurrent_bias_scale_g: 0.631270706654
# recurrent_bias_scale_c: -1.62440180779
# input_scale_c: 0.209308668971
# input_scale_g: 0.222167253494
# optimizer_logit_momentum: 5.89666414261
# recurrent_matrix_init_type: orthogonal
# input_scale_i: 1.20329034328
# lr_decay_rate: 0.108852244914
# input_scale_o: 0.407466441393
# train_steps: 1176212
# forget_gate_bias_hack: 4.88674783707
# recurrent_bias_init_type: delta
# lr_init: 0.00398490205407
# optimizer: RMSProp
# gradient_clip_val: 27.6603736877
# recurrent_matrix_scale_g: 0.035970274359
# recurrent_matrix_scale_c: 0.619497835636
# recurrent_matrix_scale_o: 0.0714097246528
# recurrent_matrix_scale_i: 0.520125091076
# l2_decay: 1.79227910735e-07
```

# License
This dataset is licensed by Google Inc. under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
