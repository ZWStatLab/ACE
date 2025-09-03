## DEPICT Experiments
This folder explains how to adapt the DEPICT (https://github.com/herandy/DEPICT) training script to generate the deep clustering results used in our experiments.

## Hyperparameter tuning
1. Clone the original JULE repo:
   `git clone https://github.com/herandy/DEPICT`
2. In the file `functions.py`, change the following lines:
   - insert the following after Line 717
   ```python
   feature_prediction = lasagne.layers.get_output(encoder, input_var, deterministic=True)
   ```
   - insert the following after Line 744
   ```python
   feature_prediction_fn = theano.function([input_var], feature_prediction)
   ```   
   - change Lines 889-894 to:
   ```python
    y_pred = np.zeros(X.shape[0])
    y_features = []

    for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
        minibatch_inputs, targets, idx = batch
        minibatch_prob = test_fn(minibatch_inputs)
        minibatch_pred = np.argmax(minibatch_prob, axis=1)
        minibatch_feature = feature_prediction_fn(minibatch_inputs)
        y_features.append(np.array(minibatch_feature))
        y_pred[idx] = minibatch_pred

    y_features = np.concatenate(y_features, axis=0)
    np.savez('output{}_{}_{}_{}_{}.npz'.format(dataset, num_clusters, learning_rate, rec_mult, clus_mult), 
        y_features=y_features, y_pred=y_pred) 
   ```   


3. Modify `DEPICT.py` and save as `DEPICT_hyper.py`:
   - add type constraints `type=float` to the arguments `reconstruct_hyperparam` and `cluster_hyperparam`
      ```python
      parser.add_argument('--reconstruct_hyperparam', type=float, default=1.)
      parser.add_argument('--cluster_hyperparam', type=float, default=1.)
      ```
   - set the default float type with `theano.config.floatX = 'float32'`
   - add a marker file at the end of the script to signal completion
      ```python
      with open('done.o', 'w') as f:
         f.write('done')
      ```

4. Use the script `run_hyper.py` to run JULE trainining jobs.



## Determining the number of clusters
1. Clone the original JULE repo:
   git clone https://github.com/herandy/DEPICT
2. In the file `functions.py`, change the following lines:
   - insert the following after Line 717
   ```python
   feature_prediction = lasagne.layers.get_output(encoder, input_var, deterministic=True)
   ```
   - insert the following after Line 744
   ```python
   feature_prediction_fn = theano.function([input_var], feature_prediction)
   ```   
   - change Lines 889-894 to:
   ```python
    y_pred = np.zeros(X.shape[0])
    y_features = []

    for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
        minibatch_inputs, targets, idx = batch
        minibatch_prob = test_fn(minibatch_inputs)
        minibatch_pred = np.argmax(minibatch_prob, axis=1)
        minibatch_feature = feature_prediction_fn(minibatch_inputs)
        y_features.append(np.array(minibatch_feature))
        y_pred[idx] = minibatch_pred

    y_features = np.concatenate(y_features, axis=0)
    np.savez('output{}_{}.npz'.format(dataset, num_clusters), 
        y_features=y_features, y_pred=y_pred)
   ```   


3. Modify `DEPICT.py` and save as `DEPICT_num.py`:
   - add the argument `parser.add_argument('--num_clusters', default=10)` 
   - change Line 78 to:
      ```python
      num_clusters = int(args.num_clusters)
      ```
   - set the default float type with `theano.config.floatX = 'float32'`
   - add a marker file at the end of the script to signal completion
      ```python
      with open('done.o', 'w') as f:
         f.write('done')
      ```

4. Use the script `run_num.py` to run JULE trainining jobs.
