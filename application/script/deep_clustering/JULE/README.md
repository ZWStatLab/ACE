## Using with JULE
The folder provides how to adapt the training script JULE (https://github.com/jwyang/JULE.torch) to generate the deep clustering results in our experiment.


## Hyperparameter tuning
1. Clone the original JULE repo:
   git clone https://github.com/jwyang/JULE.torch
2. In file `train.lua`, change the following lines, and save the modified file as `train_hyper.lua`:
   - add the code block between Line 496 and Line 497
   <pre>
   ```lua
   labelname = "label"..opt.dataset..opt.learningRate..opt.eta..".h5"
   featurename = "feature"..opt.dataset..opt.learningRate..opt.eta..".h5"

   local labelFile = hdf5.open(labelname, 'w')
   labelFile:write('label', label_pre_tensor_table[1]:long())
   labelFile:close()

   local featureFile = hdf5.open(featurename, 'w')
   featureFile:write('feature', features:float())
   featureFile:close()

   modelFile = 'model'..opt.dataset..opt.learningRate..opt.eta..'.dat'
   torch.save(modelFile, network_table[1])

   file = io.open("done.o", "w")
   io.close(file)
   ```
   </pre>

3. Save your modified file and run it with our scripts in this repo.
