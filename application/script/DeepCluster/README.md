## DeepCluster Experiments
This folder explains how to adapt the DeepCluster (https://github.com/herandy/DEPICT) training script to reproduce the deep clustering results used in our experiments.

1. Clone the original DeepCluster repo:
   git clone https://github.com/facebookresearch/deepcluster
2. In the file `clustering.py`, change the following lines:
   - change Line 179 to:
   ```python
   stats = clus.iteration_stats
   losses = np.array([stats.at(i).obj for i in range(stats.size())])
   ```   
   - change Line 219 to:
   ```python
   return loss, xb
   ```   
3. Modify `main.py`:
   - change Lines 98-105 to:
   ```python
   checkpoint = torch.load(args.resume)
   # args.start_epoch = checkpoint['epoch']
   # remove top_layer parameters from checkpoint
   keys = checkpoint['state_dict'].copy()
   for key in keys:
       if 'top_layer' in key:
           del checkpoint['state_dict'][key]
   model.load_state_dict(checkpoint['state_dict'])
   #optimizer.load_state_dict(checkpoint['optimizer'])
   ```

   - change Lines 150-155 to:
   ```python
   features, labels = compute_features(dataloader, model, len(dataset))
   # cluster the features
   if args.verbose:
      print('Cluster the features')
   clustering_loss, pro_features = deepcluster.cluster(features, verbose=args.verbose)

   ```

   - add the following after Line 200
   ```python
   estimates = clustering.arrange_clustering(deepcluster.images_lists)
   np.savez('npfiles_val/pro_output_{}.npz'.format(epoch), estimates=estimates, labels=labels, pro_features=pro_features)
   ```

   - modify the code between Line 300 and 322 to get the true label

   ```python
   labels = []
   for i, (input_tensor, label) in enumerate(dataloader):
       ...   
       labels.append(label.numpy())
   labels = np.concatenate(labels, axis=0)
   return features, labels

   ```

   - modify Line 262 to:
   ```python
   target = target.cuda(non_blocking=True)
   ```

   - modify Line 270 to:
   ```python
   losses.update(loss.data, input_tensor.size(0))
   ```

4. Use the script `run.sh` to run the trainining job.



