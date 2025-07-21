# First approach
In this approach we will try to do this:

1. Create two identical RNN, one called net and the other one called mask. You will have to initialise them randomly to ensure the weights are different.
2. Activate the highest 30% of the weights in mask.
3. Use this pruned mask to to prune 70% of weights in net (If the weight in mask is 1, keep the weight in net. If the weight in mask is 0, the weight in mask should be 0).
4. Froward the images through the pruned net and calculate the loss of the net.
5. Back-propagate that loss through the un-pruned mask and update the new weights.
6. Test the results after each epoch with the pruned net.
7. Repeat 2-6 with the un-pruned mask and net.