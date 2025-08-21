import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MaskedForward(nn.Module):
    def __init__(self, net, mask, sparsity):
        super(MaskedForward, self).__init__()
        self.net = net
        self.mask = mask
        self.sparsity = sparsity

    def _binary_mask(self, scores):
        k = int((1.0 - self.sparsity) * scores.numel())
        threshold = torch.kthvalue(scores.flatten(), k).values
        hard_mask = (scores > threshold).float()
        return hard_mask + (scores - scores.detach())  # STE: binarized forward, gradient passthrough

    def forward(self, x):
        # Check if the model has features and classifier (AlexNet)
        if hasattr(self.net, 'features') and hasattr(self.net, 'classifier'):
            # AlexNet-style forward pass
            out = x
            for idx, (n_layer, m_layer) in enumerate(zip(self.net.features, self.mask.features)):
                if isinstance(n_layer, nn.Conv2d):
                    mask_scores = m_layer.weight.abs()
                    mask_bin = self._binary_mask(mask_scores)
                    binary_mask = mask_bin + (mask_scores - mask_scores)
                    masked_weight = n_layer.weight * binary_mask
                    out = F.conv2d(out, masked_weight, n_layer.bias, stride=n_layer.stride, padding=n_layer.padding)
                else:
                    out = n_layer(out)

            out = torch.flatten(out, 1)

            for idx, (n_layer, m_layer) in enumerate(zip(self.net.classifier, self.mask.classifier)):
                if isinstance(n_layer, nn.Linear):
                    mask_scores = m_layer.weight.abs()
                    mask_bin = self._binary_mask(mask_scores)
                    binary_mask = mask_bin + (mask_scores - mask_scores)
                    masked_weight = n_layer.weight * binary_mask
                    out = F.linear(out, masked_weight, n_layer.bias)
                else:
                    out = n_layer(out)
        else:
            # SimpleMLP-style forward pass
            out = x.view(x.size(0), -1)  # Flatten input
            
            # Process fc1
            mask_scores = self.mask.fc1.weight.abs()
            mask_bin = self._binary_mask(mask_scores)
            binary_mask = mask_bin + (mask_scores - mask_scores)
            masked_weight = self.net.fc1.weight * binary_mask
            out = F.linear(out, masked_weight, self.net.fc1.bias)
            out = self.net.relu(out)
            
            # Process fc2
            mask_scores = self.mask.fc2.weight.abs()
            mask_bin = self._binary_mask(mask_scores)
            binary_mask = mask_bin + (mask_scores - mask_scores)
            masked_weight = self.net.fc2.weight * binary_mask
            out = F.linear(out, masked_weight, self.net.fc2.bias)
            out = self.net.relu(out)
            
            # Process fc3 (output layer)
            mask_scores = self.mask.fc3.weight.abs()
            mask_bin = self._binary_mask(mask_scores)
            binary_mask = mask_bin + (mask_scores - mask_scores)
            masked_weight = self.net.fc3.weight * binary_mask
            out = F.linear(out, masked_weight, self.net.fc3.bias)

        return out
