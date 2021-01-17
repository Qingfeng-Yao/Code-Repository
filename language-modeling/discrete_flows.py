import torch
import torch.nn as nn
import torch.nn.functional as F

import util

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        self.register_buffer('mask', mask)

    def forward(self, inputs):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        return output
nn.MaskedLinear = MaskedLinear

class MADE(nn.Module):
    # 添加channel维度
    # 不计算log-Jacobian

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 n_hidden,
                 channels,
                 act='relu'):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        # ensure always more hidden units than sequence length: 
        assert num_hidden >= num_inputs, "Sequence length is larger than size of hidden units. Increase num_hidden."

        self.length = num_inputs
        self.channels = channels

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        input_mask = input_mask.unsqueeze(1).repeat(1, channels, 1)
        input_mask = input_mask.view(input_mask.shape[0], input_mask.shape[-1] * channels)

        hidden_masks = []
        for _ in range(n_hidden):
            hidden_masks.append(get_mask(num_hidden, num_hidden, num_inputs))

        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')
        output_mask = output_mask.unsqueeze(-1).repeat(1, 1, channels)
        output_mask = output_mask.view(output_mask.shape[0] * channels, output_mask.shape[1])

        self.joiner = nn.MaskedLinear(num_inputs*channels, num_hidden, input_mask)

        self.trunk = []
        for m in hidden_masks:
            self.trunk += [act_func(), nn.MaskedLinear(num_hidden, num_hidden,
                                                   m)]
        self.trunk += [act_func(), nn.MaskedLinear(num_hidden, num_inputs * 2 * channels,
                                                   output_mask)]
        self.trunk = nn.Sequential(*self.trunk)

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            input_shapes = inputs.shape
            inputs = inputs.view(-1, input_shapes[-1]*input_shapes[-2])
            h = self.joiner(inputs)
            o = self.trunk(h)
            o = o.view(-1, input_shapes[-2]*2, self.channels)
            # m, a = o.chunk(2, 1)
            return o

class DiscreteAutoregressiveFlow(nn.Module):
    def __init__(self, layer, temperature, vocab_size):
        super().__init__()
        self.layer = layer
        self.temperature = temperature
        self.vocab_size = vocab_size

    def forward(self, inputs):
        """Forward pass returning the autoregressive transformation. Data to latent."""

        net = self.layer(inputs)
        loc, scale = net.chunk(2, 1)
            
        scale = util.one_hot_argmax(scale, self.temperature).type(inputs.dtype)
        scaled_inputs = util.one_hot_multiply(inputs, scale)
        
        loc = util.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
        outputs = util.one_hot_add(scaled_inputs, loc)
        return outputs

    def reverse(self, inputs):
        """Reverse pass for left-to-right autoregressive generation. Latent to data. 
        Expects to recieve a onehot."""
        length = inputs.shape[-2]
        if length is None:
            raise NotImplementedError('length dimension must be known. Ensure input is a onehot with 3 dimensions (batch, length, onehot)')
        # Slowly go down the length of the sequence. 
        # the batch is computed in parallel, dont get confused with it and the sequence components!
        # From initial sequence tensor of shape [..., 1, vocab_size]. In a loop, we
        # incrementally build a Tensor of shape [..., t, vocab_size] as t grows.
        outputs = self._initial_call(inputs[:, 0, :], length)
        # TODO: Use tf.while_loop. Unrolling is memory-expensive for big
        # models and not valid for variable lengths.
        for t in range(1, length):
            outputs = self._per_timestep_call(outputs,
                                            inputs[..., t, :],
                                            length,
                                            t)
        return outputs

    def _initial_call(self, new_inputs, length):
        """Returns Tensor of shape [..., 1, vocab_size].
        Args:
        new_inputs: Tensor of shape [..., vocab_size], the new input to generate
            its output.
        length: Length of final desired sequence.
        """
        inputs = new_inputs.unsqueeze(1) #new_inputs[..., tf.newaxis, :] # batch x 1 x onehots
        # TODO: To handle variable lengths, extend MADE to subset its
        # input and output layer weights rather than pad inputs.
        padded_inputs = F.pad(
            inputs, (0,0,0, length - 1) )
        
        """
        All this is doing is filling the input up to its length with 0s. 
        [[0, 0]] * 2 + [[0, 50 - 1], [0, 0]] -> [[0, 0], [0, 0], [0, 49], [0, 0]]
        what this means is, dont add any padding to the 0th dimension on the front or back. 
        same for the 2nd dimension (here we assume two tensors are for batches), for the length dimension, 
        add length -1 0s after. 
        
        """
        net = self.layer(padded_inputs) # feeding this into the MADE network. store these as net.
        loc, scale = net.chunk(2, 1)
        loc = loc[..., 0:1, :] 
        loc = util.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
        scale = scale[..., 0:1, :]
        scale = util.one_hot_argmax(scale, self.temperature).type(inputs.dtype)
        inverse_scale = util.multiplicative_inverse(scale, self.vocab_size) # could be made more efficient by calculating the argmax once and passing it into both functions. 
        shifted_inputs = util.one_hot_minus(inputs, loc)
        outputs = util.one_hot_multiply(shifted_inputs, inverse_scale)
        return outputs

    def _per_timestep_call(self,
                            current_outputs,
                            new_inputs,
                            length,
                            timestep):
        """Returns Tensor of shape [..., timestep+1, vocab_size].
        Args:
        current_outputs: Tensor of shape [..., timestep, vocab_size], the so-far
            generated sequence Tensor.
        new_inputs: Tensor of shape [..., vocab_size], the new input to generate
            its output given current_outputs.
        length: Length of final desired sequence.
        timestep: Current timestep.
        """
        inputs = torch.cat([current_outputs,
                            new_inputs.unsqueeze(1)], dim=-2)
        # TODO: To handle variable lengths, extend MADE to subset its
        # input and output layer weights rather than pad inputs.

        padded_inputs = F.pad(
            inputs, (0,0,0, length - timestep - 1) ) # only pad up to the current timestep

        net = self.layer(padded_inputs)
        loc, scale = net.chunk(2, 1)
        loc = loc[..., :(timestep+1), :]
        loc = util.one_hot_argmax(loc, self.temperature).type(inputs.dtype)
        scale = scale[..., :(timestep+1), :]
        scale = util.one_hot_argmax(scale, self.temperature).type(inputs.dtype)
        inverse_scale = util.multiplicative_inverse(scale, self.vocab_size)
        shifted_inputs = util.one_hot_minus(inputs, loc)
        new_outputs = util.one_hot_multiply(shifted_inputs, inverse_scale)
        outputs = torch.cat([current_outputs, new_outputs[..., -1:, :]], dim=-2)
        return outputs

class DiscreteAutoFlowModel(nn.Module):
    # combines all of the discrete flow layers into a single model
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
         # from the data to the latent space. This is how the base code is implemented. 
        for flow in self.flows:
            z = flow.forward(z)
        return z

    def reverse(self, x):
        # from the latent space to the data
        for flow in self.flows[::-1]:
            x = flow.reverse(x)
        return x