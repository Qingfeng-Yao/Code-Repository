import torch
import torch.nn as nn

from utils import get_param_val, _create_length_mask

from discrete_flows.made import MADE
from discrete_flows.disc_models import DiscreteAutoregressiveFlow, DiscreteBipartiteFlow

class DFModel(nn.Module):
    
    def __init__(self, num_classes, batch_size=64, hidden_size=8, num_flows=1, temperature=0.1, max_seq_len=-1, model_params=None, model_name="DAF"):
        super().__init__()
        hidden_size = get_param_val(model_params["discrete_flow"], "nh", hidden_size)
        num_flows = get_param_val(model_params["discrete_flow"], "num_flows", num_flows)
        temperature = get_param_val(model_params["discrete_flow"], "temperature", temperature)
        max_seq_len = get_param_val(model_params, "max_seq_len", max_seq_len)

        self.num_flows = num_flows
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.vocab_size = num_classes
        self.model_name = model_name

        flows = []
        for i in range(self.num_flows):
            if model_name == "DAF":
                layer = MADE([batch_size, max_seq_len, self.vocab_size], self.vocab_size, [self.hidden_size, self.hidden_size, self.hidden_size])
                disc_layer = DiscreteAutoregressiveFlow(layer, temperature, self.vocab_size)
            
            elif model_name == "DBF":
                vector_length = self.vocab_size*max_seq_len
                layer = lambda inputs, **kwargs: inputs
                disc_layer = DiscreteBipartiteFlow(layer, i%2, temperature, self.vocab_size, vector_length)
                # i%2 flips the parity of the masking. It splits the vector in half and alternates
                # each flow between changing the first half or the second. 
            
            flows.append(disc_layer)

        self.flows = nn.ModuleList(flows)

        # Making random base probability distribution
        self.base_log_probs = torch.randn(max_seq_len, self.vocab_size).clone().detach().requires_grad_(True)

    def forward(self, z, reverse=False, **kwargs):
        if not reverse:
            # from the data to the latent space. This is how the base code is implemented. 
            for flow in self.flows:
                z = flow.forward(z)
            return z
        else:
            # from the latent space to the data
            for flow in self.flows[::-1]:
                z = flow.reverse(z)
            return z

    def need_data_init(self):
        return False

    def info(self):
        return "%s with hidden size %i and %i flows" % (self.model_name, self.hidden_size, self.num_flows)