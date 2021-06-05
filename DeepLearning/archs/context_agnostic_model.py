import os
import torch
from overrides import overrides
from .base_model import BaseModel
from .nn_models import DecoderRNN


class ContextAgnosticModel(BaseModel):
    @overrides
    def __init__(self, embed_size, label_vocab, device, hidden_size):
        self.decoder = DecoderRNN(embed_size, hidden_size, len(label_vocab), 1).to(device)
        super(ContextAgnosticModel, self).__init__(embed_size, label_vocab, device)
        self.all_models.append(self.decoder)

    @overrides
    def _get_optimizer_params(self):
        params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) + list(
            self.encoder.bn.parameters())
        return params

    @overrides
    def _compute_outputs(self, img_features, captions, cap_lengths, contexts, con_lengths):
        outputs = self.decoder(img_features, captions, cap_lengths,
                               torch.zeros(img_features.size()).to(self.device))
        return outputs

    @overrides
    def _zero_grad(self):
        super(ContextAgnosticModel, self)._zero_grad()
        self.decoder.zero_grad()

    @overrides
    def save_model(self, model_path):
        super(ContextAgnosticModel, self).save_model(model_path)
        torch.save(self.decoder.state_dict(), os.path.join(
            model_path, 'decoder.ckpt'))

    @overrides
    def load_model(self, model_path):
        super(ContextAgnosticModel, self).load_model(model_path)
        self.decoder.load_state_dict(torch.load(os.path.join(model_path, 'decoder.ckpt'), map_location=self.device))

    @overrides
    def _sample(self, image_features, ind, contexts, con_lengths):
        sampled_ids = self.decoder.sample(image_features, torch.zeros(image_features.size()).to(self.device))
        return sampled_ids
