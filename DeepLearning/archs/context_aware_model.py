import os
import torch
from overrides import overrides
from .context_agnostic_model import ContextAgnosticModel
from .nn_models import EncoderRNN


class ContextAwareModel(ContextAgnosticModel):
    @overrides
    def __init__(self, embed_size, label_vocab, device, hidden_size, context_vocab, glove_dim, context_embed_size):
        self.context_vocab = context_vocab
        weights_matrix = context_vocab.glove_weights()
        self.weights_matrix = torch.Tensor(weights_matrix)
        self.context_encoder = EncoderRNN(context_embed_size, hidden_size, len(context_vocab), self.weights_matrix,
                                          glove_dim=glove_dim, device=device).to(device)
        super(ContextAwareModel, self).__init__(embed_size, label_vocab, device, hidden_size)
        self.all_models.append(self.context_encoder)

    @overrides
    def _get_optimizer_params(self):
        params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) + list(
            self.encoder.bn.parameters()) + list(
            self.context_encoder.parameters())  # list(decoder.parameters()) + list(encoder.embed.parameters())
        return params

    @overrides
    def _compute_outputs(self, img_features, captions, cap_lengths, contexts, con_lengths):
        output, hidden = self.context_encoder(contexts, con_lengths)
        outputs = self.decoder(img_features, captions, cap_lengths, hidden)
        return outputs

    @overrides
    def _zero_grad(self):
        super(ContextAwareModel, self)._zero_grad()
        self.context_encoder.zero_grad()

    @overrides
    def save_model(self, model_path):
        super(ContextAwareModel, self).save_model(model_path)
        torch.save(self.context_encoder.state_dict(), os.path.join(model_path, 'context.ckpt'))

    @overrides
    def load_model(self, model_path):
        super(ContextAwareModel, self).load_model(model_path)
        self.context_encoder.load_state_dict(
            torch.load(os.path.join(model_path, 'context.ckpt'), map_location=self.device))

    @overrides
    def _sample(self, image_features, ind, contexts, con_lengths):
        _, hidden = self.context_encoder(contexts[ind].unsqueeze(0), torch.tensor(con_lengths[ind]).unsqueeze(0))
        sampled_ids = self.decoder.sample(image_features, hidden.unsqueeze(0))  # .unsqueeze(0)
        return sampled_ids
