import os
import torch
from overrides import overrides
from .context_aware_model import ContextAwareModel
from .nn_models import AttnDecoderRNN


class AttentionalContextAwareModel(ContextAwareModel):
    @overrides
    def __init__(self, embed_size, label_vocab, device, hidden_size, context_vocab, glove_dim, context_embed_size):
        self.attn_decoder = AttnDecoderRNN(embed_size, hidden_size, len(label_vocab), device=device).to(device)
        super(AttentionalContextAwareModel, self).__init__(embed_size, label_vocab, device, hidden_size,
                                                           context_vocab, glove_dim, context_embed_size)
        self.all_models.append(self.attn_decoder)

    @overrides
    def _get_optimizer_params(self):
        params = list(self.attn_decoder.parameters()) + list(self.encoder.linear.parameters()) + list(
            self.encoder.bn.parameters()) + list(
            self.context_encoder.parameters())  # list(decoder.parameters()) + list(encoder.embed.parameters())
        return params

    @overrides
    def _compute_outputs(self, img_features, captions, cap_lengths, contexts, con_lengths):

        # context_features, agg_context_features = self.context_encoder(contexts, con_lengths)
        enc_out, enc_hidden = self.context_encoder(torch.cat((img_features.unsqueeze(1), contexts),
                                                             dim=1), con_lengths) #b*3,100 ,b*100
        # hidden = img_features
        outputs = torch.zeros(1, len(self.label_vocab), device=self.device)
        for idx, caption in enumerate(captions):
            # hidden = self.attn_decoder.initHidden()
            # hidden = img_features[idx].unsqueeze(0).unsqueeze(0)
            hidden = enc_hidden[idx].unsqueeze(0).unsqueeze(0)
            for j in range(cap_lengths[idx]):
                out, hidden, weights = self.attn_decoder(caption[j].unsqueeze(0).unsqueeze(0), hidden,
                                                         enc_out[idx])
                                                         # context_features[idx])
                outputs = torch.cat((outputs, out), 0)
        outputs = outputs[1:, :]
        return outputs

    @overrides
    def _zero_grad(self):
        super(AttentionalContextAwareModel, self)._zero_grad()
        self.attn_decoder.zero_grad()

    @overrides
    def save_model(self, model_path):
        super(AttentionalContextAwareModel, self).save_model(model_path)
        torch.save(self.attn_decoder.state_dict(), os.path.join(model_path, 'attn_decoder.ckpt'))

    @overrides
    def load_model(self, model_path):
        super(AttentionalContextAwareModel, self).load_model(model_path)
        self.attn_decoder.load_state_dict(
            torch.load(os.path.join(model_path, 'attn_decoder.ckpt'), map_location=self.device))

    @overrides
    def _sample(self, image_features, ind, contexts, con_lengths):
        enc_out, enc_hidden = self.context_encoder(torch.cat((image_features.unsqueeze(1), contexts[ind].unsqueeze(0)),
                                                             dim=1), torch.tensor(con_lengths[ind]).unsqueeze(0))  # b*3,100 ,b*100
        # hidden = img_features
        # context_features, hidden = self.context_encoder(contexts[ind].unsqueeze(0), torch.tensor(con_lengths[ind]).unsqueeze(0))
        sampled_ids = self.attn_decoder.sample(enc_out)
        return sampled_ids
