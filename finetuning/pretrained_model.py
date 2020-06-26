import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from finetuning.config_reforBert import ReforBertConfig
BertLayerNorm = nn.LayerNorm


class ReforBertPreTrainedModel(PreTrainedModel):
  """ An abstract class to handle weights initialization and
      a simple interface for downloading and loading pretrained models.
  """

  config_class = ReforBertConfig
  base_model_prefix = "reforbert"

  def _init_weights(self, module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, BertLayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()


