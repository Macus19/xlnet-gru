import torch.nn as nn
from transformers.modeling_xlnet import XLNetPreTrainedModel, XLNetModel, SequenceSummary
from torch.nn import CrossEntropyLoss, MSELoss

class XLNetForSequenceClassification_GRU(XLNetPreTrainedModel):
    def __init__(self, config, args):
        super(XLNetForSequenceClassification_GRU,self).__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.gru = []
        for i in range(args.gru_layers):
            self.gru.append(nn.GRU(config.d_model if i == 0 else args.gru_hidden_size*4, args.gru_hidden_size, num_layers=1, bidirectional=True, batch_first=True).cuda())
        self.gru = nn.ModuleList(self.gru)   
        self.logits_proj = nn.Linear(args.gru_hidden_size*2, config.num_labels)
        self.imprisonment_logits_proj = nn.Linear(args.gru_hidden_size*2, 12)
        self.init_weights()
    

    def forward(self, input_ids=None, attention_mask=None,mems=None, perm_mask=None, target_mapping=None, token_type_ids=None, input_mask=None, head_mask=None, inputs_embeds=None, labels=None, imprisonment_labels=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=target_mapping, token_type_ids=token_type_ids, input_mask=input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds)
        last_hidden_state = transformer_outputs[0]

        for gru in self.gru:
            try:
                gru.flatten_parameters()
            except:
                pass
            output, h_n = gru(last_hidden_state)
        x = h_n.permute(1, 0, 2).reshape(input_ids.size(0), -1).contiguous()

        logits = self.logits_proj(x)
        imprisonment_logits = self.imprisonment_logits_proj(x)
        outputs = (logits,) + (imprisonment_logits,) + transformer_outputs[1:]

        if labels is not None and imprisonment_labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss_1 = loss_fct(logits.view(-1), labels.view(-1))
                loss_2 = loss_fct(imprisonment_logits.view(-1), imprisonment_labels.view(-1))
                loss = loss_1 + loss_2
            else:
                loss_fct = CrossEntropyLoss()
                loss_1 = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss_2 = loss_fct(imprisonment_logits.view(-1, 12), imprisonment_labels.view(-1))
                loss = loss_1 + loss_2
            outputs = (loss,) + outputs
        return outputs
