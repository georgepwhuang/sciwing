import torch.nn as nn
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.contextual_lines import LineWithContext
from sciwing.modules.lstm2vecencoder import LSTM2VecEncoder
from typing import List, Union
import torch


class Lstm2VecAttnContextEncoder(nn.Module, ClassNursery):
    def __init__(
        self,
        main_encoder: LSTM2VecEncoder,
        context_encoder: nn.Module,
        attn_module: nn.Module,
        device: Union[torch.device, str] = torch.device("cpu"),
    ):
        """
        This module uses a lstm2vec encoder. The hidden dimensions can be
        enhanced by attention over some context for every line. Consider the context
        as something that is an additional information for the line. You can
        refer to LineWithContext for more information


        Parameters
        ----------
        main_encoder : Lstm2VecEncoder
            Encodes the lines using lstm encoders to get contextual representations
        context_encoder : nn.Module
            Encodes the contextual lines using encoders
        attn_module : nn.Module
            You can use attention modules from sciwing.modules.attention
        device: Union[torch.device, str]
            The device on which this encoder is run
        """
        super(Lstm2VecAttnContextEncoder, self).__init__()
        self.main_encoder = main_encoder
        self.context_encoder = context_encoder
        self.attn_module = attn_module
        self.device = device

    def forward(self, lines: List[LineWithContext]) -> torch.Tensor:
        main_lines = []
        for line in lines:
            main_lines.append(line.line)

        # batch_size, hidden_dimension
        main_encoding = self.main_encoder(lines=main_lines)

        # batch_size, max_num_context_lines, hidden_dimension
        max_num_context_lines = max([len(line.context_lines) for line in lines])
        context_encoding = []
        for line in lines:
            context_lines = line.context_lines
            num_context_lines = len(context_lines)
            encoding = self.context_encoder(lines=context_lines)
            # num_context_lines, embedding_dimension
            emb_dim = encoding.size(1)

            # adding zeros for padding
            padding_length = max_num_context_lines - num_context_lines
            zeros = torch.randn(padding_length, emb_dim, device=self.device)

            encoding = torch.cat([encoding, zeros], dim=0)
            context_encoding.append(encoding)

        context_encoding = torch.stack(context_encoding)

        # batch_size, number_of_context_lines
        attn = self.attn_module(query_matrix=main_encoding, key_matrix=context_encoding)

        attn_unsqueeze = attn.unsqueeze(1)

        # batch_size, 1, hidden_dimension
        attn_unsqueeze = torch.bmm(attn_unsqueeze, context_encoding)

        attn_encoding = attn_unsqueeze.squeeze(1)

        # concatenate the representation
        # batch_size, hidden_dimension
        final_encoding = torch.cat([main_encoding, attn_encoding], dim=1)

        return final_encoding
