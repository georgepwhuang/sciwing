import torch
import torch.nn as nn
from typing import List, Union
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.line import Line
from sciwing.data.datasets_manager import DatasetsManager
from sciwing.modules.attentions.dot_product_attention import DotProductAttention


class SeriesEncoder(nn.Module, ClassNursery):
    def __init__(
        self,
        encoder: nn.Module,
        secondary_encoder: nn.Module = None,
        attention: nn.Module = DotProductAttention(),
        attention_type: str = None,
        window_size: int = 1,
        dilution_gap_size: int = 0,
        datasets_manager: DatasetsManager = None,
        device: Union[torch.device, str] = torch.device("cpu"),
    ):
        """ Generate.

        Parameters
        ----------
        encoder : nn.Module
            A main encoder for sentence-level embeddings
        secondary_encoder : nn.Module
            A secondary encoder for sentence-level embeddings of neighboring lines
        attention_type : str
            The following choices are offered: "sliding", "global"
        window_size: int
            Only used when attention_type is "sliding"
        dilution_gap_size: int
            Only used when attention_type is "sliding"
        """
        super(SeriesEncoder, self).__init__()
        self.encoder = encoder
        self.secondary_encoder = secondary_encoder
        self.datasets_manager = datasets_manager
        self.attention = attention
        self.attention_type = attention_type
        self.window_size = window_size
        self.dilution_gap_size = dilution_gap_size + 1
        self.device = device

    def forward(self, lines: List[List[Line]]):
        """

        Parameters
        ----------
        lines : List[List[Line]]
           A list of documents comprised of list of lines.

        Returns
        -------
        torch.FloatTensor
            Returns the concatenated embedding that is of the size
            batch_size, num_lines, hidden_dimension for lstm2vec
            batch_size, num_lines, seq_len, hidden_dimension for lstm2seq
            where the ``hidden_dimension`` twice the original

        """
        encodings = []
        for doc in lines:
            # num_lines, hidden_dimension for lstm2vec
            # num_lines, seq_len, hidden_dimension for lstm2seq
            query = self.encoder(doc)

            # num_lines, context_lines, hidden_dimension for lstm2vec
            # num_lines, seq_len, context_lines, hidden_dimension for lstm2seq
            if self.attention:
                if self.secondary_encoder:
                    secondary_encoding = self.secondary_encoder(doc)
                else:
                    secondary_encoding = query

                if self.attention == "sliding":
                    dimensions = secondary_encoding.size()
                    key_list = []
                    for i in range(-self.window_size * self.diluation_gap_size, 0, self.diluation_gap_size):
                        size = [int(abs(i))].extend(dimensions[1:])
                        zeros = torch.randn(size, device=self.device)
                        encoding = torch.cat([zeros, secondary_encoding[:-self.window_size * self.diluation_gap_size]],
                                             dim=0)
                        key_list.append(encoding)
                    for i in range(0, (self.window_size + 1) * self.diluation_gap_size, self.diluation_gap_size):
                        size = [int(abs(i))].extend(dimensions[1:])
                        zeros = torch.randn(size, device=self.device)
                        encoding = torch.cat([secondary_encoding[self.window_size * self.diluation_gap_size:], zeros],
                                             dim=0)
                        key_list.append(encoding)
                    key = torch.stack(key_list, dim=-2)
                elif self.attention == "global":
                    key = query.unsqueeze(-2)
                    dimensions = [-1] * len(key.size())
                    dimensions[-2] = len(doc)
                    key = query.expand(dimensions)

                # num_lines, context_lines for lstm2vec
                # num_lines, seq_len, context_lines for lstm2seq
                attn = self.attention(query_matrix=query, key_matrix=key)

                attn_unsqueeze = attn.unsqueeze(-2)

                # num_lines, 1, hidden_dimension for lstm2vec
                # num_lines, seq_len, 1, hidden_dimension for lstm2seq
                attn_encoding = torch.matmul(attn_unsqueeze, key)

                # num_lines, hidden_dimension for lstm2vec
                # num_lines, seq_len, hidden_dimension for lstm2seq
                attn_encoding = attn_encoding.squeeze(1)
                encoding = torch.cat([query, attn_encoding])
                encodings.append(encoding)

            else:
                encodings.append(query)

        # batch_size * num_lines, hidden_dimension for lstm2vec
        # batch_size * num_lines, seq_len, hidden_dimension for lstm2seq
        final_encoding = torch.cat(encodings, dim=0)
        return final_encoding
