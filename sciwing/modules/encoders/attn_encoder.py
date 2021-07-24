import torch
import torch.nn as nn
from typing import List, Union
import itertools

from sciwing.data.doc_lines import DocLines
from sciwing.utils.class_nursery import ClassNursery


class AttnEncoder(nn.Module, ClassNursery):
    def __init__(
        self,
        encoder: nn.Module,
        attention: nn.Module = None,
        attention_type: str = None,
        window_size: int = 3,
        dilution_gap: int = 0,
        device: Union[torch.device, str] = torch.device("cpu"),
    ):
        """ Generate.

        Parameters
        ----------
        encoder : nn.Module
            A main encoder for sentence-level embeddings
        attention_type : str
            The following choices are offered: "sliding", "global"
        window_size: int
            Only used when attention_type is "sliding"
        dilution_gap: int
            Only used when attention_type is "sliding"
        """
        super(AttnEncoder, self).__init__()
        self.encoder = encoder
        self.attention = attention
        self.attention_type = attention_type
        self.window_size = int((window_size - 1) / 2.0)
        self.dilution_gap = dilution_gap + 1
        self.overlap = self.window_size * self.dilution_gap
        self.device = device

    def forward(self, lines: List[DocLines]):
        """

        Parameters
        ----------
        lines : List[DocLines]
           A list of documents comprised of list of lines.

        Returns
        -------
        torch.FloatTensor
            Returns the concatenated embedding that is of the size
            batch_size, num_lines, hidden_dimension for lstm2vec
            batch_size, num_lines, seq_len, hidden_dimension for lstm2seq
            where the ``hidden_dimension`` twice the original

        """

        if bool(self.attention) and self.attention_type == "sliding":
            encodings = []
            main_size = [len(doc.lines) for doc in lines]
            begin_size = [len(doc.begin) for doc in lines]
            end_size = [len(doc.end) for doc in lines]

            main_lines = [doc.lines for doc in lines]
            begin_lines = [doc.begin for doc in lines]
            end_lines = [doc.end for doc in lines]

            main_lines = list(itertools.chain.from_iterable(main_lines))
            begin_lines = list(itertools.chain.from_iterable(begin_lines))
            end_lines = list(itertools.chain.from_iterable(end_lines))

            all_size = [len(main_lines), len(begin_lines), len(end_lines)]
            all_lines = list(itertools.chain.from_iterable([main_lines, begin_lines, end_lines]))

            # batch * num_lines, hidden_dimension for lstm2vec
            # batch * num_lines, seq_len, hidden_dimension for lstm2seq
            all_encodings = self.encoder(all_lines)

            main_encodings, begin_encodings, end_encodings = torch.split(all_encodings, all_size)

            main_encodings_list = torch.split(main_encodings, main_size)
            begin_encodings_list = torch.split(begin_encodings, begin_size)
            end_encodings_list = torch.split(end_encodings, end_size)

            for main_encoding, begin_encoding, end_encoding in \
                    zip(main_encodings_list, begin_encodings_list, end_encodings_list):
                # num_lines, context_lines, hidden_dimension for lstm2vec
                # num_lines, seq_len, context_lines, hidden_dimension for lstm2seq
                key_list = []
                for i in range(-self.overlap, 0, self.dilution_gap):
                    encoding = torch.cat([begin_encoding[i:], main_encoding[:i]], dim=0)
                    key_list.append(encoding)
                for i in range(self.dilution_gap, self.overlap + self.dilution_gap, self.dilution_gap):
                    encoding = torch.cat([main_encoding[i:], end_encoding[:i]], dim=0)
                    key_list.append(encoding)
                key = torch.stack(key_list, dim=-2)

                # num_lines, context_lines for lstm2vec
                # num_lines, seq_len, context_lines for lstm2seq
                attn = self.attention(query_matrix=main_encoding, key_matrix=key)

                attn_unsqueeze = attn.unsqueeze(-2)

                # num_lines, 1, hidden_dimension for lstm2vec
                # num_lines, seq_len, 1, hidden_dimension for lstm2seq
                attn_encoding = torch.matmul(attn_unsqueeze, key)

                # num_lines, hidden_dimension for lstm2vec
                # num_lines, seq_len, hidden_dimension for lstm2seq
                attn_encoding = attn_encoding.squeeze(-2)
                encoding = torch.cat([main_encoding, attn_encoding], dim=-1)
                encodings.append(encoding)

            # batch_size * num_lines, hidden_dimension for lstm2vec
            # batch_size * num_lines, seq_len, hidden_dimension for lstm2seq
            final_encoding = torch.cat(encodings, dim=0)

        elif bool(self.attention) and self.attention_type == "global":
            encodings = []
            main_lines = [doc.lines for doc in lines]
            main_lines = list(itertools.chain.from_iterable(main_lines))
            main_encodings = self.encoder(main_lines)
            main_size = [len(doc.lines) for doc in lines]
            main_encodings_list = torch.split(main_encodings, main_size)
            for query, num_lines in zip(main_encodings_list, main_size):
                key = query.unsqueeze(-2)
                dimensions = [-1] * len(key.size())
                dimensions[-2] = num_lines
                key = query.expand(dimensions)
                attn = self.attention(query_matrix=query, key_matrix=key)
                attn_unsqueeze = attn.unsqueeze(-2)

                # num_lines, 1, hidden_dimension for lstm2vec
                # num_lines, seq_len, 1, hidden_dimension for lstm2seq
                attn_encoding = torch.matmul(attn_unsqueeze, key)

                # num_lines, hidden_dimension for lstm2vec
                # num_lines, seq_len, hidden_dimension for lstm2seq
                attn_encoding = attn_encoding.squeeze(-2)
                encoding = torch.cat([query, attn_encoding], dim=-1)
                encodings.append(encoding)
            final_encoding = torch.cat(encodings, dim=0)

        else:
            main_lines = [doc.lines for doc in lines]
            main_lines = list(itertools.chain.from_iterable(main_lines))
            final_encoding = self.encoder(main_lines)
        return final_encoding
