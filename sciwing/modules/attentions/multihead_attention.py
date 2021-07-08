import torch.nn as nn
import torch
from sciwing.utils.class_nursery import ClassNursery


class MultiHeadAttention(nn.Module, ClassNursery):
    def __init__(self, encode_dim: int, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=encode_dim, num_heads=num_heads)

    def forward(
        self, query_matrix: torch.Tensor, key_matrix: torch.Tensor, to_seq: bool = False
    ) -> torch.Tensor:
        """ Calculates the attention over the key

        Parameters
        ----------
        query_matrix: torch.Tensor
            Shape (batch_size, hidden_dimension) or (batch_size, seq_len, hidden_dimension)
        key_matrix: torch.Tensor
            Shape (batch_size, max_number_of_time_steps, hidden_dimension) or
            (batch_size, seq_len, max_number_of_time_steps, hidden_dimension)
        to_seq: bool
            Existence of seq_len dimension in query and key matrices

        Returns
        -------
        torch.Tensor
            The attention distribution over the keys
        """

        if to_seq:
            assert query_matrix.size(1) == key_matrix.size(1)

            times = query_matrix.size(1)

            attention_list = []

            for i in range(times):
                query = query_matrix[:, i, :]
                query = query.transpose(0, 1)
                key = key_matrix[:, i, :, :]
                key = key.transpose(0, 1)
                attention = self.get_attention_weight(query, key)
                attention_list.append(attention)

            attention = torch.stack(attention_list, dim=1)

        else:
            # (batch_size, max_number_of_time_steps)
            attention = self.get_attention_weight(query_matrix, key_matrix)

        return attention

    def get_attention_weight(self, query, key):
        # (1, batch_size, hidden_dimension)
        query = query.unsqueeze(-2)
        query = query.transpose(0, 1)

        # (max_number_time_steps, batch_size, hidden_dimension)
        key = key.transpose(0, 1)

        # (batch_size, 1, max_number_time_steps)
        _, attention = self.attention.forward(query, key, key)

        # (batch_size, max_number_of_time_steps)
        attention = attention.squeeze(-2)

        return attention
