import torch
import torch.nn as nn
import wasabi


class Lstm2SeqEncoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        embedding: torch.nn.Embedding,
        dropout_value: float = 0.0,
        hidden_dim: int = 1024,
        bidirectional: bool = False,
        combine_strategy: str = "concat",
        rnn_bias: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        """
        :param emb_dim: type: int
        The embedding dimension of the embedding used in the first layer of the LSTM
        :param embedding: type: torch.nn.Embedding
        :param dropout_value: type: float
        :param hidden_dim: type: float
        :param bidirectional: type: bool
        Is the LSTM bi-directional
        :param combine_strategy: type: str
        This can be one of concat, sum, average
        This decides how different hidden states of different embeddings are combined
        :param rnn_bias: type: bool
        should RNN bias be used
        This is mainly for debugging purposes
        use sparingly
        """
        super(Lstm2SeqEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.embedding = embedding
        self.dropout_value = dropout_value
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.combine_strategy = combine_strategy
        self.rnn_bias = rnn_bias
        self.device = device
        self.num_directions = 2 if self.bidirectional else 1
        self.num_layers = 1
        self.allowed_combine_strategies = ["sum", "concat"]
        self.msg_printer = wasabi.Printer()

        assert (
            self.combine_strategy in self.allowed_combine_strategies
        ), self.msg_printer.fail(
            f"The combine strategies can be one of "
            f"{self.allowed_combine_strategies}. You passed "
            f"{self.combine_strategy}"
        )
        self.emb_dropout = nn.Dropout(p=self.dropout_value)
        self.rnn = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hidden_dim,
            bias=self.rnn_bias,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

    def forward(
        self,
        x: torch.LongTensor,
        c0: torch.FloatTensor = None,
        h0: torch.FloatTensor = None,
    ) -> torch.Tensor:
        batch_size, seq_length = x.size()

        # batch_size * time steps * embedding dimension
        embedded_tokens = self.embedding(x)
        embedded_tokens = self.emb_dropout(embedded_tokens)

        if h0 is None or c0 is None:
            h0, c0 = self.get_initial_hidden(batch_size=batch_size)

        # output = batch_size, sequence_length, hidden_dim * num_directions
        # h_n = num_layers * num_directions, batch_size, hidden_dimension
        # c_n = num_layers * num_directions, batch_size, hidden_dimension
        output, (_, _) = self.rnn(embedded_tokens, (h0, c0))

        if self.bidirectional:
            output = output.view(batch_size, seq_length, 2, -1)
            forward_output = output[:, :, 0, :]  # batch_size, seq_length, hidden_dim
            backward_output = output[:, :, 1, :]  # batch_size, seq_length, hidden_dim
            if self.combine_strategy == "concat":
                encoding = torch.cat([forward_output, backward_output], dim=2)
            elif self.combine_strategy == "sum":
                encoding = torch.add(forward_output, backward_output)
        else:
            encoding = output

        return encoding

    def get_initial_hidden(self, batch_size: int):
        h0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dim
        )
        c0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dim
        )
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        return h0, c0
