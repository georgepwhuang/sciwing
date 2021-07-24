import torch
import torch.nn as nn
from sciwing.tokenizers.bert_tokenizer import TokenizerForBert
from sciwing.numericalizers.transformer_numericalizer import NumericalizerForTransformer
from sciwing.modules.attentions.multihead_attention import MultiHeadAttention
from typing import List, Union
import wasabi
import sciwing.constants as constants
from pytorch_pretrained_bert import BertModel
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.line import Line
import os

PATHS = constants.PATHS
EMBEDDING_CACHE_DIR = PATHS["EMBEDDING_CACHE_DIR"]


class BertSentenceEncoder(nn.Module, ClassNursery):
    def __init__(
        self,
        dropout_value: float = 0.0,
        aggregation_type: str = "sum",
        embedding_method: str = "CLS",
        bert_type: str = "scibert-base-uncased",
        device: Union[torch.device, str] = torch.device("cpu"),
    ):
        super(BertSentenceEncoder, self).__init__()

        self.dropout_value = dropout_value
        self.aggregation_type = aggregation_type
        self.embedding_method = embedding_method
        self.bert_type = bert_type
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.msg_printer = wasabi.Printer()
        self.embedder_name = bert_type

        self.scibert_foldername_mapping = {
            "scibert-base-cased": "scibert_basevocab_cased",
            "scibert-sci-cased": "scibert_scivocab_cased",
            "scibert-base-uncased": "scibert_basevocab_uncased",
            "scibert-sci-uncased": "scibert_scivocab_uncased",
        }

        if "scibert" in self.bert_type:
            foldername = self.scibert_foldername_mapping[self.bert_type]
            self.model_type_or_folder_url = os.path.join(
                EMBEDDING_CACHE_DIR, foldername, "weights.tar.gz"
            )

        else:
            self.model_type_or_folder_url = self.bert_type

        # load the bert model
        with self.msg_printer.loading(" Loading Bert tokenizer and model. "):
            self.bert_tokenizer = TokenizerForBert(
                bert_type=self.bert_type, do_basic_tokenize=False
            )
            self.bert_numericalizer = NumericalizerForTransformer(
                tokenizer=self.bert_tokenizer
            )
            self.model = BertModel.from_pretrained(self.model_type_or_folder_url)
            self.model.to(self.device)

        self.attention = MultiHeadAttention(encode_dim=self.model.config.hidden_size)

        self.msg_printer.good(f"Finished Loading {self.bert_type} model and tokenizer")
        self.dimension = self.get_dimension()

    def forward(self, lines: List[Line]) -> torch.Tensor:
        """

        Parameters
        ----------
        lines : List[Line]
            A list of lines

        Returns
        -------
        torch.Tensor
            The bert embeddings for all the words in the instances
            The size of the returned embedding is ``[batch_size, emb_dim]``

        """

        # word_tokenize all the text string in the batch
        bert_tokens_lengths = []
        for line in lines:
            text = line.text
            bert_tokenized_text = self.bert_tokenizer.tokenize(text)
            line.tokenizers[self.embedder_name] = self.bert_tokenizer
            line.add_tokens(tokens=bert_tokenized_text, namespace=self.embedder_name)
            bert_tokens_lengths.append(len(bert_tokenized_text))

        max_len_bert = max(bert_tokens_lengths)
        # pad the tokenized text to a maximum length
        indexed_tokens = []
        segment_ids = []
        for line in lines:
            bert_tokens = line.tokens[self.embedder_name]
            tokens_numericalized = self.bert_numericalizer.numericalize_instance(
                instance=bert_tokens
            )
            tokens_numericalized = self.bert_numericalizer.pad_instance(
                numericalized_text=tokens_numericalized,
                max_length=max_len_bert + 2,
                add_start_end_token=True,
            )
            segment_numbers = [0] * len(tokens_numericalized)

            tokens_numericalized = torch.LongTensor(tokens_numericalized)
            segment_numbers = torch.LongTensor(segment_numbers)

            indexed_tokens.append(tokens_numericalized)
            segment_ids.append(segment_numbers)

        tokens_tensor = torch.stack(indexed_tokens)
        segment_tensor = torch.stack(segment_ids)

        tokens_tensor = tokens_tensor.to(self.device)
        segment_tensor = segment_tensor.to(self.device)

        encoded_layers, _ = self.model(tokens_tensor, segment_tensor)

        if "base" in self.bert_type:
            assert len(encoded_layers) == 12
        elif "large" in self.bert_type:
            assert len(encoded_layers) == 24

        # batch_size, num_bert_layers, max_len_bert + 2, bert_hidden_dimension
        last_four_layers = torch.stack(encoded_layers[-4:], dim=1)

        # batch_size, num_bert_layers, bert_hidden_dimension
        if self.embedding_method == "CLS":
            sentence_encoding = last_four_layers[:, :, 0, :]
        elif self.embedding_method == "mean_pool":
            sentence_encoding = torch.mean(last_four_layers, dim=2)
        elif self.embedding_method == "max_pool":
            sentence_encoding = torch.max(last_four_layers, dim=2)
        elif self.embedding_method == "attn_pool":
            query = last_four_layers[:, :, 0, :]
            key = last_four_layers[:, :, 1:-1, :]
            attn = self.attention(query_matrix=query, key_matrix=key, augmented=True)
            attn_unsqueeze = attn.unsqueeze(-2)
            attn_encoding = torch.matmul(attn_unsqueeze, key)
            sentence_encoding = attn_encoding.squeeze(-2)
        else:
            raise ValueError(f"The embedding type {self.embedding_method} does not exist")

        # batch_size, bert_hidden_dimension
        if self.aggregation_type == "sum":
            encoding = torch.sum(sentence_encoding, dim=1)
        elif self.aggregation_type == "mean":
            encoding = torch.mean(sentence_encoding, dim=1)
        elif self.aggregation_type == "concat":
            encoding = torch.cat(sentence_encoding, dim=1)
        else:
            raise ValueError(f"The aggregation type {self.aggregation_type} does not exist")

        return encoding

    def get_dimension(self) -> int:
        if self.aggregation_type == "concat":
            return self.model.config.hidden_size * 4
        else:
            return self.model.config.hidden_size
