import torch
import torch.nn as nn
from sciwing.modules.attentions.multihead_attention import MultiHeadAttention
from typing import List, Union
import wasabi
import sciwing.constants as constants
from sciwing.utils.class_nursery import ClassNursery
from sciwing.data.line import Line
from transformers import AutoTokenizer
from transformers import AutoModel

PATHS = constants.PATHS
EMBEDDING_CACHE_DIR = PATHS["EMBEDDING_CACHE_DIR"]


class BertSentenceEncoder(nn.Module, ClassNursery):
    def __init__(
        self,
        dropout_value: float = 0.1,
        aggregation_type: str = "sum",
        layers: Union[str, List[int]] = "last_four",
        embedding_method: str = "attn_pool",
        transformer: str = "bert-base-cased",
        device: Union[torch.device, str] = torch.device("cpu"),
    ):
        super(BertSentenceEncoder, self).__init__()

        self.dropout_value = dropout_value
        self.aggregation_type = aggregation_type
        if isinstance(layers, str):
            if layers == "all":
                self.layers = list(range(1, 13))
            elif layers == "last_four":
                self.layers = [9, 10, 11, 12]
            elif layers == "last":
                self.layers = [12]
            elif layers == "second_to_last":
                self.layers = [11]
            else:
                raise ValueError(f"The layer specification {layers} is not supported")
        else:
            self.layers = layers
        self.embedding_method = embedding_method
        self.transformer = transformer
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.msg_printer = wasabi.Printer()
        self.embedder_name = self.transformer

        self.scibert_foldername_mapping = {
            "scibert-base-cased": "scibert_basevocab_cased",
            "scibert-sci-cased": "scibert_scivocab_cased",
            "scibert-base-uncased": "scibert_basevocab_uncased",
            "scibert-sci-uncased": "scibert_scivocab_uncased",
        }

        if self.transformer in self.scibert_foldername_mapping.keys():
            foldername = self.scibert_foldername_mapping[self.transformer]
            self.model_type_or_folder_url = "allenai/" + foldername

        else:
            self.model_type_or_folder_url = self.transformer

        # load the bert model
        with self.msg_printer.loading(" Loading Bert tokenizer and model. "):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type_or_folder_url,
                                                           use_fast=False,
                                                           do_basic_tokenize=False)
            self.model = AutoModel.from_pretrained(self.model_type_or_folder_url)

        self.attention = MultiHeadAttention(encode_dim=self.model.config.hidden_size)

        self.msg_printer.good(f"Finished Loading {self.transformer} model and tokenizer")
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
        transformer_tokens_lengths = []
        for line in lines:
            text = line.text
            tokenized_text = self.tokenizer.tokenize(text)
            line.tokenizers[self.embedder_name] = self.tokenizer
            line.add_tokens(tokens=tokenized_text, namespace=self.embedder_name)
            transformer_tokens_lengths.append(len(tokenized_text))
            assert len(tokenized_text) != 0, f"{line.text}"

        tokenized_input = self.tokenizer([line.text for line in lines],
                                         padding=True, truncation=True,
                                         return_tensors="pt").to(self.device)

        output = self.model(**tokenized_input, output_hidden_states=True)

        encoded_layers = output.hidden_states

        if "base" in self.transformer:
            assert len(encoded_layers) == 13, f"{len(encoded_layers)}"
        elif "large" in self.transformer:
            assert len(encoded_layers) == 25, f"{len(encoded_layers)}"

        # batch_size, num_bert_layers, max_len_bert + 2, bert_hidden_dimension
        filtered_layers = [encoded_layers[layer] for layer in self.layers]
        filtered_layers = torch.stack(filtered_layers, dim=1)

        # batch_size, num_bert_layers, bert_hidden_dimension
        if self.embedding_method == "CLS":
            sentence_encoding = filtered_layers[:, :, 0, :]
        elif self.embedding_method == "mean_pool":
            sentence_encoding = filtered_layers
            attention_mask = tokenized_input.attention_mask
            input_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(-1).expand(sentence_encoding.size()).float()
            sum_embeddings = torch.sum(sentence_encoding * input_mask_expanded, dim=2)
            sum_mask = torch.clamp(input_mask_expanded.sum(2), min=1e-9)
            sentence_encoding = sum_embeddings / sum_mask
        elif self.embedding_method == "max_pool":
            sentence_encoding = filtered_layers
            attention_mask = tokenized_input.attention_mask
            input_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(-1).expand(sentence_encoding.size()).float()
            sentence_encoding[input_mask_expanded == 0] = -1e9
            sentence_encoding = torch.max(sentence_encoding, dim=2)
        elif self.embedding_method == "attn_pool":
            sentence_encoding = []
            attention_mask = tokenized_input.attention_mask
            for idx, embedding in enumerate(filtered_layers):
                query = embedding[:, 0, :]
                key = embedding[:, 1: transformer_tokens_lengths[idx] + 1, :]
                assert transformer_tokens_lengths[idx] == int(torch.sum(attention_mask[idx])), \
                    f'token count{transformer_tokens_lengths[idx]} and ' \
                    f'attention mask {int(torch.sum(attention_mask[idx]))} does not match'
                attn = self.attention(query_matrix=query, key_matrix=key)
                attn_unsqueeze = attn.unsqueeze(-2)
                attn_encoding = torch.matmul(attn_unsqueeze, key)
                single_encoding = attn_encoding.squeeze(-2)
                sentence_encoding.append(single_encoding)
            sentence_encoding = torch.stack(sentence_encoding, dim=0)
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
            return self.model.config.hidden_size * len(self.layers)
        else:
            return self.model.config.hidden_size
