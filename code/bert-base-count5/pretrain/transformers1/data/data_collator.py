from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NewType, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
from ..tokenization_utils import PreTrainedTokenizer


class DataCollator(ABC):
    """
    A `DataCollator` is responsible for batching
    and pre-processing samples of data as requested by the training loop.
    """

    @abstractmethod
    def collate_batch(self) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.

        Returns:
            A dictionary of tensors
        """
        pass


InputDataClass = NewType("InputDataClass", Any)


@dataclass
class DefaultDataCollator(DataCollator):
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.
    """

    def collate_batch(self, features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        # In this method we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        first = features[0]

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if hasattr(first, "label") and first.label is not None:
            if type(first.label) is int:
                labels = torch.tensor([f.label for f in features], dtype=torch.long)
            else:
                labels = torch.tensor([f.label for f in features], dtype=torch.float)
            batch = {"labels": labels}
        elif hasattr(first, "label_ids") and first.label_ids is not None:
            if type(first.label_ids[0]) is int:
                labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
            else:
                labels = torch.tensor([f.label_ids for f in features], dtype=torch.float)
            batch = {"labels": labels}
        else:
            batch = {}

        # Handling of all other possible attributes.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in vars(first).items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        return batch


@dataclass
class DataCollatorForLanguageModeling(DataCollator):
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def collate_batch(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens7(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            return {"input_ids": batch, "labels": batch}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def mask_tokens2(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        inputs = inputs.numpy()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        probability_matrix = probability_matrix.numpy()
        labels = labels.numpy()
        for i in range(len(probability_matrix)):
            for j in range(len(probability_matrix[0])):
                if probability_matrix[i][j] == np.float32(0.15):
                    if random.random() > 0.85:
                        if random.random() > 0.2:
                            inputs[i][j] = self.tokenizer.mask_token_id
                        elif random.random() > 0.5:
                            inputs[i][j] = random.randint(5, len(self.tokenizer) - 1)
                        else:
                            pass
                    else:
                        labels[i][j] = np.float32(-100)
                else:
                    labels[i][j] = np.float32(-100)

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        return inputs, labels


    def mask_tokens3(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        inputs = inputs.numpy()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        probability_matrix = probability_matrix.numpy()
        labels = labels.numpy()
        covered = set()
        for i in range(len(probability_matrix)):
            for j in range(len(probability_matrix[0])):
                if probability_matrix[i][j] == np.float32(0.15) and (i,j) not in covered:
                    if random.random() > 0.85:
                        if random.random() > 0.2:
                            if random.random() > 0.85:
                                for k in range(j,min(j+5,len(probability_matrix[0]))):
                                    if probability_matrix[i][k] == np.float32(0.15) and (i,k) not in covered:
                                        inputs[i][k] = self.tokenizer.mask_token_id
                                        covered.add((i,k))
                            elif random.random() > 0.7647:
                                for k in range(j,min(j+4,len(probability_matrix[0]))):
                                    if probability_matrix[i][k] == np.float32(0.15) and (i,k) not in covered:
                                        inputs[i][k] = self.tokenizer.mask_token_id
                                        covered.add((i,k))
                            elif random.random() > 0.5384:
                                for k in range(j,min(j+3,len(probability_matrix[0]))):
                                    if probability_matrix[i][k] == np.float32(0.15) and (i,k) not in covered:
                                        inputs[i][k] = self.tokenizer.mask_token_id
                                        covered.add((i,k))
                            elif random.random() > 0.42857:
                                for k in range(j,min(j+2,len(probability_matrix[0]))):
                                    if probability_matrix[i][k] == np.float32(0.15) and (i,k) not in covered:
                                        inputs[i][k] = self.tokenizer.mask_token_id
                                        covered.add((i,k))
                            else:
                                inputs[i][j] = self.tokenizer.mask_token_id
                                covered.add((i,j))

                        elif random.random() > 0.5:
                            inputs[i][j] = random.randint(5, len(self.tokenizer) - 1)
                        else:
                            pass
                    else:
                        labels[i][j] = np.float32(-100)
                else:
                    labels[i][j] = np.float32(-100)

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        return inputs, labels

    def mask_tokens4(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        inputs = inputs.numpy()
        ids = [i for i in range(len(inputs))]
        random.shuffle(ids)
        inputs = inputs[ids]
        inputs = torch.from_numpy(inputs)

        labels = inputs.clone()
        inputs = inputs.numpy()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        total_token = 0
        for i in range(len(probability_matrix)):
            for j in range(len(probability_matrix[0])):
                if probability_matrix[i][j] == np.float32(0.15):
                    total_token += 1

        cur_token = 0
        probability_matrix = probability_matrix.numpy()
        labels = labels.numpy()
        covered = set()
        ngramFlag = True
        for i in range(len(probability_matrix)):
            if cur_token > total_token * 0.03:
                ngramFlag = False
            for j in range(len(probability_matrix[0])):
                if probability_matrix[i][j] == np.float32(0.15) and (i,j) not in covered:
                    if random.random() > 0.85:
                        if random.random() > 0.2:
                            if random.random() > 0.9 and ngramFlag:
                                for k in range(j,min(j+4,len(probability_matrix[0]))):
                                    if probability_matrix[i][k] == np.float32(0.15) and (i,k) not in covered:
                                        inputs[i][k] = self.tokenizer.mask_token_id
                                        covered.add((i,k))
                                        cur_token += 1
                            elif random.random() > 0.222 and ngramFlag:
                                for k in range(j,min(j+3,len(probability_matrix[0]))):
                                    if probability_matrix[i][k] == np.float32(0.15) and (i,k) not in covered:
                                        inputs[i][k] = self.tokenizer.mask_token_id
                                        covered.add((i,k))
                                        cur_token += 1
                            elif random.random() > 0.42857 and ngramFlag:
                                for k in range(j,min(j+2,len(probability_matrix[0]))):
                                    if probability_matrix[i][k] == np.float32(0.15) and (i,k) not in covered:
                                        inputs[i][k] = self.tokenizer.mask_token_id
                                        covered.add((i,k))
                                        cur_token += 1
                            else:
                                inputs[i][j] = self.tokenizer.mask_token_id
                                covered.add((i,j))
                                cur_token += 1

                        elif random.random() > 0.5:
                            inputs[i][j] = random.randint(5, len(self.tokenizer) - 1)
                            cur_token += 1
                        else:
                            pass
                    else:
                        labels[i][j] = np.float32(-100)
                else:
                    labels[i][j] = np.float32(-100)

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        return inputs, labels

    def mask_tokens5(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        inputs = inputs.numpy()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        covered = set()
        pvals = [0.4, 0.3, 0.2, 0.1]
        ngrams = np.arange(1, 5, dtype=np.int64)

        probability_matrix = probability_matrix.numpy()
        labels = labels.numpy()
        for i in range(len(probability_matrix)):
            cur_token = 0
            total_token = 0
            for j in range(len(probability_matrix[0])):
                if probability_matrix[i][j] == np.float32(0.15):
                    total_token += 1
            choose = random.randint(0, 1)
            if choose == 0:
                startIndex = 0
                endIndex = np.argwhere(inputs[i] == np.float32(2))[-1][0]
            elif choose == 1:
                startIndex = np.argwhere(inputs[i] == np.float32(2))[-1][0]
                endIndex = np.argwhere(inputs[i] == np.float32(3))[-1][0]

            valid_j = [index for index in range(startIndex, endIndex + 1)]

            for j in range(len(probability_matrix[0])):
                if cur_token < total_token * 0.15:
                    if probability_matrix[i][j] == np.float32(0.15):
                        n = np.random.choice(ngrams, p=pvals)
                        for k in range(n):
                            if j + k >= len(probability_matrix[0]):
                                break
                            if (i, j+k) in covered:
                                continue
                            if j+k in valid_j:
                                if random.random() > 0.7:
                                    if random.random() > 0.2:
                                        if probability_matrix[i][j+k] == np.float32(0.15):
                                            inputs[i][j+k] = self.tokenizer.mask_token_id
                                            covered.add((i, j + k))
                                            cur_token += 1

                                    elif random.random() > 0.5:
                                        if probability_matrix[i][j + k] == np.float32(0.15):
                                            inputs[i][j+k] = random.randint(5, len(self.tokenizer) - 1)
                                            covered.add((i, j + k))
                                            cur_token += 1

                                    else:
                                        if probability_matrix[i][j + k] == np.float32(0.15):
                                            covered.add((i, j + k))
                                            cur_token += 1

                                else:
                                    labels[i][j] = np.float32(-100)
                            else:
                                labels[i][j] = np.float32(-100)
                    else:
                        labels[i][j] = np.float32(-100)
                else:
                    labels[i][j] = np.float32(-100)

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        return inputs, labels

    def mask_tokens6(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        inputs = inputs.numpy()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        covered = set()


        probability_matrix = probability_matrix.numpy()
        labels = labels.numpy()
        for i in range(len(probability_matrix)):
            cur_token = 0
            total_token = 0
            for j in range(len(probability_matrix[0])):
                if probability_matrix[i][j] == np.float32(0.15):
                    total_token += 1
            for j in range(len(probability_matrix[0])):
                if cur_token > total_token*0.15:
                    break
                if probability_matrix[i][j] == np.float32(0.15):
                    if random.random() > 0.85:
                        if random.random() > 0.2:
                            if random.random() > 0.9:
                                for k in range(j, min(j + 4, len(probability_matrix[0]))):
                                    if probability_matrix[i][k] == np.float32(0.15) and (i, k) not in covered:
                                        inputs[i][k] = self.tokenizer.mask_token_id
                                        covered.add((i, k))
                                        cur_token += 1
                            elif random.random() > 0.222:
                                for k in range(j, min(j + 3, len(probability_matrix[0]))):
                                    if probability_matrix[i][k] == np.float32(0.15) and (i, k) not in covered:
                                        inputs[i][k] = self.tokenizer.mask_token_id
                                        covered.add((i, k))
                                        cur_token += 1
                            elif random.random() > 0.42857:
                                for k in range(j, min(j + 2, len(probability_matrix[0]))):
                                    if probability_matrix[i][k] == np.float32(0.15) and (i, k) not in covered:
                                        inputs[i][k] = self.tokenizer.mask_token_id
                                        covered.add((i, k))
                                        cur_token += 1
                            else:
                                inputs[i][j] = self.tokenizer.mask_token_id
                                covered.add((i, j))
                                cur_token += 1

                        elif random.random() > 0.5:
                            inputs[i][j] = random.randint(5, len(self.tokenizer) - 1)
                            cur_token += 1
                        else:
                            cur_token += 1

                    else:
                        labels[i][j] = np.float32(-100)


                else:
                    labels[i][j] = np.float32(-100)

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        return inputs, labels


    def mask_tokens7(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        inputs = inputs.numpy()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        covered = set()
        ngrams = np.arange(1, 3 + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, 3 + 1)
        pvals /= pvals.sum(keepdims=True)

        probability_matrix = probability_matrix.numpy()
        labels = labels.numpy()
        for i in range(len(probability_matrix)):
            cur_token = 0
            total_token = 0
            for j in range(len(probability_matrix[0])):
                if probability_matrix[i][j] == np.float32(0.15):
                    total_token += 1
            for j in range(len(probability_matrix[0])):
                if cur_token <= total_token * 0.15:
                    n = np.random.choice(ngrams, p=pvals)
                    if probability_matrix[i][j] == np.float32(0.15):
                        for k in range(n):
                            if j + k >= len(probability_matrix[0]):
                                break
                            if (i, j+k) in covered:
                                continue
                            if random.random() > 0.85:
                                if random.random() > 0.2:
                                    if probability_matrix[i][j+k] == np.float32(0.15):
                                        inputs[i][j+k] = self.tokenizer.mask_token_id
                                        covered.add((i, j + k))
                                        cur_token += 1

                                elif random.random() > 0.5:
                                    if probability_matrix[i][j + k] == np.float32(0.15):
                                        inputs[i][j+k] = random.randint(5, len(self.tokenizer) - 1)
                                        covered.add((i, j + k))
                                        cur_token += 1

                                else:
                                    if probability_matrix[i][j + k] == np.float32(0.15):
                                        covered.add((i, j + k))
                                        cur_token += 1

                            else:
                                labels[i][j] = np.float32(-100)

                    else:
                        labels[i][j] = np.float32(-100)
                else:
                    labels[i][j] = np.float32(-100)

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        return inputs, labels

