import itertools
import os
import sys

import numpy as np
import onnxruntime as rt
from scipy.special import expit
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# sys.path.insert(1, '/home/mambauser/adopt2')
root_path = os.path.abspath("../../")
sys.path.insert(1, root_path)

model_seed = 57
sigmoid_transform = True
print(f"Using model seed: {model_seed}, sigmoid_transform={sigmoid_transform}")

"""
This script contains the building blocks of the model and inference 
"""

# load esm2 that will be needed for the residue level representations
# -------------------------------------------------------------------
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# the pooling layer is not used, and current transformers cannot load it in 8-bit
model_8bit = AutoModel.from_pretrained(
    model_name,
    add_pooling_layer=False,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
)

# load the BiEncoder model used for the sequence level representations
# --------------------------------------------------------------------
st_model = SentenceTransformer(os.path.join(root_path, f"data/st/{model_seed}"))

# load the onnx session for inference
# -----------------------------------
sess = rt.InferenceSession(
    os.path.join(root_path, f"data/onnx/{model_seed}.onnx"),
    providers=["TensorrtExecutionProvider"],
)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


class AdoptTwo:
    tokenizer = tokenizer
    esm_model_8bit = model_8bit
    sigmoid_transform = sigmoid_transform
    input_name = input_name
    label_name = label_name

    def chunk(self, seqs):
        """
        Generator that yields chunks (lists) of strings where the total
        characters per chunk is < max_chars.
        """
        current_chunk = []
        current_length = 0

        for s in seqs:
            s_len = len(s)

            # If adding this string would exceed the limit, yield current chunk
            if current_chunk and current_length + s_len >= self._max_sequence_len:
                yield current_chunk
                current_chunk = [s]
                current_length = s_len
            else:
                current_chunk.append(s)
                current_length += s_len

        # Yield the last chunk if it exists
        if current_chunk:
            yield current_chunk

    def __init__(self, embedder, prediction_head):
        self.embedder = embedder
        self.prediction_head = prediction_head
        # same value as the size of the old position embedding table, which current transformers removed
        self._max_sequence_len = self.esm_model_8bit.config.max_position_embeddings - 2

    def get_embeddings(self, sequences):
        container = []

        for k in self.chunk(sequences):
            seqs = k
            inputs = self.tokenizer(seqs, return_tensors="pt", padding=True)
            # older transformers moved the inputs to the GPU automatically, current versions do not
            inputs = {key: value.to(self.esm_model_8bit.device) for key, value in inputs.items()}

            try:
                outputs = self.esm_model_8bit(**inputs)
                biencoder_outputs = self.embedder.encode(seqs)
            except:
                n_resid = 0
                for seq in seqs:
                    n_resid += len(seq)
                print(f"seqs: {seqs}\n n_seqs: {len(seqs)}\n n_resid: {n_resid}")
                raise
            for i, seq in enumerate(k):
                residue_level_embs = outputs.last_hidden_state[i, 1 : 1 + len(seq), ...].clone().detach().cpu().numpy()
                sequence_level_embs = np.tile(biencoder_outputs[i, ...], (len(seq), 1))

                container.append(np.concatenate([residue_level_embs, sequence_level_embs], axis=1))

        return container

    def get_scores(self, embeddings):
        if self.sigmoid_transform:
            return [
                expit(self.prediction_head.run([self.label_name], {self.input_name: x})[0].squeeze()).tolist()
                for x in embeddings
            ]
        else:
            return [
                self.prediction_head.run([self.label_name], {self.input_name: x})[0].squeeze().tolist()
                for x in embeddings
            ]

    def predict_disorder_no_nan(self, sequences):
        """Relies on tokenizer to handle non standard amino acids"""
        results = []

        # deal with input sequences shorter than max size
        short_seqs = [seq for seq in sequences if len(seq) <= self._max_sequence_len]
        short_scores = self.get_scores(self.get_embeddings(short_seqs))

        min_overlap = self._max_sequence_len // 2
        j = 0
        for sequence in sequences:
            if len(sequence) <= self._max_sequence_len:
                # we computed the scores a few lines above
                results.append(short_scores[j])
                j += 1
            else:
                # split sequence in chunks of max_sequence_len overlapping by half
                num_chunks = int(np.ceil((len(sequence) - min_overlap) / (self._max_sequence_len - min_overlap)))
                overlap = int(np.floor((self._max_sequence_len * num_chunks - len(sequence)) / (num_chunks - 1)))
                chunks = []
                for i in range(num_chunks):
                    start = i * (self._max_sequence_len - overlap)
                    end = min(len(sequence), start + self._max_sequence_len)
                    chunks.append(sequence[start:end])

                # compute scores
                preds = np.zeros(len(sequence), dtype=float)
                counts = np.zeros(len(sequence), dtype=int)
                long_scores = self.get_scores(self.get_embeddings(chunks))
                for i in range(num_chunks):
                    start = i * (self._max_sequence_len - overlap)
                    end = min(len(sequence), start + self._max_sequence_len)
                    preds[start:end] += long_scores[i]
                    counts[start:end] += 1
                # average on the overlapping sections
                preds /= counts
                results.append(preds.tolist())
        assert len(sequences) == len(results)
        return results

    def predict_disorder(self, sequences):
        """If non standard amino acids are present, replace them with 'X' run the prediction and set np.nan as gscore"""
        standard_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        non_std_masks = []
        for i, seq in enumerate(sequences):
            seq_arr = np.array(list(seq))
            non_std_masks.append(np.flatnonzero(~np.isin(seq_arr, list(standard_amino_acids))))
            if len(non_std_masks[-1]) > 0:
                seq_arr[non_std_masks[-1]] = "X"
                sequences[i] = "".join(seq_arr)

        results = self.predict_disorder_no_nan(sequences)
        for i, mask in enumerate(non_std_masks):
            for m in mask:
                results[i][m] = np.nan
        return results


# example usage
# -------------
# seq_examples = ['GQGGGTHNQWNKPSKPKTNMKHMAGAAAAGAVVGGLGGYMLGSAMSRPMMHFGNDWEDRYYRENMNRYPNQVYYRPVDQYNNQNNFVHDCVNITIKQHTVTTTTKGENFTETDIKIMERVVEQMCTTQYQKESQAYYDGRRS',
#                 'MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQPRGRRQPIPKARRPEGRTWAQPGYPHHHHHH',
#                 'GTQNRPLLRNSLDDLVGPPSNLEGQSDERALLDQLHTLLSNTDATGLEEIDRALGIPELVNQGQALEPKQD',
#                 'GESGERWARQVSGGHFVTLHGAERLEEETNDEDVSDIERRIAMRLAERRQEDSATHGDE']

# adopt2 = AdoptTwo(st_model, sess)
# disorder_scores = adopt2.predict_disorder(seq_example)
