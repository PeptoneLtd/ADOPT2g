import copy
import math
import os
import random
from typing import List, Tuple, TypeVar

import numpy as np
import pandas as pd
import scipy
import sklearn.metrics
import torch
from scipy.special import logit
from sentence_transformers import InputExample, SentenceTransformer, losses
from skl2onnx import to_onnx
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from torch.utils.data import DataLoader

PandasDataFrame = TypeVar("pandas.core.frame.DataFrame")


"""
This script is a brief demo, that shows how to find ESM-based disorder predictors that use SentenceTransformers.
The idea is to concatenate the residue level representations with sequence level representations and use
these combined embeddings with a prediction head (a simple ElasticNet e.g.) to predict g-scores.

The residue level representations are extracted from the esm model "esm2_t33_650M_UR50D".

The sequence level representations are coming from a BiEncoder, which is based on a fine-tuned tiny esm model,
esm2_t6_8M_UR50D.

Here we rely on huggingface and use the esm ecosystem through huggingface.
"""

# prepare the data using the same train and test set that appear in the final version of the ADOPT paper
root_path = "../../"
training_set_json = os.path.join(root_path, "data/trizod_2024-05-09/train.json")
test_set_json = os.path.join(root_path, "data/trizod_2024-05-09/test.json")
index_col_name = "ID"
df_train = pd.read_json(training_set_json, dtype={index_col_name: str})
df_test = pd.read_json(test_set_json, dtype={index_col_name: str})


metric = "g-score"
train_datasets_str = "trizod_tolerant"

# just in case, remove any overlaps from the train set (1159)
overlaps = list(set(list(df_train[index_col_name])) & set(list(df_test[index_col_name])))
if len(overlaps) > 0:
    print(f"WARNING: {len(overlaps)} overlaps between the train and test sets. Removing these from the train set.")
df_cleared = df_train[~df_train[index_col_name].isin(overlaps)]


# bucket sequences in terms of sequence level disorder profiles - more details on this will be provided separately
# ----------------------------------------------------------
df_cleared[f"mean-{metric}"] = df_cleared[metric].apply(lambda x: np.nanmean(np.array(x, dtype=float)))
df_cleared[f"std-{metric}"] = df_cleared[metric].apply(lambda x: np.nanstd(np.array(x, dtype=float)))
df_cleared[f"nan_ratio-{metric}"] = df_cleared[metric].apply(lambda x: sum(np.isnan(np.array(x, dtype=float))) / len(x))
std_threshold = np.percentile(df_cleared[f"std-{metric}"].to_numpy(), 75)
mean_threshold = 0.5
nan_threshold = 0.5

print(f"std threshold: {std_threshold:g}")
print(f"mean threshold: {mean_threshold:g}")
print(f"nan ratio threshold: {nan_threshold:g}")


def mean_std_bucketing(
    df_row,
    metric=metric,
    mean_threshold=mean_threshold,
    std_threshold=std_threshold,
    nan_threshold=nan_threshold,
):
    if df_row[f"nan_ratio-{metric}"] > nan_threshold or df_row[f"std-{metric}"] > std_threshold:
        return 2  # uncertain
    if df_row[f"mean-{metric}"] < mean_threshold:
        return 0  # ordered
    else:
        return 1  # disordered


df_cleared[f"bucket-{metric}"] = df_cleared.apply(mean_std_bucketing, axis=1)
print(df_cleared[f"bucket-{metric}"].value_counts())

# bucketing the test set, i.e. df_117 is not relevant as we do not use this at this stage
df_test[f"mean-{metric}"] = df_test[metric].apply(lambda x: np.nanmean(np.array(x, dtype=float)))
df_test[f"std-{metric}"] = df_test[metric].apply(lambda x: np.nanstd(np.array(x, dtype=float)))
df_test[f"nan_ratio-{metric}"] = df_test[metric].apply(lambda x: sum(np.isnan(np.array(x, dtype=float))) / len(x))
df_test[f"bucket-{metric}"] = df_test.apply(mean_std_bucketing, axis=1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# SetFit approach
# ---------------
def sequence_pairs_generation(sequences: List[str], labels: List[int], pairs: List) -> List:
    # initialize two empty lists to hold the (sequence, sequence) pairs and
    # labels to indicate if a pair is positive or negative

    numClassesList = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in numClassesList]

    for idxA in range(len(sequences)):
        currentSequence = sequences[idxA]
        label = labels[idxA]
        idxB = np.random.choice(idx[np.where(numClassesList == label)[0][0]])
        posSequence = sequences[idxB]
        # prepare a positive pair and update the sentences and labels
        # lists, respectively
        pairs.append(InputExample(texts=[currentSequence, posSequence], label=1.0))

        negIdx = np.where(labels != label)[0]
        negSequence = sequences[np.random.choice(negIdx)]
        # prepare a negative pair of sequences and update our lists
        pairs.append(InputExample(texts=[currentSequence, negSequence], label=0.0))

        # return a 2-tuple of our sequence pairs and labels
    return pairs


seq_col = "sequence"
category_col = f"bucket-{metric}"


def pedestrian_input_new(
    indexes: List[str],
    df: PandasDataFrame,
    path: str,
    st_repr: List,
    g_col="g-score",
    huggingface=False,
    drop_missing=True,
    add_st_repr=True,
) -> Tuple:
    """
    Prepares the input using residue and also sequence level representations by concatenating them

    More detail later. Clean up will be required as well.
    """

    zeds = []
    exes = []

    for k in range(len(indexes)):
        seq = df[df[index_col_name] == indexes[k]].sequence.item()
        if huggingface == True:
            repr_esm = torch.load(f"{path}{indexes[k]}.pt", weights_only=False)["last_hidden_state"][0].clone().cpu().detach()
        else:
            repr_esm = torch.load(f"{path}{indexes[k]}.pt", weights_only=False)["representations"][33].clone().cpu().detach()
        g_s = np.array(df[df[index_col_name] == indexes[k]][g_col].to_numpy()[0], dtype=float)
        if drop_missing == True:
            idxs = np.where(~np.isnan(g_s))[0]
        else:
            idxs = np.arange(len(g_s))

        for i in idxs:
            zeds.append(g_s[i])
            if add_st_repr:
                exes.append(np.concatenate([repr_esm[i].numpy(), st_repr[seq]]))
            else:
                exes.append(repr_esm[i].numpy())

    return np.array(exes), np.array(zeds)


seed_list = [57]
sigmoid_transform = True
rebalance = True

res = {}
# @title SetFit
st_model = "facebook/esm2_t6_8M_UR50D"  # @param Other esm models available in huggingface could be used as well
num_training = min(
    512, df_cleared[f"bucket-{metric}"].value_counts().min()
)  # @param ["8", "16", "32", "54", "128", "256", "512"] {type:"raw"}
num_itr = 5  # @param ["1", "2", "3", "4", "5", "10"] {type:"raw"}


os.makedirs(os.path.join(root_path, "data/onnx"), exist_ok=True)
os.makedirs(os.path.join(root_path, "data/st"), exist_ok=True)

for _seed in seed_list:
    set_seed(_seed)
    # Equal samples per class training
    train_df_sample = pd.concat(
        [df_cleared[df_cleared[category_col] == k].sample(num_training) for k in df_cleared[category_col].unique()],
    )
    x_train = train_df_sample[seq_col].values.tolist()
    y_train = train_df_sample[category_col].values.tolist()

    train_examples = []
    for x in range(num_itr):
        train_examples = sequence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)

    # orig_model = SentenceTransformer(st_model)
    model = SentenceTransformer(st_model)

    # S-BERT adaptation
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=24)
    train_loss = losses.CosineSimilarityLoss(model)

    num_epochs = 4
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

    # train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    # prepare the data
    ex_train, zed_train = {}, {}
    ex_test, zed_test = {}, {}

    # first the sequence level representations extracted from the fine-tuned BiEncoder
    train_st_repr = dict(zip(df_cleared["sequence"], model.encode(list(df_cleared["sequence"]))))
    test_st_repr = dict(zip(df_test["sequence"], model.encode(list(df_test["sequence"]))))

    for model_type in ["esm-st"]:
        ex_train[model_type], zed_train[model_type] = pedestrian_input_new(
            list(df_cleared[index_col_name]),
            df_cleared,
            os.path.join(root_path, "data/esm2/train/"),
            train_st_repr,
            g_col="g-score",
            huggingface=True,
            drop_missing=True,
            add_st_repr=True,
        )
        # assemble the test data from the 117 set
        ex_test[model_type], zed_test[model_type] = pedestrian_input_new(
            list(df_test[index_col_name]),
            df_test,
            os.path.join(root_path, "data/esm2/test/"),
            test_st_repr,
            g_col="g-score",
            huggingface=True,
            drop_missing=True,
            add_st_repr=True,
        )

    X_train = copy.deepcopy(ex_train["esm-st"])
    y_train = copy.deepcopy(zed_train["esm-st"])
    X_test = copy.deepcopy(ex_test["esm-st"])
    y_test = copy.deepcopy(zed_test["esm-st"])

    if rebalance:
        print("Rebalancing the training set...")
        a, b = 1.5, 3.5  # adjust depending on the dataset
        weights = np.minimum(1, a / (1 + np.exp(-b * (y_train - 0.5))))
        assert min(weights) > 0, "better not to remove training data"

        try:
            import matplotlib.pyplot as plt

            plt.title(f"order / disorder ratio: {weights[y_train < 0.5].sum() / weights[y_train > 0.5].sum():g}")
            plt.hist(y_train, bins=100, alpha=0.5, label="original")
            plt.hist(y_train, weights=weights, bins=100, alpha=0.5, label="weighted")
            plt.xlim(0, 1)
            plt.yscale("log")
            plt.legend(loc="upper center")
            plt.xlabel("G-score")
            plt.ylabel("Histogram")
            plt.gca().twinx()
            plt.plot(y_train, weights, ",", color="g", label="weights")
            plt.legend(loc="upper right")
            plt.savefig("rebalancing_weights.png", bbox_inches="tight")
        except ImportError:
            print("Could not plot the rebalancing weights, matplotlib is not available.")
    else:
        weights = None

    if sigmoid_transform:
        print("Applying sigmoid transformation to the g-scores...")
        y_train[y_train == 0] = 1e-5 * np.random.rand(len(y_train[y_train == 0]))
        y_train[y_train == 1] -= 1e-5 * np.random.rand(len(y_train[y_train == 1]))
        y_train = logit(y_train)
        y_test[y_test == 0] = 1e-5 * np.random.rand(len(y_test[y_test == 0]))
        y_test[y_test == 1] -= 1e-5 * np.random.rand(len(y_test[y_test == 1]))
        y_test = logit(y_test)

    print("Fitting prediction head...")
    regr = ElasticNet(alpha=0.0001, max_iter=500000, tol=0.0005, precompute=True, random_state=_seed)  # 40,48,49
    # regr = HistGradientBoostingRegressor(random_state=_seed, verbose=1)  # 41
    # regr = HistGradientBoostingRegressor(random_state=_seed, verbose=1, learning_rate=0.01, max_iter=1000, early_stopping=True)  # 46,47
    # regr = RandomForestRegressor(random_state=_seed, verbose=1)  # 42
    # regr = Ridge(random_state=_seed) #43
    regr.fit(X_train, y_train, sample_weight=weights)

    # metrics
    print("Evaluating the model...")
    predict_skl = regr.predict(X_test)
    r2_score_skl = sklearn.metrics.r2_score(y_test, predict_skl)
    print(f"min: {np.min(predict_skl):g} ({np.min(y_test):g}), max: {np.max(predict_skl):g} ({np.max(y_test):g})")
    print("r2 with skl:", r2_score_skl)
    print(
        f"MAE/std with skl - {sklearn.metrics.mean_absolute_error(y_test, predict_skl):.4f} / {np.std(abs(y_test - predict_skl)):.4f}",
    )
    # correlation
    print(
        f"Seed {_seed} ElasticNet - Correlation between the predicted and the ground"
        f" truth on the test set:  {scipy.stats.spearmanr(y_test, predict_skl).correlation:.4f}",
    )
    print("---------------------------------")

    onx = to_onnx(regr, X_train[:1])  # onnx cannot export a TransformedTargetRegressor
    with open(f"{root_path}/data/onnx/{_seed}.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    model.save(
        path=f"{root_path}/data/st/{_seed}",
        model_name=f"adopt2-gscore-seed-{_seed}",
        train_datasets=[train_datasets_str],
    )

    res[_seed] = {
        "seed": _seed,
        "r2": r2_score_skl,
        "MAE": sklearn.metrics.mean_absolute_error(y_test, predict_skl),
        "corr": scipy.stats.spearmanr(y_test, predict_skl).correlation,
    }

df_res = pd.DataFrame.from_dict(res).T
df_res.to_csv(os.path.join(root_path, "data/res_run.csv"), index=False)
