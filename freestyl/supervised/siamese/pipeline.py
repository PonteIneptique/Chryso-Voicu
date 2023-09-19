import warnings
from typing import Optional, List, Dict, Tuple
from sklearn import preprocessing
from pytorch_lightning import Trainer

from freestyl.supervised.base import BaseSupervisedPipeline
from freestyl.supervised.siamese.utils import train_dataframewrappers, train_k_folds, get_df_prediction, \
    score_from_preds, PairsDataframe, ScoreDataframe
from freestyl.dataset.dataframe_wrapper import DataframeWrapper

from freestyl.supervised.siamese.features.model import SiameseFeatureModule
from freestyl.supervised.siamese.sequential.model import SiameseSequentialModule


class SiamesePipeline(BaseSupervisedPipeline):
    def __init__(self, accelerator: Optional[str] = "cpu"):
        super(SiamesePipeline, self).__init__()
        self._pipeline = None
        self._hparams = {}
        self._features: Optional[List[str]] = None
        self._le: preprocessing.LabelEncoder = preprocessing.LabelEncoder()
        self.accelerator: str = accelerator
        self.models: Dict[str, SiameseFeatureModule] = {}
        self.comparison_sets: Dict[str, Tuple[DataframeWrapper, DataframeWrapper]] = {}

    def build(
            self,
            dimension: int = 128,
            margin: float = 1.0,
            learning_rate: float = 1e-2,
            loss: Optional[str] = "contrastive",
            batch_size: int = 8,
            patience: int = 30
    ):
        """
                dataframe: DataframeWrapper,
        ks: int = 5,
        sample: bool = False,
        accelerator: Optional[str] = "cuda",
        **hyperparams

            features: Iterable[str], label_encoder: LabelEncoder,
            dim: int = 128, margin: float = 1.0, lr: float = 1e-2,
            loss: Optional[str] = "contrastive"
        """
        self._hparams = {
            "model": {
                "dimension": dimension,
                "margin": margin,
                "loss": loss,
                "learning_rate": learning_rate,
            },
            "training": {
                "batch_size": batch_size,
                "patience": patience
            }
        }

    def fit(
        self,
        data: DataframeWrapper,
        dev: Optional[DataframeWrapper] = None,
        dev_ratio: float = .1,
        model_name: str = 'Fit',
        sample: bool = False,
        *args,
        **kwargs
    ) -> "SiamesePipeline":
        """ Function to fit the Siamese Module

        :param data: Training data
        :param dev: Optional development set
        :param dev_ratio: If dev is not provided, dev_ratio is taken out of data for training
        :param model_name: Name of the model in the Pipeline
        """
        if dev is None:
            data, dev = data.split(ratio=dev_ratio)
        result_model = train_dataframewrappers(
            train=data,
            dev=dev,
            accelerator=self.accelerator,
            sample=sample,
            **self._hparams["training"],
            **self._hparams["model"]
        )
        self.models[model_name] = result_model
        return self

    def get_a_model(self, model: Optional[str] = None):
        if model is None:
            model = next(iter(self.models.keys()))
            warnings.warn(f"No model was provided, using the first found in the Pipeline named `{model}`.")
        elif model not in self.models:
            raise ValueError(f"Model {model} unknown. Available models: {','.join(self.models.keys())}")
        return self.models[model]

    def test(
            self,
            data: DataframeWrapper,
            comparator: Optional[DataframeWrapper] = None,
            model: Optional[str] = None,
            threshold: float = .5
    ) -> Tuple[ScoreDataframe, PairsDataframe]:
        model = self.get_a_model(model)

        predictions, _ = get_df_prediction(
            trainer=Trainer(gpus=0, accelerator=self.accelerator),
            model=model,
            compared=data,
            comparator=comparator,
            threshold=threshold
        )

        return score_from_preds(predictions), predictions

    def cross_validate(
        self,
        data: DataframeWrapper,
        ks: int = 5,
        sample: bool = False,
        keep_sets: bool = True,
        threshold: float = .5,
        *args,
        **kwargs
    ) -> Tuple[ScoreDataframe, PairsDataframe]:
        """ Function to fit the Siamese Module

        :param data: Training data
        :param ks: Number of folding to use
        :param sample: Apply a sampler for batching
        :param keep_sets: Stores the sets for comparisons
        :param threshold: Threshold for classification of pairs
        """
        accumulated_pairs: Optional[PairsDataframe] = None
        accumulated_scores: Optional[ScoreDataframe] = None

        for (idx, (model, ktrain, ktest)) in enumerate(train_k_folds(
            data,
            ks=ks,
            sample=sample,
            accelerator=self.accelerator,
            **self._hparams["model"])
        ):
            self.models[f"KFold{idx}"] = model
            if keep_sets:
                self.comparison_sets[f"KFold{idx}"] = (ktrain, ktest)
            # Do tests, record the results
            pairs, _ = get_df_prediction(
                trainer=Trainer(gpus=0, accelerator=self.accelerator),
                model=model,
                compared=ktest,
                comparator=ktrain,
                threshold=threshold,
                k=idx
            )
            score = score_from_preds(pairs, k=idx)
            if accumulated_pairs is None:
                accumulated_scores = score
                accumulated_pairs = pairs
            else:
                accumulated_scores = accumulated_scores.append(score)
                accumulated_pairs = accumulated_pairs.append(pairs)

        # Return DataFrame of results
        return accumulated_scores, accumulated_pairs

    def predict(
        self,
        data: DataframeWrapper,
        comparator: Optional[DataframeWrapper] = None,
        threshold: float = .5,
        model: Optional[str] = None,
        *args,
        **kwargs
    ):
        model = self.get_a_model(model)
        pairs, _ = get_df_prediction(
            trainer=Trainer(gpus=0, accelerator=self.accelerator),
            model=model,
            compared=data,
            comparator=comparator,
            threshold=threshold,
            k=-1
        )
        return pairs


if __name__ == "__main__":
    import pandas as pd
    from loc_script import read_feature_frame
    from freestyl.dataset.dataframe_wrapper import DataframeWrapper

    data = read_feature_frame("/home/tclerice/dev/chrysostylom/data/02-ngrams/fwords.csv", min_words=2000)
    data.reset_index(inplace=True)
    data.set_index("Titre", inplace=True)
    #data.head(2)
    #data["Target"] = data.Auteur  # data.Auteur.apply(lambda x: "Non-Chrysostome" if x != "Chrysostome" else x)
    #data.head(2)
    # data = data.sample(1.) # Shuffle the data !
    data["Target"] = data.Auteur.apply(lambda x: x if "Pseudo" not in x else "Spuria")
    print(data.head())
    keep = pd.read_csv("/home/tclerice/dev/chrysostylom/03-GT.csv", index_col="Titre").index.tolist()
    test = data.loc[~data.index.isin(keep), :].reset_index()
    print(f"Before filtering: {data.shape[0]} texts, after {data.loc[keep, :].shape[0]}")
    data = data.loc[keep, :]
    data.reset_index(inplace=True)
    data.head()
    data = data[data.Target != "Spuria"]
    data = DataframeWrapper(data, target="Target", label="Titre", x_ignore=["Auteur"])
    data.drop_low(documents_min=.05, frequency_min=1000)
    # data.dropna()
    data.xs.head(2)
    print(len(data.features))
    data.normalized.make_relative(inplace=True)

    test = DataframeWrapper(test, target="Target", label="Titre", x_ignore=["Auteur"])
    test.align_to(data)

    pipeline = SiamesePipeline(accelerator="cpu")
    pipeline.build(dimension=128, learning_rate=1e-4, loss="contrastive", margin=1)
    pipeline.fit(data, dev_ratio=.1, sample=False, patience=5)
    print(pipeline.test(test, data))
