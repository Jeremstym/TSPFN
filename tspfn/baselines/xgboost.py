import torch
import numpy as np
import xgboost as xgb
import csv
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassAUROC, 
    MulticlassF1Score, 
    MulticlassAveragePrecision,
    MulticlassCohenKappa,
    MulticlassRecall,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryAUROC,
    BinaryAccuracy,
    BinaryCohenKappa,
    BinaryRecall,
)

class XGBoostStaticBaseline(pl.LightningModule):
    def __init__(self, num_classes: int = 10, num_channels:int = 1, xgb_params: dict = None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_channels = num_channels
        print(f"num_classes: {num_classes}, num_channels: {num_channels}")
        # Optimized XGBoost params for large tabular data
        self.xgb_params = xgb_params or {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "tree_method": "hist",     # Change to "gpu_hist" for GPU speedup
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "verbosity": 0
        }
        
        # Metric Collection for clean evaluation
        if num_classes == 2:
            # Binary classification metrics
            metrics = MetricCollection({
                "acc": BinaryAccuracy(),
                "auroc": BinaryAUROC(),
                "f1": BinaryF1Score(),
                "auprc": BinaryAveragePrecision(),
                "cohen_kappa": BinaryCohenKappa(),
                "recall": BinaryRecall()
            })
        else:
            # Multiclass classification metrics
            metrics = MetricCollection({
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "auroc": MulticlassAUROC(num_classes=num_classes),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
                "auprc": MulticlassAveragePrecision(num_classes=num_classes),
                "cohen_kappa": MulticlassCohenKappa(num_classes=num_classes),
                "recall": MulticlassRecall(num_classes=num_classes, average="macro")
            })
        
        self.test_metrics = metrics.clone(prefix="test")
        self.clf = None

    def setup(self, stage=None):
        # Trigger fitting for both fit and validate stages
        if stage in ["fit", "validate", "test"] or stage is None:
            if self.clf is None:
                print("--- Fitting XGBoost on Training Data ---")
                # Access the underlying train_dataloader from the datamodule
                train_loader = self.trainer.datamodule.train_dataloader()
                
                all_x, all_y = [], []
                for batch in train_loader:
                    # Handle if train_loader is also a CombinedLoader or simple tuple
                    x, y = batch if isinstance(batch, (tuple, list)) else batch["train"]
                    batch_size, num_channels, seq_len = x.shape
                    if num_channels > 1:
                    # if self.num_channels > 1:
                        # Flatten
                        x = x.flatten(start_dim=1)
                    all_x.append(x.view(-1, x.size(-1)).cpu())
                    all_y.append(y.view(-1).cpu())
                
                X_train = torch.cat(all_x, dim=0).numpy()
                y_train = torch.cat(all_y, dim=0).numpy()
                print(f"--- Training Data Loaded: {X_train.shape[0]} samples ---")
                print(f"XGBoost Parameters: {self.xgb_params}")
                
                print(f"Support shape: {X_train.shape}, Labels shape: {y_train.shape}")
                self.clf = xgb.XGBClassifier(**self.xgb_params)
                self.clf.fit(X_train, y_train)
                print(f"--- XGBoost Fit Complete ({len(X_train)} samples) ---")
                
                if np.unique(y_train).shape[0] < self.num_classes:
                    print(f"Warning: Only {np.unique(y_train).shape[0]} unique classes found in training data, but num_classes is set to {self.num_classes}. Changing metrics")
                    self.num_classes = np.unique(y_train).shape[0]
                    # Reinitialize metrics with the correct number of classes
                    if self.num_classes == 2:
                        metrics = MetricCollection({
                            "acc": BinaryAccuracy(),
                            "auroc": BinaryAUROC(),
                            "f1": BinaryF1Score(),
                            "auprc": BinaryAveragePrecision(),
                            "cohen_kappa": BinaryCohenKappa(),
                            "recall": BinaryRecall()
                        })
                    else:
                        metrics = MetricCollection({
                            "acc": MulticlassAccuracy(num_classes=self.num_classes),
                            "auroc": MulticlassAUROC(num_classes=self.num_classes),
                            "f1": MulticlassF1Score(num_classes=self.num_classes, average="macro"),
                            "auprc": MulticlassAveragePrecision(num_classes=self.num_classes),
                            "cohen_kappa": MulticlassCohenKappa(num_classes=self.num_classes),
                            "recall": MulticlassRecall(num_classes=self.num_classes, average="macro")
                        })
                    self.test_metrics = metrics.clone(prefix="test")

    def test_step(self, batch, batch_idx):
        if self.clf is None:
            raise ValueError("XGBoost classifier has not been fitted yet.")
            
        batch_dict, _, _ = batch if isinstance(batch, (tuple, list)) else (batch, None, None)
        if "val" not in batch_dict:
            raise ValueError("Expected 'val' key in batch for validation data.")
            
        x, y = batch_dict["val"] 
        print(f"Original test batch shape: {x.shape}, labels shape: {y.shape}")
        batch_size, num_channels, seq_len = x.shape
        
        if num_channels > 1:
        # if self.num_channels > 1:
            x = x.flatten(start_dim=1)
        
        x_eval = x.view(-1, x.size(-1)).cpu().numpy()
        y_eval = y.view(-1).cpu()
        
        print(f"Query shape for XGBoost: {x_eval.shape}, Labels shape: {y_eval.shape}")
        y_probs = self.clf.predict_proba(x_eval)
        y_probs_ts = torch.tensor(y_probs, device=self.device)

        print(f"--- Test Step {batch_idx}: Evaluated {x_eval.shape[0]} samples ---")
        
        if self.num_classes == 2:
            # For binary classification, use probabilities of the positive class
            y_probs_ts = torch.softmax(y_probs_ts, dim=-1)[:, 1]
        self.test_metrics.update(y_probs_ts, y_eval.to(self.device))

    def on_test_epoch_end(self):
        output_data = []
        test_metrics = self.test_metrics.compute()
        for name, value in test_metrics.items():
            self.log(name, value, prog_bar=True, on_epoch=True, on_step=False)
            output_data.append({"metric": name, "value": value.item()})
        self.test_metrics.reset()

        with open("xgboost_test_metrics.csv", mode="w", newline="") as csvfile:
            fieldnames = ["metric", "value"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in output_data:
                writer.writerow(row)

    def configure_optimizers(self):
        # Dummy optimizer since XGBoost isn't trained via AdamW
        return torch.optim.Adam(self.parameters(), lr=1e-3)
