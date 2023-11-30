class EarlyStopper:
    def __init__(self, early_stopping_metrics, patience=1):
        if not early_stopping_metrics in ["f1", "mcc"]:
            raise Exception(f"{early_stopping_metrics} is not supported yet.")
        self.early_stopping_metrics = early_stopping_metrics
        self.patience = patience
        self.counter = 0
        self.best_early_stopping_value = float('-inf')

    def is_early_stop(self, early_stopping_value):
        # only for f1 or mcc for now
        if early_stopping_value >= self.best_early_stopping_value:
            self.best_early_stopping_value = early_stopping_value
            self.counter = 0
        elif early_stopping_value < self.best_early_stopping_value:
            self.counter += 1
            if self.counter >= self.patience:
                print(
                    f"Early stopping reached, current best {self.early_stopping_metrics}: {self.best_early_stopping_value}"
                )
                return True
        return False
