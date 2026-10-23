"""MLflow wrapper so the registered model takes raw applicant columns and returns probability and flag."""

import mlflow.pyfunc

from loan_lib.scoring import score


class LoanDefaultModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, cutoff):
        self.model = model
        self.cutoff = cutoff

    def predict(self, context, model_input, params=None):
        return score(self.model, self.cutoff, model_input)
