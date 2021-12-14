import pandas as pd
import numpy as np
from datetime import datetime

def create_output(predictions):
    """
    Function that creates output csv submission file. Returns the submission dataframe.
    Input:
    - predictions: an array with shape (n_predictions, 1) with the ARRIVAL_DELAY outputs (ordered).
    Return:
    - submission: dataframe with id and ARRIVAL_DELAY fields.
    """
    submission = pd.DataFrame(data={"id": np.arange(len(predictions)),
                                    "ARRIVAL_DELAY": np.reshape(predictions, (len(predictions)))})
    filename = f"../submissions/submission_{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}.csv"
    submission.to_csv(filename, index=False)
    return submission
