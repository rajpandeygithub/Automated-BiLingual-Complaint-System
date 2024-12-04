from typing import List
from kfp.dsl import component, Input, Artifact

@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        'pandas==1.5.3',
        'numpy==1.26.4'
        ]
)
def select_best_model(metrics_artifacts: Input[List[Artifact]]) -> str:
    import pandas as pd
    import json

    metrics_data = []
    for artifact in metrics_artifacts:
        with open(artifact.path, 'r') as f:
            metrics_data.append(json.load(f))

    df = pd.DataFrame(metrics_data)
    # Assuming 'f1' is the metric of interest
    best_model_row = df.loc[df['f1'].idxmax()]
    best_model_name = best_model_row['huggingface_model_name']

    print(f"The best model is: {best_model_name} with F1 score: {best_model_row['f1']}")
    return best_model_name