from typing import List, NamedTuple
from kfp.dsl import component, Input, Output, Artifact, Dataset, Model

@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        'pandas==1.5.3',
        'numpy==1.26.4'
        ]
)
def select_best_model(
    metrics_artifacts: Input[List[Artifact]], 
    test_datasets: Input[List[Dataset]], 
    models: Input[List[Model]],
    best_model: Output[Model],
    best_model_test_data: Output[Dataset]
    ) -> NamedTuple('Outputs', [('best_model_name', str), ('best_f1_score', float)]):
    import pandas as pd
    import json
    from collections import namedtuple

    metrics_data = []
    for metric in metrics_artifacts:
        with open(metric.path, 'r') as f:
            metrics_data.append(json.load(f))

    df = pd.DataFrame(metrics_data)
    best_metric_row_idx = df['f1'].idxmax()
    best_metric_row = df.loc[best_metric_row_idx]
    
    best_model.uri = models[best_metric_row_idx].uri
    best_model_test_data.uri = test_datasets[best_metric_row_idx].uri

    output = namedtuple('Outputs', ['best_model_name', 'best_f1_score'])
    return output(best_metric_row['huggingface_model_name'], best_metric_row['f1'])