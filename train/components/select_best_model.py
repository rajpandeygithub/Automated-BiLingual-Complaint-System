from typing import List, NamedTuple
from kfp.dsl import component, Input, Output, Artifact, Dataset, Model

@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        'pandas==1.5.3',
        'numpy==1.26.4',
        'requests==2.32.3'
        ]
)
def select_best_model(
    metrics_artifacts: Input[List[Artifact]], 
    test_datasets: Input[List[Dataset]], 
    models: Input[List[Model]],
    best_model: Output[Model],
    best_model_test_data: Output[Dataset],
    slack_url: str = None,
    ) -> NamedTuple('Outputs', [('best_model_name', str), ('best_f1_score', float)]):
    import pandas as pd
    import json
    import requests
    from collections import namedtuple
    from datetime import datetime

    def send_slack_message(
        webhook_url: str,
        message_str: str,
        execution_date: str, 
        execution_time: str, 
        duration: str,
        is_success: bool,
        ):
    
        if is_success:
            color = "#36a64f"
            pretext = f":large_green_circle: {message_str}"
        else:
            color = "FF0000"
            pretext = f":large_red_circle: {message_str}"

        message = {
            "attachments": [
                {
                    "color": color,  # Green color for success
                    "pretext": pretext,
                    "fields": [
                        {
                            "title": "Component Name",
                            "value": "Get Data KubeFlow Component",
                            "short": True
                        },
                        {
                            "title": "Execution Date",
                            "value": str(execution_date),
                            "short": True
                        },
                        {
                            "title": "Execution Time",
                            "value": str(execution_time),
                            "short": True
                        },
                        {
                            "title": "Duration",
                            "value": f"{duration} minutes",
                            "short": True
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(webhook_url, json=message)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(e)
    
    start_time = datetime.now()

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

    if slack_url:
       send_slack_message(
          webhook_url=slack_url, message_str=f'KubeFlow Component: Select Best Model Component | Model: {best_metric_row["huggingface_model_name"]} Sucess',
          execution_date=start_time.date(), execution_time=start_time.time(), 
          duration=(datetime.now() - start_time).total_seconds() / 60, is_success=True
          )

    return output(best_metric_row['huggingface_model_name'], best_metric_row['f1'])