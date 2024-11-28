from google.cloud import aiplatform
from transformers import BertTokenizer

def make_inference(
    text: str, 
    tokenizer: BertTokenizer,
    max_length: int,
    endpoint: aiplatform.models.Endpoint
    ) -> aiplatform.models.Prediction:
  
  inference_features_tf = tokenizer(
      text,
      padding=True, truncation=True,
      return_tensors="tf",
      max_length=max_length
      )
  instances = [
      {
          'input_ids': inference_features_tf.get('input_ids').numpy().tolist()[0],
          'token_type_ids': inference_features_tf.get('token_type_ids').numpy().tolist()[0],
          'attention_mask': inference_features_tf.get('attention_mask').numpy().tolist()[0]
      }
  ]

  prediction = endpoint.predict(instances=instances)
  return prediction
    