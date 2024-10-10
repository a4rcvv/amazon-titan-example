import json
import numpy as np
import boto3


bedrock_client = boto3.client("bedrock-runtime", "ap-northeast-1")
model_id = "amazon.titan-embed-text-v1"

texts = [
    "今日はとても天気が良く、散歩日和でした。",
    "空は晴れていて、風も心地よかったので公園を散歩しました。",
    "今日は会議がたくさんあってとても忙しかったです。",
    "会議が続いていたので、少し疲れました。",
]


def embed_text(text: str):
    """embed text using bedrock model"""
    body = {
        "inputText": text
    }
    response = bedrock_client.invoke_model(
        body=json.dumps(body), modelId=model_id, contentType="application/json", accept="application/json"
    )
    response_body =  json.loads(response.get("body").read())
    return response_body


def embed_to_ndarray(text: str) -> np.ndarray:
    """embed text and convert to numpy array"""
    response = embed_text(text)
    return np.array(response["embedding"])

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """calculate cosine similarity between two vectors"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def main():
    similarities = []
    embeddings = [embed_to_ndarray(text) for text in texts]
    for i, e1 in enumerate(embeddings):
        for j, e2 in enumerate(embeddings):
            if i <= j:
                continue
            similarity = cosine_similarity(e1, e2)
            similarities.append((i, j, similarity))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    for i, j, similarity in similarities:
        print(f"{texts[i]} vs {texts[j]}: {similarity}")
            


if __name__ == "__main__":
    main()