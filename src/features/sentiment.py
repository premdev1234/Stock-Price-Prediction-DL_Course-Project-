import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class FinBERTSentimentAnalyzer:
    """
    PyTorch-based FinBERT sentiment extractor.
    Produces numerical sentiment embeddings per text.
    """

    def __init__(self, device=None):
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.model = AutoModel.from_pretrained("yiyanghkust/finbert-tone").to(self.device)
        self.model.eval()

    def batch_sentiment_analysis(self, text_list, batch_size=32):
        scores = []

        with torch.no_grad():
            for i in range(0, len(text_list), batch_size):
                batch = text_list[i:i+batch_size]

                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256
                ).to(self.device)

                outputs = self.model(**inputs)

                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                batch_scores = cls_embeddings.mean(dim=1).cpu().numpy()

                scores.extend(batch_scores)

        return np.array(scores)
