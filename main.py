from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

qa_dataset = load_dataset("squad_v2")
print("QA Sample:", qa_dataset["train"][0])

sentiment_dataset = load_dataset("sst2")
print("Sentiment Sample:", sentiment_dataset["train"][0])

qa_dataset["train"] = qa_dataset["train"].select(range(1000))
qa_dataset["validation"] = qa_dataset["validation"].select(range(200))
sentiment_dataset["train"] = sentiment_dataset["train"].select(range(1000))
sentiment_dataset["validation"] = sentiment_dataset["validation"].select(range(200))
qa_dataset.save_to_disk("./data/qa_data")
sentiment_dataset.save_to_disk("./data/sentiment_data")