from sentence_transformers import SentenceTransformer

# Download once (locally)
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("local_model")
