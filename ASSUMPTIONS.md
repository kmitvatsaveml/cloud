# Assumptions (build-time decisions)
- The build sandbox blocked pytorch.org, Hugging Face, YouTube and LangSmith. So:
  - Embeddings run `BAAI/bge-small-en-v1.5` through **fastembed** (ONNX) instead of sentence-transformers + torch. A hashed bag-of-words embedder is the automatic fallback.
  - The FlashRank reranker falls back to a lexical-overlap reranker when its model can't be downloaded.
  - `data/samples/` holds two **synthetic, paraphrased** sample transcripts (clearly labeled). Real fetching uses yt-dlp + youtube-transcript-api on your machine.
  - LangSmith tracing turns on only when `LANGSMITH_API_KEY` is set.
- No ANTHROPIC_API_KEY was available, so **offline mode** uses project templates (`agent/templates/`). With a key, Claude writes the code.
- Generated projects default to Python + Flask (simplest stack that demos well, no paid services).
- The plant-disease demo uses an interpretable color-segmentation classifier as the MVP baseline. A CNN trained on PlantVillage is a stretch goal.
- Per-course notes are stored per persona slug (one course = one playlist = one slug).
