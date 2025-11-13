from typing import List, Dict
import tiktoken
import nltk

# Lazy/optional import for sentence-transformers to allow non-semantic mode usage
try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore


class SemanticChunker:
    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 50,
        similarity_threshold: float = 0.6,
        *,
        min_chunk_tokens: int = 80,
        batch_size: int = 32,
        language: str = "english",
        model_name: str = "all-MiniLM-L6-v2",
        encoder_name: str = "cl100k_base",
        use_semantic_splits: bool = True,
    ):
        # Guard and store config
        self.chunk_size = max(1, int(chunk_size))
        self.overlap = max(0, int(overlap))
        self.similarity_threshold = float(similarity_threshold)
        self.min_chunk_tokens = max(1, int(min_chunk_tokens))
        self.batch_size = max(1, int(batch_size))
        self.language = language
        self.use_semantic_splits = bool(use_semantic_splits)
        self.model_name = model_name
        self.encoder_name = encoder_name

        # Token encoder
        self.encoder = tiktoken.get_encoding(self.encoder_name)

        # Sentence embedding model (optional)
        self.model = None
        if self.use_semantic_splits:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is required when use_semantic_splits=True.\n"
                    "Install with: pip install sentence-transformers"
                )
            self.model = SentenceTransformer(self.model_name)  # type: ignore

        # Ensure sentence tokenizer availability
        self._ensure_nltk_punkt()

    def _ensure_nltk_punkt(self) -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                # Some environments ship an alternative resource name
                nltk.download("punkt_tab", quiet=True)

    def chunk_document(self, text: str, metadata: Dict) -> List[Dict]:
        """Chunk a document into token-budgeted, sentence-aware chunks.
        If use_semantic_splits is True, adjacent sentence similarities guide split points.
        """
        # Tokenize text into sentences
        sentences = nltk.sent_tokenize(text or "", language=self.language)

        if not sentences:
            return []

        # Single sentence path: still ensure token budget
        if len(sentences) == 1:
            return self._handle_large_chunk(sentences[0], metadata)

        # Semantic similarities between consecutive sentences (optional)
        similarities: List[float] = []
        use_semantic = self.use_semantic_splits and self.model is not None and util is not None  # type: ignore
        if use_semantic:
            embeddings = self.model.encode(  # type: ignore
                sentences,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            sim_matrix = util.cos_sim(embeddings[:-1], embeddings[1:])  # type: ignore
            similarities = sim_matrix.diag().tolist()

        # Precompute sentence token counts
        sent_tokens = [self._token_count(s) for s in sentences]

        # Greedy packing with token budget; semantic split when threshold crossed and min tokens reached
        chunks_text: List[str] = []
        curr_sentences: List[str] = []
        curr_tokens = 0

        for i, sentence in enumerate(sentences):
            stoks = sent_tokens[i]

            # If a single sentence exceeds the budget, flush current and slice this sentence
            if stoks > self.chunk_size:
                if curr_sentences:
                    chunks_text.append(" ".join(curr_sentences))
                    curr_sentences, curr_tokens = [], 0
                big_parts = self._split_with_overlap(sentence, self.chunk_size, self.overlap)
                chunks_text.extend(big_parts)
                continue

            # Try to add to current chunk; flush if would exceed budget
            will_exceed = curr_tokens + stoks > self.chunk_size
            if will_exceed:
                if curr_sentences:
                    chunks_text.append(" ".join(curr_sentences))
                curr_sentences, curr_tokens = [sentence], stoks
            else:
                curr_sentences.append(sentence)
                curr_tokens += stoks

            # Check semantic split (only if similarities computed and not at last sentence)
            if use_semantic and i < len(sentences) - 1:
                if similarities and similarities[i] < self.similarity_threshold and curr_tokens >= self.min_chunk_tokens:
                    chunks_text.append(" ".join(curr_sentences))
                    curr_sentences, curr_tokens = [], 0

        # Flush remaining
        if curr_sentences:
            chunks_text.append(" ".join(curr_sentences))

        # Ensure final compliance with token budget using token slicing for any oversized residuals
        final_chunks: List[Dict] = []
        for ch in chunks_text:
            tok_count = self._token_count(ch)
            if tok_count <= self.chunk_size:
                final_chunks.append(self._create_chunk(ch, metadata))
            else:
                for sub in self._split_with_overlap(ch, self.chunk_size, self.overlap):
                    final_chunks.append(self._create_chunk(sub, metadata))

        return final_chunks

    def _token_count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _handle_large_chunk(self, text: str, metadata: Dict) -> List[Dict]:
        tokens = self.encoder.encode(text)
        if len(tokens) <= self.chunk_size:
            return [self._create_chunk(text, metadata)]
        return [self._create_chunk(t, metadata) for t in self._split_with_overlap(text, self.chunk_size, self.overlap)]

    def _split_with_overlap(self, text: str, size: int, overlap: int) -> List[str]:
        tokens = self.encoder.encode(text)
        parts: List[str] = []
        if size <= 0:
            return [self.encoder.decode(tokens)]
        step = max(1, size - max(0, overlap))
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + size]
            if not chunk_tokens:
                continue
            parts.append(self.encoder.decode(chunk_tokens))
        return parts

    def _create_chunk(self, text: str, metadata: Dict) -> Dict:
        tok_len = self._token_count(text)
        preview = (text[:100] + "...") if len(text) > 100 else text
        return {
            "text": text,
            "metadata": {
                **(metadata or {}),
                "chunk_size": tok_len,
                "preview": preview,
                "language": self.language,
            },
        }


if __name__ == '__main__':
    sample = """Deep learning revolutionized computer vision. Convolutional neural networks achieved state-of-the-art results.
    However, training requires large datasets and compute. Transfer learning mitigates data scarcity.
    In contrast, decision trees are interpretable but often less accurate. Random forests improve stability.
    This is a single-super-long-sentence-with-many-words that might exceed the token budget significantly depending on the tokenizer and configuration settings applied here.
    """

    # Use non-semantic path to avoid requiring sentence-transformers
    chunker = SemanticChunker(chunk_size=30, overlap=5, use_semantic_splits=False)
    chunks = chunker.chunk_document(sample, {"source": "test"})
    print(len(chunks))
    for i, ch in enumerate(chunks):
        print(i, ch["metadata"]["chunk_size"], ch["text"].replace('\n', ' '))
