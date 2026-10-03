import re


class JaccardEvaluator:
    """Sentence-aligned lexical grounding score.

    A single Jaccard comparison between an answer and a long collection of
    clauses is dominated by unrelated words in the clauses. This evaluator
    measures each answer sentence against its best supporting passage and
    returns a word-count-weighted mean.
    """

    _STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "been", "being",
        "by", "for", "from", "has", "have", "he", "her", "hers", "him",
        "his", "i", "if", "in", "into", "is", "it", "its", "of", "on",
        "or", "our", "ours", "she", "that", "the", "their", "theirs",
        "them", "they", "this", "those", "to", "was", "were", "will",
        "with", "you", "your", "yours",
    }
    _HEADING_WORDS = {
        "analysis", "conclusion", "executive", "limitations", "provisions",
        "relevant", "summary", "synthesis",
    }

    def _clean_text(self, text):
        text = re.sub(r"\[[^\]]{0,80}\]", " ", text or "")
        text = re.sub(r"https?://\S+", " ", text)
        return re.sub(r"[*_`#>|]", " ", text)

    def _split_sentences(self, text):
        pieces = re.split(r"(?<=[.!?;])\s+|\n+", self._clean_text(text))
        return [piece.strip(" -:\t") for piece in pieces if piece.strip(" -:\t")]

    @staticmethod
    def _normalise_token(token):
        """Apply conservative suffix folding without extra dependencies."""
        irregular = {
            "parties": "party", "obligations": "obligation",
            "rights": "right", "agreements": "agreement",
            "terminated": "terminate", "terminates": "terminate",
            "terminating": "terminate", "termination": "terminate",
        }
        if token in irregular:
            return irregular[token]
        if len(token) > 5 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 5 and token.endswith("ing"):
            stem = token[:-3]
            return stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem
        if len(token) > 4 and token.endswith("ed"):
            stem = token[:-2]
            return stem + "e" if stem.endswith(("at", "iz")) else stem
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    def _tokenize(self, text):
        words = re.findall(r"\b[a-zA-Z0-9]+\b", self._clean_text(text).lower())
        return {
            self._normalise_token(word)
            for word in words
            if word not in self._STOP_WORDS and not word.isdigit()
        }

    def _content_sentences(self, text):
        sentences = []
        for sentence in self._split_sentences(text):
            tokens = self._tokenize(sentence)
            if tokens and not tokens.issubset(self._HEADING_WORDS):
                sentences.append(sentence)
        return sentences

    def _reference_passages(self, reference_answer):
        sentences = self._content_sentences(reference_answer)
        passages = list(sentences)
        passages.extend(
            f"{sentences[index]} {sentences[index + 1]}"
            for index in range(len(sentences) - 1)
        )
        return passages

    @staticmethod
    def _jaccard(left, right):
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    def score(self, answer, reference_answer):
        if not answer or not answer.strip() or not reference_answer or not reference_answer.strip():
            return 0.0

        answer_sentences = self._content_sentences(answer)
        reference_sets = [
            tokens
            for tokens in (self._tokenize(p) for p in self._reference_passages(reference_answer))
            if tokens
        ]
        if not answer_sentences or not reference_sets:
            return 0.0

        weighted_total = 0.0
        total_weight = 0.0
        for sentence in answer_sentences:
            answer_tokens = self._tokenize(sentence)
            if not answer_tokens:
                continue
            best = max(self._jaccard(answer_tokens, reference) for reference in reference_sets)
            weight = float(len(answer_tokens))
            weighted_total += best * weight
            total_weight += weight

        return weighted_total / total_weight if total_weight else 0.0

    def evaluate(self, answer, reference_answer):
        similarity = self.score(answer, reference_answer)
        return {
            "jaccard_similarity": round(similarity, 4),
            "jaccard_hallucination_rate": round(100.0 * (1.0 - similarity), 2),
        }
