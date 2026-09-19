import re
import numpy as np


class FaithfulnessEvaluator:

    def __init__(
        self,
        embedding_model,
        bert_model_type="distilbert-base-uncased"
    ):

        self.embedding_model = embedding_model

        # ============================================================
        # BERTScore
        # ============================================================

        self.bert_model_type = bert_model_type
        self._bert_scorer = None
        self.bert_score_available = False

        try:

            from bert_score import BERTScorer

            self._bert_scorer = BERTScorer(
                model_type=self.bert_model_type,
                lang="en",
                rescale_with_baseline=False
            )

            self.bert_score_available = True

        except Exception:

            self._bert_scorer = None
            self.bert_score_available = False

    # ============================================================
    # SENTENCE SPLITTER
    # ============================================================

    def _split_sentences(self, text):

        raw_sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip()
        )

        sentences = [
            sentence.strip()
            for sentence in raw_sentences
            if len(
                re.findall(
                    r"[a-zA-Z]{2,}",
                    sentence
                )
            ) >= 3
        ]

        if not sentences and text.strip():

            sentences = [
                text.strip()
            ]

        return sentences

    # ============================================================
    # JACCARD TOKENIZATION
    # ============================================================

    def _tokenize_for_jaccard(self, text):

        words = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )

        return set(words)

    # ============================================================
    # JACCARD
    # ============================================================

    def _calculate_jaccard(
        self,
        answer,
        reference_answer
    ):

        answer_words = self._tokenize_for_jaccard(
            answer
        )

        reference_words = self._tokenize_for_jaccard(
            reference_answer
        )

        if not answer_words or not reference_words:

            return 0.0

        intersection = answer_words & reference_words

        union = answer_words | reference_words

        if not union:

            return 0.0

        return (
            len(intersection)
            /
            len(union)
        )

    # ============================================================
    # MAIN EVALUATION
    # ============================================================

    def evaluate(
        self,
        answer,
        selected_clauses,
        reference_answer=None
    ):

        # ========================================================
        # EMPTY ANSWER
        # ========================================================

        if not answer or not answer.strip():

            return {
                "faithfulness_score": 0.0,
                "hallucination_rate": 100.0,
                "semantic_similarity": 0.0,
                "weakest_sentence_similarity": 0.0,

                "bert_precision": 0.0,
                "bert_recall": 0.0,
                "bert_f1": 0.0,
                "bert_hallucination_rate": 100.0,

                "jaccard_similarity": 0.0,
                "jaccard_hallucination_rate": 100.0,

                "answer_quality": "No answer generated"
            }

        # ========================================================
        # EMPTY CONTEXT
        # ========================================================

        if not selected_clauses:

            return {
                "faithfulness_score": 0.0,
                "hallucination_rate": 100.0,
                "semantic_similarity": 0.0,
                "weakest_sentence_similarity": 0.0,

                "bert_precision": 0.0,
                "bert_recall": 0.0,
                "bert_f1": 0.0,
                "bert_hallucination_rate": 100.0,

                "jaccard_similarity": 0.0,
                "jaccard_hallucination_rate": 100.0,

                "answer_quality": "No supporting clauses"
            }

        # ========================================================
        # BUILD CONTEXT
        # ========================================================

        context = "\n\n".join(
            clause.get("text", "")
            for clause in selected_clauses
            if clause.get("text", "").strip()
        )

        # ========================================================
        # COSINE SIMILARITY
        # ========================================================

        answer_sentences = self._split_sentences(
            answer
        )

        if not answer_sentences:

            answer_sentences = [
                answer
            ]

        clause_texts = [
            clause.get("text", "")
            for clause in selected_clauses
            if clause.get("text", "").strip()
        ]

        if not clause_texts:

            clause_texts = [
                context
            ]

        # --------------------------------------------------------
        # Sentence embeddings
        # --------------------------------------------------------

        sentence_embeddings = (
            self.embedding_model.model.encode(
                answer_sentences,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        )

        # --------------------------------------------------------
        # Clause embeddings
        # --------------------------------------------------------

        clause_embeddings = (
            self.embedding_model.model.encode(
                clause_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        )

        # --------------------------------------------------------
        # Cosine similarity matrix
        # --------------------------------------------------------

        similarity_matrix = np.matmul(
            sentence_embeddings,
            clause_embeddings.T
        )

        # --------------------------------------------------------
        # Best matching clause for each sentence
        # --------------------------------------------------------

        best_match_per_sentence = (
            similarity_matrix.max(axis=1)
        )

        # --------------------------------------------------------
        # ORIGINAL RAW COSINE
        # --------------------------------------------------------

        raw_similarity = float(
            np.mean(
                best_match_per_sentence
            )
        )

        # Keep it between 0 and 1

        raw_similarity = np.clip(
            raw_similarity,
            0.0,
            1.0
        )

        # Raw cosine as percentage

        raw_cosine_percentage = (
            raw_similarity * 100.0
        )

        # Weakest sentence

        weakest_sentence_similarity = float(
            np.min(
                best_match_per_sentence
            )
        )

        # ========================================================
        # BERTSCORE
        # ========================================================

        bert_precision = 0.0
        bert_recall = 0.0
        bert_f1 = 0.0

        bert_reference = (
            reference_answer
            if reference_answer
            and reference_answer.strip()
            else context
        )

        if (
            self.bert_score_available
            and bert_reference.strip()
        ):

            try:

                precision, recall, f1 = (
                    self._bert_scorer.score(
                        [answer],
                        [bert_reference]
                    )
                )

                bert_precision = float(
                    precision[0]
                )

                bert_recall = float(
                    recall[0]
                )

                bert_f1 = float(
                    f1[0]
                )

            except Exception:

                bert_precision = 0.0
                bert_recall = 0.0
                bert_f1 = 0.0

        # ========================================================
        # BERT SCORE PERCENTAGE
        # ========================================================

        bert_score_pct = np.clip(
            bert_f1 * 100.0,
            0.0,
            100.0
        )

        # ========================================================
        # FINAL COSINE-BASED FAITHFULNESS
        # ========================================================
        # IMPORTANT:
        # Keep cosine similarity completely independent from
        # BERTScore.
        #
        # The graph and the raw similarity must represent the
        # exact same metric:
        #
        # raw_similarity = 0.5198
        # raw_cosine_percentage = 51.98
        # faithfulness_score = 51.98
        #
        # BERTScore is calculated separately and is NOT used
        # to modify the cosine similarity.
        # ========================================================

        faithfulness_score = np.clip(
            raw_cosine_percentage,
            0.0,
            100.0
        )

        # ========================================================
        # HALLUCINATION RISK
        # ========================================================

        hallucination_rate = max(
            0.0,
            100.0 - faithfulness_score
        )

        # ========================================================
        # BERT HALLUCINATION
        # ========================================================

        bert_hallucination_rate = max(
            0.0,
            100.0 - bert_score_pct
        )

        # ========================================================
        # JACCARD
        # ========================================================

        jaccard_reference = (
            reference_answer
            if reference_answer
            and reference_answer.strip()
            else context
        )

        jaccard_similarity = (
            self._calculate_jaccard(
                answer,
                jaccard_reference
            )
        )

        jaccard_score_pct = (
            jaccard_similarity * 100.0
        )

        jaccard_hallucination_rate = max(
            0.0,
            100.0 - jaccard_score_pct
        )

        # ========================================================
        # QUALITY
        # ========================================================

        if faithfulness_score >= 85:

            quality = "Excellent grounding"

        elif faithfulness_score >= 70:

            quality = "Good grounding"

        elif faithfulness_score >= 50:

            quality = "Moderate grounding"

        else:

            quality = (
                "Low grounding / "
                "High hallucination risk"
            )

        # ========================================================
        # RETURN RESULTS
        # ========================================================

        return {

            # ----------------------------------------------------
            # COSINE
            # ----------------------------------------------------

            "faithfulness_score": round(
                float(faithfulness_score),
                2
            ),

            "hallucination_rate": round(
                float(hallucination_rate),
                2
            ),

            "semantic_similarity": round(
                float(raw_similarity),
                4
            ),

            "weakest_sentence_similarity": round(
                weakest_sentence_similarity,
                4
            ),

            # ----------------------------------------------------
            # BERTSCORE
            # ----------------------------------------------------

            "bert_precision": round(
                bert_precision,
                4
            ),

            "bert_recall": round(
                bert_recall,
                4
            ),

            "bert_f1": round(
                bert_f1,
                4
            ),

            "bert_hallucination_rate": round(
                bert_hallucination_rate,
                2
            ),

            # ----------------------------------------------------
            # JACCARD
            # ----------------------------------------------------

            "jaccard_similarity": round(
                jaccard_similarity,
                4
            ),

            "jaccard_hallucination_rate": round(
                jaccard_hallucination_rate,
                2
            ),

            # ----------------------------------------------------
            # EXTRA DEBUG VALUES
            # ----------------------------------------------------

            # Explicit cosine percentage for the UI.
            # This is exactly raw_similarity * 100 and must be used
            # for the Cosine Similarity gauge.
            "cosine_score_percentage": round(
                raw_cosine_percentage,
                2
            ),

            "raw_cosine_percentage": round(
                raw_cosine_percentage,
                2
            ),

            "bert_score_percentage": round(
                bert_score_pct,
                2
            ),

            "answer_quality": quality
        }