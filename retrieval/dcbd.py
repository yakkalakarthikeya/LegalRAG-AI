class DCBD:
    
    def __init__(
        self,
        coverage_threshold=0.60,
        min_clauses=2,
        max_clauses=4
    ):

        self.coverage_threshold = coverage_threshold
        self.min_clauses = min_clauses
        self.max_clauses = max_clauses

    # =====================================================
    # SELECT MOST RELEVANT CLAUSES
    # =====================================================

    def select_clauses(
        self,
        retrieval_results
    ):

        if not retrieval_results:
            return []

        # -----------------------------------------
        # Sort by similarity
        # -----------------------------------------

        results = sorted(
            retrieval_results,
            key=lambda x: float(x["score"]),
            reverse=True
        )

        # -----------------------------------------
        # Keep only positive similarities
        # -----------------------------------------

        results = [
            item
            for item in results
            if float(item["score"]) > 0
        ]

        if not results:
            return []

        # -----------------------------------------
        # If only a few clauses exist
        # -----------------------------------------

        if len(results) <= self.min_clauses:

            return results

        # -----------------------------------------
        # Total similarity
        # -----------------------------------------

        total_similarity = sum(
            float(item["score"])
            for item in results
        )

        if total_similarity <= 0:

            return results[
                :self.min_clauses
            ]

        # -----------------------------------------
        # Cumulative similarity
        # -----------------------------------------

        selected = []

        cumulative_similarity = 0.0

        for item in results:

            selected.append(item)

            cumulative_similarity += float(
                item["score"]
            )

            coverage = (
                cumulative_similarity
                /
                total_similarity
            )

            # -------------------------------------
            # Stop when enough relevant context
            # has been selected
            # -------------------------------------

            if (
                coverage >= self.coverage_threshold
                and
                len(selected) >= self.min_clauses
            ):

                break

            # -------------------------------------
            # Maximum context limit
            # -------------------------------------

            if len(selected) >= self.max_clauses:

                break

        # -----------------------------------------
        # Guarantee minimum
        # -----------------------------------------

        if len(selected) < self.min_clauses:

            selected = results[
                :min(
                    self.min_clauses,
                    len(results)
                )
            ]

        return selected