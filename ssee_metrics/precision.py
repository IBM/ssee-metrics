from typing import List, Optional

def structured_semantic_entity_evaluation_precision(
    E_cand: List[str],
    E_gold: List[str],
    T: float,
    su: 'SemanticUtils'
    ) -> Optional[float]:
    """
    Calculate the precision of semantic entity matching between gold entities and candidate entities.

    Args:
        - E_cand (list of str) : List of candidate entities represented as strings.
        - E_gold (list of str) : List of gold standard entities represented as strings.
        - T (float) : Similarity threshold for considering a match.
        - su (concrete implementation of 'SemanticUtils') : Semantic Utilities class to use.

    Returns:
        float: Precision value, which is the proportion of candidate entities that have a matching gold entity.
    """
    if len(E_cand) == 0:
        return float('nan')
    
    if su.threshold_check(T) is False:
        return None

    # Step 1 Encode all entities
    encoded_E_gold = [su.sem_enc(e) for e in E_gold]
    encoded_E_cand = [su.sem_enc(e) for e in E_cand]

    # Step 2: Initialise the list of gold matches for each candidate entity
    gold_matches = {e: [] for e in E_cand}

    # Step 3: Find gold matches for each candidate entity
    for e_c, encoded_e_c in zip(E_cand, encoded_E_cand):
        for e_g, encoded_e_g in zip(E_gold, encoded_E_gold):
            if su.sem_sim(encoded_e_c, encoded_e_g) >= T:
                gold_matches[e_c].append(e_g)

    # Step 4 Initialise the set of matched gold entities
    used_gold = set()
    L_cand = list(E_cand)

    # Step 5: Sort candidate entities by the similarity score of their best gold match
    L_sorted_cand = sorted(
        L_cand,
        key=lambda e_c: (
            su.sem_sim(encoded_E_cand[E_cand.index(e_c)], su.sem_enc(gold_matches[e_c][0]))
            if gold_matches[e_c] else 0.0
        ),
        reverse=True
    )

    # Step 6 Find the best available match for each candidate entity
    for e_c in L_sorted_cand:
        for e_g in gold_matches[e_c]:
            if e_g not in used_gold:
                used_gold.add(e_g)
                break  # Add at most one match for each candidate element.

    # Step 7: Calculate precision
    precision = len(used_gold)/len(E_cand)

    return precision