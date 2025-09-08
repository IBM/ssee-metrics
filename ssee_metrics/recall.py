from typing import List, Optional, Dict, Any

def sort_candidate_matches_by_similarity(
    E_gold: List[str],
    encoded_E_gold: List[Any],
    cand_matches: Dict[str, List[Any]],
    su: 'SemanticUtils'
    ) -> Dict[str, List[Any]]:
    """
    Sort candidate matches by similarity score for each gold entity.

    Args:
        - E_gold (list of str): List of gold standard entities represented as strings.
        - encoded_E_gold (list): List of encoded representations of gold entities.
        - cand_matches (dict):  Dictionary where keys are gold entities and values 
                                are lists of candidate matches.
        - su (concrete implementation of SemanticUtils class) : Semantic Ultities class to use.
    Returns:
        - dict: Dictionary where keys are gold entities and values are lists of candidate 
                matches sorted by similarity score.
    """
    cand_matches_sorted = {}

    for e_g, encoded_e_g in zip(E_gold, encoded_E_gold):
        # Sort the candidate matches for the current gold entity by similarity score
        sorted_candidates = sorted(
            cand_matches[e_g],
            key=lambda e_c: su.sem_sim(encoded_e_g, su.sem_enc(e_c)),
            reverse=True
        )

        cand_matches_sorted[e_g] = sorted_candidates

    return cand_matches_sorted

def structured_semantic_entity_evaluation_recall(
    E_cand: List[str],
    E_gold: List[str],
    T: float,
    su: 'SemanticUtils'
    ) -> Optional[float]:
    """
    Calculate the recall of semantic entity matching between gold entities and candidate entities.

    Args:
        E_cand (list of str): List of candidate entities represented as strings.
        E_gold (list of str): List of gold standard entities represented as strings.
        T (float): Similarity threshold for considering a match.
        su (concrete implementation of 'SemanticUtils') : Semantic Utilities class to use.

    Returns:
        float: Recall value, which is the proportion of gold entities that have a matching candidate entity.
    """
    if len(E_gold) == 0:
        return float('nan')
    if su.threshold_check(T) is False:
        return None

    # Step 1: Encode all entities
    encoded_E_gold = [su.sem_enc(e) for e in E_gold]
    encoded_E_cand = [su.sem_enc(e) for e in E_cand]

    # Step 2 Initialise the list of candidate matches for each gold entity
    cand_matches = {e: [] for e in E_gold}

    # Step 3: Find candidate matches for each gold entity
    for e_g, encoded_e_g in zip(E_gold, encoded_E_gold):
        for e_c, encoded_e_c in zip(E_cand, encoded_E_cand):
            if su.sem_sim(encoded_e_g, encoded_e_c) >= T:
                cand_matches[e_g].append(e_c)

    # Step 4 Sort by the similarity score of their best match
    cand_matches_sorted = sort_candidate_matches_by_similarity(E_gold, encoded_E_gold, cand_matches, su)

    # Step 5 Initialise the set of matched candidate entities
    used_candidates = set()
    L_gold = list(E_gold)

    # Step 6: Sort gold entities by the similarity score of their best candidate match
    L_sorted_gold = sorted(
                L_gold,
                key=lambda e_g: (
                    su.sem_sim(encoded_E_gold[E_gold.index(e_g)], su.sem_enc(cand_matches_sorted[e_g][0]))
                    if cand_matches_sorted[e_g] else 0.0
                ),
                reverse=True)

    # Step 7: Find the best available match for each gold entity
    for e_g in L_sorted_gold:
        for e_c in cand_matches_sorted[e_g]:
            if e_c not in used_candidates:
                used_candidates.add(e_c)
                break # Add at most one match for each gold element.

    # Step 8: Calculate recall
    recall = len(used_candidates)/len(E_gold)

    return recall
