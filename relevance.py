"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""
def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP at.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    relevant_count = 0
    cumulative_precision = 0.0

    for rank, relevance in enumerate(search_result_relevances[:cut_off], start=1):
        if relevance > 0: 
            relevant_count += 1
            precision_at_rank = relevant_count / rank
            cumulative_precision += precision_at_rank

    if relevant_count == 0:
        return 0.0
    return cumulative_precision / cut_off
    


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off: int = 10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: 
    import math

    def dcg(scores):
        return sum(score / math.log2(rank + 1) if rank > 0 else score for rank, score in enumerate(scores[:cut_off]))

    actual_dcg = dcg(search_result_relevances)
    ideal_dcg = dcg(ideal_relevance_score_ordering)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg



def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset
    import pandas as pd

    relevance_data = pd.read_csv(relevance_data_filename,encoding='ISO-8859-1')

    map_scores = []
    ndcg_scores = []

    unique_queries = relevance_data['query'].unique()

    for query in unique_queries:
        query_relevance_data = relevance_data[relevance_data['query'] == query]
        doc_relevance_mapping = query_relevance_data.set_index('docid')['rel'].to_dict()
        relevances = list(doc_relevance_mapping.values())
        ideal_relevances = sorted(relevances, reverse=True)

        ranked_results = ranker.query(query)

        search_relevances_map = []
        search_relevances_ndcg = []
        for doc_id, _ in ranked_results:
            search_relevances_map.append(1 if doc_relevance_mapping.get(doc_id, 0) > 0 else 0)
            search_relevances_ndcg.append(doc_relevance_mapping.get(doc_id, 0))

        map_scores.append(map_score(search_relevances_map))
        ndcg_scores.append(ndcg_score(search_relevances_ndcg, ideal_relevances))

    avg_map = sum(map_scores) / len(map_scores)
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

    return {
        'MAP': avg_map,
        'NDCG': avg_ndcg,
        'MAP_scores': map_scores,
        'NDCG_scores': ndcg_scores
    }

if __name__ == '__main__':
    pass
    