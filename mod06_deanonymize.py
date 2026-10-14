import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasi_identifiers = ["age", "zip3", "gender"]

    aux_counts = (
        aux_df.groupby(quasi_identifiers)
        .size()
        .reset_index(name="match_count")
    )

    unique_combinations = aux_counts[
        aux_counts["match_count"] == 1
    ][quasi_identifiers]

    candidates = anon_df.merge(
        aux_df,
        on=quasi_identifiers,
        how="inner",
        suffixes=("_anon", "_aux")
    )

    candidates = candidates.merge(
        unique_combinations,
        on=quasi_identifiers,
        how="inner"
    )

    matches_df = candidates[["anon_id", "name"]].rename(
        columns={"name": "matched_name"}
    )

    return matches_df


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    total_records = len(anon_df)

    if total_records == 0:
        return 0.0

    uniquely_matched = matches_df["anon_id"].nunique()

    return uniquely_matched / total_records
