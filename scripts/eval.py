#!/usr/bin/env python
# import
## batteries
import os
import sys
import argparse
from typing import List, Dict, Literal, Any
## 3rd party
from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate
from pypika import Query, Table, Criterion, functions as fn
## package
from SRAgent.db.connect import db_connect
from SRAgent.db.upsert import db_upsert


# argparse
def parse_args():
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          argparse.RawDescriptionHelpFormatter):
        pass

    desc = 'Evaluate a the accuracy of the data'
    epi = """DESCRIPTION:
    Evaluate the accuracy of SRAgent predictions.
    Evaluation datasets are stored in the SRAgent postgresql database.
    The predictions and ground truth metadata are pulled from the database and compared.
    So, you must first add the predictions to the database.
    To add an evaluation dataset, use the `--add-dataset` flag.
    """
    parser = argparse.ArgumentParser(description=desc, epilog=epi,
                                     formatter_class=CustomFormatter)
    parser.add_argument('--eval-datasets', type=str, nargs="+", 
                        help='Evaluation dataset(s) names to use')
    parser.add_argument('--list-datasets', action='store_true', default=False,
                        help='List available evaluation datasets')
    parser.add_argument('--add-dataset', type=str, default=None,
                        help='Provide an evaluation dataset csv to add to the database')
    parser.add_argument('--outfile', type=str, default="incorrect.tsv",
                        help='Output file for incorrect predictions')
    parser.add_argument('--tenant', type=str, default="prod",
                        choices=["prod", "test"],
                        help='Database tenant to use')
    parser.add_argument('--srx-no-eval', type=str, default=None,
                        help='If a file path is provided, find SRX accessions in SRX_metadata that are not yet in the eval table (lack ground truth) and save to file (CSV)')
    return parser.parse_args()

# functions
def add_suffix(columns: list, suffix: str="_x") -> list:
    """
    Add suffixes to duplicate column names.
    Args:
        columns: List of column names.
        suffix: Suffix to add to duplicate column names.
    Return:
        List of column names with suffixes added.
    """
    seen = set()
    result = []
    
    for col in columns:
        if col in seen:
            result.append(col + suffix)
        else:
            seen.add(col) 
            result.append(col)  
    return result

def list_datasets() -> pd.DataFrame:
    """
    List available datasets in the database.
    Return:
        DataFrame  of dataset IDs and record counts.
    """
    with db_connect() as conn:
        tbl = Table("eval")
        stmt = Query \
            .from_(tbl) \
            .select(tbl.dataset_id, fn.Count(tbl.dataset_id).as_("record_count")) \
            .groupby(tbl.dataset_id)
        datasets = pd.read_sql(str(stmt), conn)
        return datasets

def add_update_eval_dataset(csv_file: str) -> None:
    """
    Add or update an evaluation dataset in the database.
    Args:
        csv_file: Path to the dataset CSV file.
    """
    # check if file exists
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return None
    # load csv
    df = pd.read_csv(csv_file)
    dataset_id = os.path.splitext(os.path.split(csv_file)[1])[0]
    df["dataset_id"] = dataset_id
    # does dataset exist?
    existing_datasets = list_datasets()["dataset_id"].tolist()
    action = "Updated existing" if dataset_id in existing_datasets else "Added new"
    # add to database
    with db_connect() as conn:
        db_upsert(df, "eval", conn)
    print(f"{action} dataset: {dataset_id}")

def load_eval_datasets(eval_datasets: List[str]) -> pd.DataFrame:
    """
    Load the evaluation dataset(s) (eval table) and the associated predictions (SRX_metadata table).
    Args:
        eval_datasets: List of evaluation dataset IDs.
    Return:
        DataFrame of the evaluation dataset and associated predictions.
    """
    df = None
    with db_connect() as conn:
        tbl_eval = Table("eval")
        tbl_pred = Table("srx_metadata")
        stmt = Query \
            .from_(tbl_eval) \
            .where(tbl_eval.dataset_id.isin(eval_datasets)) \
            .join(tbl_pred) \
            .on(
                (tbl_eval.database == tbl_pred.database) & 
                (tbl_eval.entrez_id == tbl_pred.entrez_id) &
                (tbl_eval.srx_accession == tbl_pred.srx_accession)
            ) \
            .select("*") 
        df = pd.read_sql(str(stmt), conn)
        df.columns = add_suffix(df.columns, "_pred")
    # drop "created_at" and "updated_at" columns
    cols_to_drop = [col for col in df.columns if col.startswith("created_at") or col.startswith("updated_at")]
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df

def srx_no_eval() -> pd.DataFrame:
    """
    Find SRX accessions in the SRX_metadata table that are not in the eval table.
    
    Returns:
        DataFrame of SRX_metadata records that don't have corresponding eval records.
    """
    with db_connect() as conn:
        tbl_pred = Table("srx_metadata")
        tbl_eval = Table("eval")
        
        # Subquery to get all srx_accessions in eval table
        subquery = Query.from_(tbl_eval).select(tbl_eval.srx_accession).distinct()
        
        # Main query to get all metadata where srx_accession not in eval table
        stmt = Query \
            .from_(tbl_pred) \
            .where(tbl_pred.srx_accession.notin(subquery)) \
            .select("*")
        
        df = pd.read_sql(str(stmt), conn)
        
        print(f"Found {len(df)} SRX accessions in SRX_metadata that are not in the eval table")
        return df

def eval(
    df: pd.DataFrame, 
    exclude_cols: List[str]=["database", "entrez_id", "srx_accession"], 
    outfile: str="incorrect.tsv"
    ) -> None:
    """
    Evaluate the accuracy of the predictions.
    Args:
        df: DataFrame of the evaluation dataset.
        exclude_cols: Columns to exclude from evaluation.
        outfile: Output file for incorrect predictions.
    """
    # Get base columns (those without _pred suffix)
    base_cols = [col.replace("_pred", "") for col in df.columns if col.endswith('_pred')]

    # Create comparison for each column pair
    accuracy = {} 
    idx = set()
    for col in base_cols:
        if col in exclude_cols:
            continue
        pred_col = f"{col}_pred"
        if pred_col in df.columns:  # Check if prediction column exists
            # Compare values and show where they differ
            mismatches = df[df[col] != df[pred_col]]
            idx.update(mismatches.index)
            
            # Calculate mismatch percentage
            mismatch_pct = (len(mismatches) / len(df)) * 100
            accuracy[col] = 100.0 - mismatch_pct
            
            print(f"\n#-- {col} --#")
            print(f"# Total mismatches: {len(mismatches)} ({mismatch_pct:.2f}%)")
            
            if len(mismatches) > 0:
                # Display count of each 
                print("\n# Mismatches")
                df_mm = mismatches.groupby([col, pred_col]).size().reset_index(name="count")
                print(tabulate(df_mm.values, headers=df_mm.columns, tablefmt="github"))

    # convert to dataframe
    accuracy = pd.DataFrame(accuracy.items(), columns=["column", "accuracy_percent"])
    accuracy["accuracy_percent"] = accuracy["accuracy_percent"].round(2)
    accuracy["count"] = df.shape[0]

    # write to stdout
    print("\n#-- Accuracy Table --#")
    #accuracy.to_csv(sys.stdout, index=False, sep="\t")
    print(tabulate(accuracy.values, headers=df.columns, tablefmt="github"))

    # overall accuracy
    overall_accuracy = accuracy["accuracy_percent"].mean()
    print(f"\nOverall accuracy: {overall_accuracy:.2f}%")

    # print out the mismatch records
    print("\n#-- Mismatch Records --#")
    df_wrong = df.iloc[list(idx)]
    outdir = os.path.dirname(outfile)
    if outdir and outdir != ".":
        os.makedirs(outdir, exist_ok=True)
    df_wrong.to_csv(outfile, sep="\t", index=False)
    print(f"Saved mismatch records to: {outfile}")
    
    # Get comma-separated list of SRX accessions with at least one mismatch
    srx_col = "srx_accession" if "srx_accession" in df_wrong.columns else "srx_accession_pred"
    srx_mismatches = df_wrong[srx_col].unique()
    print("\n#-- SRX IDs with Mismatches --#")
    print(f"Count: {len(srx_mismatches)}")
    print(f"List: {','.join(srx_mismatches)}")

def main(args):
    # set pandas options
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 100)

    # set tenant
    if args.tenant:
        os.environ["DYNACONF"] = args.tenant
    print(f"Using database tenant: {args.tenant}")

    # add evaluation dataset
    if args.add_dataset:
        add_update_eval_dataset(args.add_dataset)
        return None

    # list available datasets
    if args.list_datasets:
        print(list_datasets())
        return None
    
    # find missing SRX accessions
    if args.srx_no_eval:
        srx_missing_eval = srx_no_eval()
        outdir = os.path.dirname(args.srx_no_eval)
        if outdir and outdir != ".":
            os.makedirs(outdir, exist_ok=True)
        srx_missing_eval.to_csv(args.srx_no_eval, sep=",", index=False)
        print(f"Saved SRX records lacking eval records to: {args.srx_no_eval}")
        return None

    # evaluation
    if not args.eval_datasets:
        print("Please provide an evaluation dataset ID (use --list-datasets to see available datasets)")
        return None
    missing_eval_datasets = [x for x in args.eval_datasets if x not in list_datasets()["dataset_id"].tolist()]
    if missing_eval_datasets:
        for missing in missing_eval_datasets:
            print(f"Dataset not found: {missing}")
        return None
    df = load_eval_datasets(args.eval_datasets)
    eval(df, outfile=args.outfile)


# Example usage
if __name__ == "__main__":
    load_dotenv(override=True)
    args = parse_args()
    main(args)