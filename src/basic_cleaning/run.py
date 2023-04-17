#!/usr/bin/env python
"""
Download the raw data set, clean it, export the resulting artifact.
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    ######################
    # YOUR CODE HERE     #
    ######################
    logger.info("Downloading artifact.")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv = pd.read_csv(artifact_local_path)

    logger.info(f"Selecting data with price between {args.min_price:.2f} and {args.max_price:.2f}.")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    
    logger.info("Removing null values.")
    df.dropna(inplace=True)

    logger.info(f"Saving artifact as {args.output_artifact}.")
    df.to_csv(args.output_artifact, index=False)

    logger.info("Logging cleaned data.")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning step.")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Full input artifact name.",
        required=True
        )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Full output artifact name.",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Output artifact type.",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact.",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to include in data.",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to include in data.",
        required=True
    )

    args = parser.parse_args()

    go(args)
