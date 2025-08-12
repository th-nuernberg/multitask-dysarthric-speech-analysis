#!/usr/bin/env python3

"""
This is a local version of https://github.com/xiuwenz2/SAPC-template/blob/main/utils/evaluate.py
"""

import logging
import sys
import argparse
import os
import re
import json
from metrics import calculate_word_error_rate, SemScore
from tqdm import tqdm

logger = logging.getLogger(__name__)


def process_punc(trans):
    codes = """
            \u0041-\u005a\u0027\u0020
            \u00c0\u00c1\u00c4\u00c5\u00c8\u00c9\u00cd\u00cf
            \u00d1\u00d3\u00d6\u00d8\u00db\u00dc\u0106
            """

    trans = trans.strip().upper()
    trans = re.sub("([^" + codes + "])", "", trans)
    trans = " ".join(trans.strip().split())

    return trans


def evaluate(results_file, hypo_pth, ref_pth):
    logger.info("Evaluation in Progress...")

    output = {"dev_split": []}

    for split, split_name in zip(["dev"], ["dev_split"]):
        references = {"wrd.without.parentheses": [], "origin.wrd": []}
        hypotheses = []

        with (
            open(
                os.path.join(ref_pth, split + ".wrd.without.parentheses"), "r"
            ) as fref1,
            open(os.path.join(ref_pth, split + ".origin.wrd"), "r") as fref2,
            open(os.path.join(hypo_pth, split + ".hypo"), "r") as fhypo,
        ):
            for r1, r2, h in tqdm(
                zip(fref1.readlines(), fref2.readlines(), fhypo.readlines())
            ):
                references["wrd.without.parentheses"].append(process_punc(r1.strip()))
                references["origin.wrd"].append(process_punc(r2.strip()))
                hypotheses.append(process_punc(h.strip()))

        ### Calculating WER
        wer = calculate_word_error_rate(
            references["wrd.without.parentheses"], references["origin.wrd"], hypotheses
        )

        ### Calculating SemScore
        semscores = {}
        for ref_type in ["wrd.without.parentheses", "origin.wrd"]:
            semscores[ref_type] = SemScore().score_all(
                refs=references[ref_type], hyps=hypotheses
            )
        semscore = [
            float(i) if i > j else float(j)
            for i, j in zip(
                semscores["wrd.without.parentheses"], semscores["origin.wrd"]
            )
        ]

        output[split_name] = [
            round(float(wer) * 100, 4),
            round(sum(semscore) / len(semscore) * 100, 4),
        ]

    logger.info("Results:")
    logger.info(output)
    json.dump(output, open(results_file, "w"), indent=6)
    logger.info(f"The evaluation for {results_file} has been successfully completed.")

    return output


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="sapc_result.json", type=str)
    parser.add_argument(
        "--hypo_path", default="./asr_results", type=str, help="path to hypo file"
    )
    parser.add_argument(
        "--ref_path", default="./refs", type=str, help="path to references file"
    )
    return parser


if __name__ == "__main__":
    formatter = "[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s >> %(message)s"
    logging.basicConfig(
        format=formatter,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    parser = get_parser()
    args = parser.parse_args()

    evaluate(args.results_file, args.hypo_path, args.ref_path)
