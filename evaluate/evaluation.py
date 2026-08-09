"""
Running some simple evaluation steps against the model on a small test set
Will record the BLEU score per language pair.
This sort of evaluation should be ran on every model, model version and config change
before deploying against a threshold that if it doesn't pass should fail the deployment

Evaluation should also run on a schedule with results from production to see how real world
performance compares. Aim to catch drift in model output that using a static test set can't really capture

run with: uv run -m evaluate.evaluation
"""

import json
from collections import defaultdict
from pathlib import Path

import sacrebleu

from application.translation import translate_batch, warm_up

testset_path = Path(__file__).parent / "testset.jsonl"


def load_testset():

    rows = []
    with open(testset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def main():

    warm_up()
    rows = load_testset()

    pair = defaultdict(list)
    for row in rows:
        pair[(row["source_lang"], row["target_lang"])].append(row)

    for (src, tgt), items in sorted(pair.items()):
        sources = [i["source"] for i in items]
        references = [i["reference"] for i in items]

        test_cases_input = translate_batch(sources, src, tgt)

        bleu_score = sacrebleu.corpus_bleu(test_cases_input, [references])

        # here I would add far more metrics rather than just one
        # would add some semantic evaluation like cosine similarity between embedding vectors for example.
        print(
            f"{src}->{tgt:<2} Num of cases:{len(items)} BLEU Score:{bleu_score.score:.2f}"
        )


if __name__ == "__main__":
    main()
