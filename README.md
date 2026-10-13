# Annotation and Active Learning Lab

A human-in-the-loop annotation and model-evaluation pipeline on a small subset of Anthropic HH-RLHF (helpful-base, MIT license). Items and annotations live in MongoDB 7. A TF-IDF + logistic regression classifier is trained on human labels, and uncertainty sampling is compared against random sampling.

See [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for a plain-language walkthrough and [DECISIONS.md](DECISIONS.md) for design choices.

## Setup

```bash
git clone <this repo> && cd mongo-annotation-lab
docker compose up --build        # starts MongoDB 7 and loads 300 items
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Usage

```bash
python -m annlab.labeling                          # rate items 1-5 + one-sentence rationale
python -m annlab.labeling --annotator second --second   # second annotator, 40 shared items
python -m annlab.run_experiments --second second   # classifier, active learning, kappa, charts
pytest                                             # tests (DB tests need MongoDB running)
```

Outputs are written to `results/`. `src/annlab/queries.py` holds the aggregation pipelines.

## Results

Not yet generated. Results require real human labels; this section is filled in from `results/report.json` after labeling.
