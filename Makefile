PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: install data pipeline lexical semantic eval predict test clean

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

## Q1: one-command rebuild of the full data pipeline (download -> clean -> temporal split -> feature store)
data:
	$(PY) -m ire_rec.build_pipeline --rebuild

## everything from raw data (assumes data/raw/ already populated)
pipeline:
	$(PY) -m ire_rec.build_pipeline

## Q2: BM25 lexical candidate generation + recall@K
lexical:
	$(PY) -m ire_rec.retrieval.run_bm25

## Q3: embedding / ANN semantic candidate generation + recall@K
semantic:
	$(PY) -m ire_rec.retrieval.run_semantic

## Q4: offline evaluation harness on BM25 + semantic results
eval:
	$(PY) -m ire_rec.evaluation.run_eval

## Q5: generate Codabench prediction files
predict:
	$(PY) -m ire_rec.make_predictions

## run unit/integration tests (incl. no-future-click-leakage assertion, Q9)
test:
	$(PY) -m pytest tests/ -v

clean:
	rm -rf data/processed __pycache__ .pytest_cache