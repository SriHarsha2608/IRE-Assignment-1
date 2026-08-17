.PHONY: data pipeline lexical semantic eval predict test clean

## Q1: one-command rebuild of the full data pipeline (download -> clean -> temporal split -> feature store)
data:
	python -m ire_rec.build_pipeline --rebuild

## everything from raw data (assumes data/raw/ already populated)
pipeline:
	python -m ire_rec.build_pipeline

## Q2: BM25 lexical candidate generation + recall@K
lexical:
	python -m ire_rec.retrieval.run_bm25

## Q3: embedding / ANN semantic candidate generation + recall@K
semantic:
	python -m ire_rec.retrieval.run_semantic

## Q4: offline evaluation harness on BM25 + semantic results
eval:
	python -m ire_rec.evaluation.run_eval

## Q5: generate Codabench prediction files
predict:
	python -m ire_rec.make_predictions

## run unit/integration tests (incl. no-future-click-leakage assertion, Q9)
test:
	python -m pytest tests/ -v

clean:
	rm -rf data/processed __pycache__ .pytest_cache