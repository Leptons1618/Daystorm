.RECIPEPREFIX = >
PY = .venv/bin/python

.PHONY: test lint gate demo clean
test:
> $(PY) -m pytest -q

lint:
> $(PY) -m ruff check src tests

gate:
> $(PY) -m daystorm.train.stage_a --overfit 8 --steps 300

demo:
> $(PY) -m daystorm.data.synthetic --demo

clean:
> rm -rf .pytest_cache .ruff_cache **/__pycache__
