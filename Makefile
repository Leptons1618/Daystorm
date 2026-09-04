.RECIPEPREFIX = >
PY = .venv/bin/python

.PHONY: test lint gate demo eval bench serve clean
test:
> $(PY) -m pytest -q

lint:
> $(PY) -m ruff check src tests

gate:
> $(PY) -m daystorm.train.stage_a --overfit 8 --steps 300

demo:
> $(PY) -m daystorm.data.synthetic --demo

eval:
> $(PY) -m daystorm.eval.harness --ckpt ckpt/stage_a --max-eval 16

bench:
> $(PY) -m daystorm.bench.latency
> $(PY) -m daystorm.bench.end_to_end

serve:
> DAYSTORM_CKPT=ckpt/stage_a $(PY) -m uvicorn daystorm.serve.app:app --port 8000

clean:
> rm -rf .pytest_cache .ruff_cache **/__pycache__
