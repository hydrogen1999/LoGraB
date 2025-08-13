# ==== General configuration (able to override when calling `make`) ====
PYTHON      ?= .venv/bin/python
PIP         ?= .venv/bin/pip
DATASETS    ?= Cora Citeseer PubMed ogbn-arxiv

# Instance generation params
STRATEGY    ?= d-hop
D           ?= 2
K           ?= 32
SIGMA       ?= 0.05
P           ?= 0.8
SEED        ?= 42
LAP         ?= unnormalized
ROOT_DIR    ?= instances

# Train/Eval params
EPOCHS      ?= 100
BATCH       ?= 32
LR          ?= 1e-2
WD          ?= 5e-4
NUM_WORKERS ?= 0
SCORER      ?= cosine

.PHONY: venv install setup gen-data gen-% eval eval-% run clean

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

setup: install
	mkdir -p scripts
	chmod +x scripts/*.sh || true

# --- Generate + splits only---
gen-data: setup
	STRATEGY=$(STRATEGY) D=$(D) K=$(K) SIGMA=$(SIGMA) P=$(P) SEED=$(SEED) LAP=$(LAP) ROOT_DIR=$(ROOT_DIR) \
	DATASETS="$(DATASETS)" \
	scripts/gen_data_all.sh

gen-%: setup
	DATASETS="$*" STRATEGY=$(STRATEGY) D=$(D) K=$(K) SIGMA=$(SIGMA) P=$(P) SEED=$(SEED) LAP=$(LAP) ROOT_DIR=$(ROOT_DIR) \
	scripts/gen_data_all.sh

# ---  Evaluate only---
eval: setup
	EPOCHS=$(EPOCHS) BATCH=$(BATCH) LR=$(LR) WD=$(WD) NUM_WORKERS=$(NUM_WORKERS) SCORER=$(SCORER) \
	STRATEGY=$(STRATEGY) D=$(D) K=$(K) SIGMA=$(SIGMA) P=$(P) SEED=$(SEED) LAP=$(LAP) ROOT_DIR=$(ROOT_DIR) \
	DATASETS="$(DATASETS)" \
	scripts/eval_all.sh

eval-%: setup
	DATASETS="$*" EPOCHS=$(EPOCHS) BATCH=$(BATCH) LR=$(LR) WD=$(WD) NUM_WORKERS=$(NUM_WORKERS) SCORER=$(SCORER) \
	STRATEGY=$(STRATEGY) D=$(D) K=$(K) SIGMA=$(SIGMA) P=$(P) SEED=$(SEED) LAP=$(LAP) ROOT_DIR=$(ROOT_DIR) \
	scripts/eval_all.sh

# --- gen-data -> eval ---
run: gen-data eval

clean:
	rm -rf .venv artifacts