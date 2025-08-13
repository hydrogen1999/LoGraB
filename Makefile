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


# ==== Baselines nâng cao ====
K_RECON     ?= 10
HOPS_SEAL   ?= 2
EPOCHS_SEAL ?= 20
BATCH_SEAL  ?= 64

.PHONY: recon-eigsync recon-eigsync-% linkpred-seal linkpred-seal-%

recon-eigsync: setup
	@for ds in $(DATASETS); do \
	  TAG=$$( \
	    $(PYTHON) - <<'PY' \
strategy="$(STRATEGY)"; d=int("$(D)"); k=int("$(K)"); sigma=float("$(SIGMA)"); p=float("$(P)"); lap="$(LAP)"; \
print(f"{strategy}_d{d}_k{k}_s{sigma:.2f}_p{p}_{lap[0]}") \
PY \
	  ); \
	  INSTANCE="$(ROOT_DIR)/$$ds/$$TAG"; \
	  ARTI="artifacts/$$ds/$$TAG"; mkdir -p "$$ARTI" "artifacts/logs/$$ds/$$TAG"; \
	  echo "[eigsync] $$INSTANCE"; \
	  scripts/gen_pred_eigsync.sh "$$INSTANCE" "$$ARTI/pred_eigsync.txt" "$(K_RECON)" | tee "artifacts/logs/$$ds/$$TAG/07_eigsync_gen.log"; \
	  $(PYTHON) -m lograb eval --task reconstruct --instance "$$INSTANCE" --pred "$$ARTI/pred_eigsync.txt" | tee "artifacts/logs/$$ds/$$TAG/08_eigsync_eval.log"; \
	done

recon-eigsync-%: setup
	@ds=$*; \
	TAG=$$( \
	  $(PYTHON) - <<'PY' \
strategy="$(STRATEGY)"; d=int("$(D)"); k=int("$(K)"); sigma=float("$(SIGMA)"); p=float("$(P)"); lap="$(LAP)"; \
print(f"{strategy}_d{d}_k{k}_s{sigma:.2f}_p{p}_{lap[0]}") \
PY \
	); \
	INSTANCE="$(ROOT_DIR)/$$ds/$$TAG"; \
	ARTI="artifacts/$$ds/$$TAG"; mkdir -p "$$ARTI" "artifacts/logs/$$ds/$$TAG"; \
	echo "[eigsync] $$INSTANCE"; \
	scripts/gen_pred_eigsync.sh "$$INSTANCE" "$$ARTI/pred_eigsync.txt" "$(K_RECON)" | tee "artifacts/logs/$$ds/$$TAG/07_eigsync_gen.log"; \
	$(PYTHON) -m lograb eval --task reconstruct --instance "$$INSTANCE" --pred "$$ARTI/pred_eigsync.txt" | tee "artifacts/logs/$$ds/$$TAG/08_eigsync_eval.log"

linkpred-seal: setup
	@for ds in $(DATASETS); do \
	  TAG=$$( \
	    $(PYTHON) - <<'PY' \
strategy="$(STRATEGY)"; d=int("$(D)"); k=int("$(K)"); sigma=float("$(SIGMA)"); p=float("$(P)"); lap="$(LAP)"; \
print(f"{strategy}_d{d}_k{k}_s{sigma:.2f}_p{p}_{lap[0]}") \
PY \
	  ); \
	  INSTANCE="$(ROOT_DIR)/$$ds/$$TAG"; \
	  mkdir -p "artifacts/logs/$$ds/$$TAG"; \
	  echo "[seal] $$INSTANCE"; \
	  scripts/eval_linkpred_seal.sh "$$INSTANCE" "$(HOPS_SEAL)" "$(EPOCHS_SEAL)" "$(BATCH_SEAL)" | tee "artifacts/logs/$$ds/$$TAG/09_seal_eval.log"; \
	done

linkpred-seal-%: setup
	@ds=$*; \
	TAG=$$( \
	  $(PYTHON) - <<'PY' \
strategy="$(STRATEGY)"; d=int("$(D)"); k=int("$(K)"); sigma=float("$(SIGMA)"); p=float("$(P)"); lap="$(LAP)"; \
print(f"{strategy}_d{d}_k{k}_s{sigma:.2f}_p{p}_{lap[0]}") \
PY \
	); \
	INSTANCE="$(ROOT_DIR)/$$ds/$$TAG"; \
	mkdir -p "artifacts/logs/$$ds/$$TAG"; \
	echo "[seal] $$INSTANCE"; \
	scripts/eval_linkpred_seal.sh "$$INSTANCE" "$(HOPS_SEAL)" "$(EPOCHS_SEAL)" "$(BATCH_SEAL)" | tee "artifacts/logs/$$ds/$$TAG/09_seal_eval.log"