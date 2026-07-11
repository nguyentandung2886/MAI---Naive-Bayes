# One command per stage so anyone who clones can rebuild everything.
# WHY a Makefile: it documents the canonical commands in an executable form,
# so "how do I run this?" has a single source of truth. On Windows without
# `make`, just run the underlying `python ...` line by hand.

PYTHON ?= python

.PHONY: data train app test

data:        ## Download ViHSD into data/raw/ (requires huggingface-cli login)
	$(PYTHON) scripts/download_data.py

train:       ## Fit the pipeline and write models/pipeline.pkl
	$(PYTHON) src/train.py

app:         ## Launch the Streamlit demo locally
	streamlit run app/streamlit_app.py

test:        ## Run the test suite
	pytest
