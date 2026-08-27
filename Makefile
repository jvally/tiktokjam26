PYTHON ?= python3

.PHONY: demo test evaluate serve check
demo:
	$(PYTHON) -m shopping_copilot demo
test:
	$(PYTHON) -m unittest discover -s tests -v
evaluate:
	$(PYTHON) -m shopping_copilot evaluate
serve:
	$(PYTHON) -m shopping_copilot serve
check: test evaluate
