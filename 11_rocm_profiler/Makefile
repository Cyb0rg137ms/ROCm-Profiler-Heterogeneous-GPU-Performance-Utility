.PHONY: test run clean

PYTHON=D:\project\attachments\thinkingLLM\python311\tools\python.exe

test:
	$(PYTHON) -m pytest tests/ -v

run:
	$(PYTHON) run_profiler.py

clean:
	rmdir /s /q profiler\__pycache__ 2>nul || exit 0
	rmdir /s /q tests\__pycache__ 2>nul || exit 0
	rmdir /s /q .pytest_cache 2>nul || exit 0
