.PHONY: install dev check test hud listen voices download

install:     ## FRIDAY with full voice support
	pip install -e ".[voice]"

dev:
	pip install -e ".[voice,dev]"

check:       ## lint + types + tests (what CI runs)
	ruff check . && ruff format --check . && mypy friday && pytest

test:
	pytest

download:    ## male + female neural voices and Whisper (one time, ~300 MB)
	friday download female male

hud:         ## web HUD at http://localhost:8765
	friday serve

listen:      ## hands-free voice on this computer
	friday listen

voices:
	friday voices
