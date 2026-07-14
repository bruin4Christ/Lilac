.PHONY: app apps serve compose pairings molecules hub data ingredients test

# One-command shortcut: build the web apps + hub and serve them locally.
app: ; python scripts/launch.py

# Just (re)build the HTML apps + hub, no server.
apps: ; python scripts/launch.py --build-only

compose:   ; python scripts/build_compose_app.py
pairings:  ; python scripts/build_app.py
molecules: ; python scripts/build_molecule_widget.py
hub:       ; python scripts/build_hub.py

# Rebuild the underlying data tables (dataset, library, ingredients).
data:
	python scripts/build_dataset.py
	python scripts/build_ingredients.py

test: ; pytest
