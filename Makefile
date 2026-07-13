.PHONY: app apps serve pairings molecules data ingredients test

# One-command shortcut: build both web apps and serve them locally.
app: ; python scripts/launch.py

# Just (re)build the two HTML apps, no server.
apps: ; python scripts/launch.py --build-only

pairings:  ; python scripts/build_app.py
molecules: ; python scripts/build_molecule_widget.py

# Rebuild the underlying data tables (dataset, library, ingredients).
data:
	python scripts/build_dataset.py
	python scripts/build_ingredients.py

test: ; pytest
