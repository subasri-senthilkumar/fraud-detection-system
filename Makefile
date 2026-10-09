.PHONY: install data train test api dashboard seed up down

install:
	pip install -r requirements.txt

data:            ## synthetic demo dataset -> data/raw/transactions.csv
	python -m src.data_generation

train:           ## train, compare, select, save artifacts + reports
	python -m src.train --generate-if-missing

test:
	pytest -v

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

seed:            ## load demo predictions into the DB via the running API
	python scripts/seed_demo.py

up:
	docker compose up --build

down:
	docker compose down
