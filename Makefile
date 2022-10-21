NEW_LAB_NUMBER := $(shell ls labs/ | cut -d "b" -f 2 | cut -d . -f 1 | sort | tail -1 | xargs expr 1 +)

new-lab:
	@cp labs/lab1.html labs/lab${NEW_LAB_NUMBER}.html

run-jupyter:
	@docker-compose up