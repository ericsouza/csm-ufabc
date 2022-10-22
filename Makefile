NEW_LAB_NUMBER := $(shell ls labs/ | cut -d "b" -f 2 | cut -d . -f 1 | sort | tail -1 | xargs expr 1 +)

new-lab:
	@cp labs/lab3.html labs/lab${NEW_LAB_NUMBER}.html

run-jupyter:
	@docker-compose up

create-lab-dist:
	@echo "Criando zip do $$LAB para upload..."
	@echo "Copiando arquivos..."
	@mkdir -p dist_labs/${LAB}/files/${LAB} && cp labs_notebooks/${LAB}_notebook.ipynb dist_labs/${LAB}/${LAB}_notebook.ipynb && cp -r labs_notebooks/files/${LAB} dist_labs/${LAB}/files/
	@echo "Gerando arquivo zip..."
	@cd dist_labs && zip -q -r ${LAB}.zip ${LAB} && cd ..
	@rm -rf dist_labs/${LAB}
	@echo "zip salvo em dist_labs/${LAB}"
