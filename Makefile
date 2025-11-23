.PHONY: all install test export-pdf export-pdf-show export-all clean

all: install submodules-install

install: install-additional
	@echo "Verificando Python..."
	@python3 --version || (echo "Python não encontrado. Por favor, instale o Python primeiro." && exit 1)
	@echo "Criando ambiente virtual..."
	python3 -m venv .venv
	@echo "Instalando dependências..."
	.venv/bin/pip install -r requirements.txt
	@echo "Instalação concluída!"
	@echo "Instalando componentes adicionais..."

install-additional:
	@sudo apt update
	@sudo apt install texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y

clean:
	rm -rf .venv
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

submodules-install:
	git submodule update --init --recursive

submodules-update:
	git submodule update --remote

run:
	@echo "Executando Makefile em notebooks/"
	@$(MAKE) -C notebooks
