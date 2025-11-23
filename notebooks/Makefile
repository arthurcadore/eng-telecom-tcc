.PHONY: receiver transmitter

all: receiver transmitter

receiver:
	@echo "Deletando arquivos antigos..."
	@rm -f ../out/notebook_receiver.log
	@echo "Executando scripts do receptor..." | tee ../out/notebook_receiver.log
	@{ \
		python3 receiver/convolutional.py; \
		python3 receiver/detection.py; \
		python3 receiver/viterbi.py; \
	} 2>&1 | tee -a ../out/notebook_receiver.log

transmitter:
	@echo "Deletando arquivos antigos..."
	@rm -f ../out/notebook_transmitter.log
	@echo "Executando scripts do transmissor..." | tee ../out/notebook_transmitter.log
	@{ \
		python3 transmitter/carrier.py; \
		python3 transmitter/modulador.py; \
	} 2>&1 | tee -a ../out/notebook_transmitter.log