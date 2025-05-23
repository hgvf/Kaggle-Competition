FROM huggingface/transformers-all-latest-gpu

WORKDIR /data

RUN \
apt-get update -y && \
pip install -U openai-whisper && \
git clone https://github.com/AndreyGuzhov/AudioCLIP.git && \
git clone https://github.com/microsoft/unilm.git

CMD [ "python3" ]
