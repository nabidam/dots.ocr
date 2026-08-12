# dots.ocr support is natively integrated into vLLM since v0.11.0.
# vllm/vllm-openai:v0.13.0 works out-of-the-box without requiring code modifications or custom builds.
FROM vllm/vllm-openai:v0.13.0

EXPOSE 8000

ENTRYPOINT ["vllm", "serve"]
CMD [ \
  "rednote-hilab/dots.mocr", \
  "--host", "0.0.0.0", \
  "--port", "8000", \
  "--tensor-parallel-size", "1", \
  "--gpu-memory-utilization", "0.9", \
  "--chat-template-content-format", "string", \
  "--served-model-name", "model", \
  "--trust-remote-code" \
]
