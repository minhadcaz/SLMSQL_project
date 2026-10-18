"""Author SLM-SQL baseline and leakage-safe CSC LoRA experiments.

Heavy GPU dependencies are imported only by the generation/training commands.
"""

GENERATOR_MODEL = "cycloneboy/SLM-SQL-1.5B"
MERGER_MODEL = "cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct"
PROTOCOL_VERSION = "slm-sql-csc-v1"
