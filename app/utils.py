import logging
import hashlib

logger = logging.getLogger(__name__)

# load files into list
def load_files(file_names: list) -> list:
    texts = []
    for file_name in file_names:
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                text = f.read()
                
                if not text.strip():
                    logger.info(f'Empty file: {file_name}')

                texts.append(text)
        except Exception as e:
            logger.error(f'Failed to load file: {file_name}')

    return texts

# логирование промпта
def log_prompt(prompt: str) -> str:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    logger.debug(f"Промпт (хэш: {prompt_hash}) отправлен в LLM")
    return prompt_hash