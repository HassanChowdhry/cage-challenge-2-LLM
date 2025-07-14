from .gemini import GeminiBackend
from .huggingface import LocalHFBackend
from .model import LLMBackend

def create_backend(backend_type: str) -> LLMBackend:
    logger.info("Creating Backend")    
    backend_map = {
        "local": LocalHFBackend,
        "gemini": GeminiBackend,
    }
    return backend_map[backend_type]