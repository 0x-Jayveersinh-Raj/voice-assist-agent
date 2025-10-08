from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str
    src_lang: str = "hi"
    tgt_lang: str = "en"


class LLMRequest(BaseModel):
    text: str
    provider: str | None = None


