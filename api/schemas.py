from pydantic import BaseModel, Field


class EntradaFlor(BaseModel):
    sepal_length: float = Field(..., ge=0, examples=[5.1])
    sepal_width: float = Field(..., ge=0, examples=[3.5])
    petal_length: float = Field(..., ge=0, examples=[1.4])
    petal_width: float = Field(..., ge=0, examples=[0.2])


class Predicao(BaseModel):
    especie: str
    probabilidades: dict
    modelo: str
    versao: int
