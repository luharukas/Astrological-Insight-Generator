"""
FastAPI application for the Astrological Insight Generator.
Provides REST endpoints for health check, zodiac lookup, and astrological predictions.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
import uvicorn

from astrology.input_parser import parse_date, parse_time, parse_location
from astrology.zodiac import get_zodiac_sign
from llmkit.llm import invoke_llm_with_fallback


class PredictRequest(BaseModel):
    name: str
    birth_date: str  # YYYY-MM-DD
    birth_time: str  # HH:MM
    birth_place: str
    language: str = 'en'


class LLMResponse(BaseModel):
    insight: str


class PredictResponse(BaseModel):
    zodiac: str
    insight: str
    language: str
    cached: bool
    timestamp: str


app = FastAPI(title='Astrological Insight API')
_predict_cache = {}


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/zodiac/{date_str}')
def zodiac(date_str: str):
    try:
        d = parse_date(date_str)
        sign = get_zodiac_sign(d)
        return {'zodiac': sign}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    # Validate input and parse components
    try:
        bd = parse_date(req.birth_date)
        _ = parse_time(req.birth_time)
        _ = parse_location(req.birth_place)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Determine zodiac sign
    try:
        sign = get_zodiac_sign(bd)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Prepare cache entries keyed by birth date, time, and place
    base_key = (req.birth_date, req.birth_time, req.birth_place)
    if base_key not in _predict_cache:
        _predict_cache[base_key] = {}
    lang_cache = _predict_cache[base_key]
    # Check if English insight is already cached
    en_cached = 'en' in lang_cache
    # Generate English insight with LLM if not cached
    if not en_cached:
        prompt = (
            f"Generate a personalized daily astrological insight.\n"
            f"User: {req.name}\n"
            f"Birth date: {req.birth_date}\n"
            f"Birth time: {req.birth_time}\n"
            f"Location: {req.birth_place}\n"
            f"Zodiac sign: {sign}\n"
            f"Provide a concise, friendly insight."
        )
        try:
            llm_resp = invoke_llm_with_fallback(
                prompt=prompt,
                output_pydantic=LLMResponse,
                system_prompt='You are a friendly astrological advisor.',
                provider='all',
            )
            lang_cache['en'] = llm_resp.insight
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM invocation failed: {e}")
    insight_en = lang_cache['en']

    # Translate to requested language if needed, caching translations
    if req.language != 'en':
        if req.language not in lang_cache:
            from astrology.translator import translate
            try:
                translated = translate(
                    texts=insight_en,
                    target_lang=req.language,
                    src_lang='en',
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except NotImplementedError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Translation failed: {e}")
            lang_cache[req.language] = translated
        insight = lang_cache[req.language]
    else:
        insight = insight_en
    # cached flag indicates whether LLM call was skipped
    cached = en_cached

    # Build response
    timestamp = datetime.now(timezone.utc).isoformat()
    return PredictResponse(
        zodiac=sign,
        insight=insight,
        language=req.language,
        cached=cached,
        timestamp=timestamp,
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0',port=8000)

