"""
Kinesis Reactivation — LLM provider initialization (Mistral AI primary, OpenAI fallback).
"""

from __future__ import annotations

import os

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from src.config import _dbg


def get_embeddings():
    api_key = os.getenv("MISTRAL_API_KEY") or ""
    if not api_key or api_key == "your-mistral-key-here":
        return None
    return MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)


def get_llm():
    print("Trying to load Mistral LLM...")
    mistral_key = os.getenv("MISTRAL_API_KEY") or ""
    if mistral_key and mistral_key != "your-mistral-key-here":
        try:
            llm = ChatMistralAI(model="mistral-large-latest", temperature=0.3, mistral_api_key=mistral_key)
            print("LLM loaded: Mistral")
            return llm
        except Exception as e:
            print(f"Mistral LLM failed: {e}")
    else:
        print("LLM failed — key missing or import error")
    openai_key = os.getenv("OPENAI_API_KEY") or ""
    if openai_key and openai_key != "your-actual-key-here" and ChatOpenAI is not None:
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=openai_key)
            print("LLM loaded: OpenAI fallback")
            return llm
        except Exception as e:
            print(f"OpenAI fallback failed: {e}")
    return None


def get_llm_evaluator():
    """Evaluator node: temperature=0.0 for determinism."""
    mistral_key = os.getenv("MISTRAL_API_KEY") or ""
    if mistral_key and mistral_key != "your-mistral-key-here":
        try:
            return ChatMistralAI(model="mistral-large-latest", temperature=0.0, max_tokens=400, mistral_api_key=mistral_key)
        except Exception:
            pass
    if ChatOpenAI is not None:
        openai_key = os.getenv("OPENAI_API_KEY") or ""
        if openai_key and openai_key != "your-actual-key-here":
            try:
                return ChatOpenAI(model="gpt-4o", temperature=0.0, max_tokens=400, openai_api_key=openai_key)
            except Exception:
                pass
    return None


def get_llm_writer():
    """Writer node: Mistral mistral-large-latest, temperature=0.18, max_tokens=400."""
    mistral_key = os.getenv("MISTRAL_API_KEY") or ""
    # #region agent log
    _dbg("get_llm_writer entry", {"mistral_has_key": bool(mistral_key), "mistral_is_placeholder": mistral_key == "your-mistral-key-here"}, "H2")
    # #endregion
    if mistral_key and mistral_key != "your-mistral-key-here":
        try:
            llm = ChatMistralAI(
                model="mistral-large-latest",
                temperature=0.18,
                max_tokens=400,
                mistral_api_key=mistral_key,
            )
            # #region agent log
            _dbg("get_llm_writer Mistral ok", {"returned": "ChatMistralAI"}, "H2")
            # #endregion
            return llm
        except Exception as e:
            # #region agent log
            _dbg("get_llm_writer Mistral failed", {"err": str(e)[:200]}, "H2")
            # #endregion
            pass
    if ChatOpenAI is not None:
        openai_key = os.getenv("OPENAI_API_KEY") or ""
        if openai_key and openai_key != "your-actual-key-here":
            try:
                llm = ChatOpenAI(model="gpt-4o", temperature=0.18, max_tokens=400, openai_api_key=openai_key)
                # #region agent log
                _dbg("get_llm_writer OpenAI fallback ok", {}, "H2")
                # #endregion
                return llm
            except Exception as e:
                # #region agent log
                _dbg("get_llm_writer OpenAI failed", {"err": str(e)[:200]}, "H2")
                # #endregion
                pass
    # #region agent log
    _dbg("get_llm_writer returning None", {}, "H2")
    # #endregion
    return None
