"""Yerel (kendi makinende çalışan) modeller. Ağır bağımlılıklar yalnızca kullanıldığında içe aktarılır.

tfidf_lr   Eğitimli temel model: karakter+kelime TF-IDF + lojistik regresyon (train bölümünden öğrenir)
embedding  Cümle gömmesi: 'zero_shot' (etiket açıklamasına benzerlik) veya 'lr' (train üzerinde LR; SetFit'siz)
nli        Sıfır örnekli NLI sınıflandırıcı (transformers zero-shot-classification)
gliclass   GLiClass encoder (sıfır örnekli, tek/çok etiketli) — pip install gliclass
laya       Laya encoder karar modeli — pip install laya
setfit     SetFit few-shot ince ayar — pip install setfit
"""
from __future__ import annotations

import math

import numpy as np

from ..core import Example, Prediction, ProviderError, TaskSpec
from ..prompts import QUESTION_NAME
from .base import Provider, match_label, normalize_probs


def tr_lower(s: str) -> str:
    """Türkçe'ye duyarlı küçük harf: 'I'->'ı', 'İ'->'i' (str.lower() bunu yanlış yapar)."""
    return s.replace("İ", "i").replace("I", "ı").lower()


def _softmax(x: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = (x - x.max()) / t
    e = np.exp(z)
    return e / e.sum()


def _pred_from_probs(keys: list[str], p: np.ndarray, kind: str = "native") -> Prediction:
    i = int(np.argmax(p))
    return Prediction(label=keys[i], probs={k: float(v) for k, v in zip(keys, p)},
                      confidence=float(p[i]), conf_kind=kind)


# ------------------------------------------------------------------------------------ TF-IDF + LR
class TfidfLRProvider(Provider):
    needs_training = True

    def prepare(self, task, train, shots=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        if not train:
            raise ProviderError("TF-IDF modeli için train bölümü gerekli")
        feats = FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, preprocessor=tr_lower)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, preprocessor=tr_lower)),
        ])
        clf = LogisticRegression(C=float(self.opt("C", 20.0)), max_iter=3000, class_weight="balanced")
        self._pipe = Pipeline([("f", feats), ("c", clf)])
        self._pipe.fit([e.text for e in train], [e.label for e in train])

    def predict(self, text, task, order=None):
        p = self._pipe.predict_proba([text])[0]
        classes = list(self._pipe.classes_)
        full = np.array([p[classes.index(k)] if k in classes else 0.0 for k in task.label_keys])
        full = full / full.sum()
        return _pred_from_probs(task.label_keys, full)


# ------------------------------------------------------------------------------------ Embedding
class EmbeddingProvider(Provider):
    """mode=zero_shot: etiket metnine kosinüs benzerliği | mode=lr: train gömmeleri üzerinde lojistik regresyon."""

    def prepare(self, task, train, shots=None):
        from sentence_transformers import SentenceTransformer

        self._mode = self.opt("mode", "lr")
        self._qp = self.opt("query_prefix", "")
        self._pp = self.opt("passage_prefix", "")
        self._st = SentenceTransformer(self.spec.model, device=self.opt("device", "cpu"))
        if self._mode == "lr":
            from sklearn.linear_model import LogisticRegression

            if not train:
                raise ProviderError("embedding(mode=lr) için train bölümü gerekli")
            self.needs_training = True
            X = self._enc([e.text for e in train], self._qp)
            self._clf = LogisticRegression(C=float(self.opt("C", 20.0)), max_iter=3000, class_weight="balanced")
            self._clf.fit(X, [e.label for e in train])
        else:
            texts = [f"{self._pp}{task.short_text(k)}: {task.labels[k]}" for k in task.label_keys]
            self._label_emb = self._enc(texts, "")

    def _enc(self, texts, prefix):
        return np.asarray(self._st.encode([prefix + t for t in texts], normalize_embeddings=True,
                                          show_progress_bar=False, batch_size=32))

    def predict(self, text, task, order=None):
        v = self._enc([text], self._qp)[0]
        if self._mode == "lr":
            p = self._clf.predict_proba([v])[0]
            classes = list(self._clf.classes_)
            full = np.array([p[classes.index(k)] if k in classes else 0.0 for k in task.label_keys])
            return _pred_from_probs(task.label_keys, full / full.sum())
        sims = self._label_emb @ v
        p = _softmax(sims, float(self.opt("temperature", 0.05)))
        return _pred_from_probs(task.label_keys, p, kind="derived")


# ------------------------------------------------------------------------------------ NLI
class NLIProvider(Provider):
    def prepare(self, task, train, shots=None):
        from transformers import pipeline

        dev = self.opt("device", -1)
        self._pipe = pipeline("zero-shot-classification", model=self.spec.model, device=dev)
        self._tmpl = self.opt("hypothesis_template", "Bu mesaj {} ile ilgilidir.")
        self._texts = {k: task.short_text(k) for k in task.label_keys}
        inv = {v: k for k, v in self._texts.items()}
        if len(inv) != len(self._texts):
            raise ProviderError("Kısa etiket metinleri benzersiz olmalı (labels.json 'short')")
        self._inv = inv

    def predict(self, text, task, order=None):
        out = self._pipe(text, list(self._texts.values()), hypothesis_template=self._tmpl, multi_label=False)
        probs = {self._inv[l]: float(s) for l, s in zip(out["labels"], out["scores"])}
        keys = task.label_keys
        p = np.array([probs.get(k, 0.0) for k in keys])
        return _pred_from_probs(keys, p / p.sum())


# ------------------------------------------------------------------------------------ GLiClass
class GLiClassProvider(Provider):
    def prepare(self, task, train, shots=None):
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer

        model = GLiClassModel.from_pretrained(self.spec.model)
        tok = AutoTokenizer.from_pretrained(self.spec.model)
        self._pipe = ZeroShotClassificationPipeline(
            model, tok, classification_type=self.opt("classification_type", "single-label"),
            device=self.opt("device", "cpu"))
        self._inv = {task.short_text(k): k for k in task.label_keys}
        if len(self._inv) != len(task.label_keys):
            raise ProviderError("Kısa etiket metinleri benzersiz olmalı (labels.json 'short')")

    def predict(self, text, task, order=None):
        # Etiketler modele tek bir istemde verilir: sıra sonucu etkileyebilir (dayanıklılık testi bunu ölçer)
        shown = [task.short_text(k) for k in (order or task.label_keys)]
        res = self._pipe(text, shown, threshold=0.0)[0]
        scores = {self._inv[r["label"]]: float(r["score"]) for r in res if r["label"] in self._inv}
        keys = task.label_keys
        p = np.array([scores.get(k, 0.0) for k in keys])
        if p.sum() <= 0:
            raise ProviderError("GLiClass skor döndürmedi")
        pred = _pred_from_probs(keys, p / p.sum())
        if not math.isclose(float(p.sum()), 1.0, abs_tol=0.05):
            pred.conf_kind = "derived"  # skorlar olasılık dağılımı değil, normalize edildi
        return pred


# ------------------------------------------------------------------------------------ Laya
class LayaProvider(Provider):
    def prepare(self, task, train, shots=None):
        from laya import Router

        self._router = Router()
        self._checkpoint = self.opt("checkpoint")  # örn. "multilingual"

    def predict(self, text, task, order=None):
        labels = task.ordered(order)
        questions = {QUESTION_NAME: {"type": "choice", "instructions": task.instruction, "criteria": labels}}
        kwargs = {"model": self._checkpoint} if self._checkpoint else {}
        result = self._router.predict(text, questions, **kwargs)
        ans = (result.get("answers") or {}).get(QUESTION_NAME)
        if not isinstance(ans, dict):
            raise ProviderError(f"Laya yanıtı beklenmedik: {str(result)[:160]}")
        keys = list(labels)
        label = match_label(ans.get("choice"), keys)
        probs = normalize_probs(ans.get("probabilities"), keys)
        if label is None and probs:
            label = max(probs, key=probs.get)
        if label is None:
            raise ProviderError("Laya geçerli etiket döndürmedi")
        return Prediction(label=label, probs=probs, confidence=(probs or {}).get(label), conf_kind="native")


# ------------------------------------------------------------------------------------ SetFit
class SetFitProvider(Provider):
    """Few-shot: train bölümünden etiket başına k_shot örnekle SetFit ince ayarı (CPU'da dakikalar sürebilir)."""

    needs_training = True

    def prepare(self, task, train, shots=None):
        import random

        from datasets import Dataset
        from setfit import SetFitModel, Trainer, TrainingArguments

        k = int(self.opt("k_shot", 8))
        rng = random.Random(int(self.opt("seed", 0)))
        keys = task.label_keys
        picked: list[Example] = []
        for lab in keys:
            pool = sorted((e for e in train if e.label == lab), key=lambda e: e.id)
            rng.shuffle(pool)
            picked += pool[:k]
        self._keys = keys
        ds = Dataset.from_dict({"text": [e.text for e in picked], "label": [keys.index(e.label) for e in picked]})
        self._model = SetFitModel.from_pretrained(self.spec.model)
        args = TrainingArguments(batch_size=int(self.opt("batch_size", 16)),
                                 num_epochs=int(self.opt("num_epochs", 1)),
                                 num_iterations=int(self.opt("num_iterations", 10)),
                                 show_progress_bar=False)
        Trainer(model=self._model, args=args, train_dataset=ds).train()
        self.trained_on = len(picked)

    def predict(self, text, task, order=None):
        p = np.asarray(self._model.predict_proba([text]))[0]
        p = np.asarray(p, dtype=float)
        return _pred_from_probs(self._keys, p / p.sum())
