from collections import Counter

import pytest

from clfbench.core import Example, shuffled_order
from clfbench.data import auto_tags, dataset_fingerprint, load_dataset, parse_seed_line, select


def test_seed_dataset_shape(task_and_data):
    task, ex = task_and_data
    assert len(ex) >= 300
    assert set(task.labels) == {e.label for e in ex}
    for split in ("train", "dev", "test"):
        c = Counter(e.label for e in ex if e.split == split)
        assert set(c) == set(task.labels), f"{split} bölümünde eksik sınıf var"
        assert min(c.values()) >= 5


def test_no_text_leaks_between_splits(task_and_data):
    _, ex = task_and_data
    by_text = {}
    for e in ex:
        by_text.setdefault(e.text, set()).add(e.split)
    assert all(len(s) == 1 for s in by_text.values())
    assert len({e.id for e in ex}) == len(ex)


def test_select_never_returns_train(task_and_data):
    _, ex = task_and_data
    for spec in ("test", "dev", "dev+test"):
        assert all(e.split != "train" for e in select(ex, spec))
    with pytest.raises(ValueError):
        select(ex, "train")


def test_seed_parsing_and_auto_tags():
    assert parse_seed_line("# yorum") is None and parse_seed_line("   ") is None
    tags, text = parse_seed_line("[ascii,typo] kargom nerde")
    assert tags == ("ascii", "typo") and text == "kargom nerde"
    assert "short" in auto_tags("Fiyatı ne kadar?", ())
    assert "long" in auto_tags(" ".join(["kelime"] * 30), ())


def test_fingerprint_changes_with_data(task_and_data):
    task, ex = task_and_data
    fp = dataset_fingerprint(task, ex)
    assert fp == dataset_fingerprint(task, list(reversed(ex)))  # sıradan bağımsız
    changed = ex[:-1] + [Example(id=ex[-1].id, text=ex[-1].text + "!", label=ex[-1].label, split=ex[-1].split)]
    assert dataset_fingerprint(task, changed) != fp


def test_shuffled_order_is_deterministic_and_a_permutation():
    keys = list("abcdefgh")
    a = shuffled_order(keys, 0, "ex1")
    assert a == shuffled_order(keys, 0, "ex1") and sorted(a) == keys
    assert a != shuffled_order(keys, 0, "ex2") or a != shuffled_order(keys, 1, "ex1")


def test_custom_csv_and_jsonl_loading(tmp_path):
    import json

    (tmp_path / "d.labels.json").write_text(json.dumps(
        {"instruction": "sınıfla", "labels": {"pos": "olumlu", "neg": "olumsuz"}}), encoding="utf-8")
    (tmp_path / "d.csv").write_text("text,label,split\niyi,pos,train\nkötü,neg,test\n", encoding="utf-8")
    task, ex = load_dataset(str(tmp_path / "d.csv"))
    assert [e.label for e in ex] == ["pos", "neg"] and ex[1].split == "test"
    (tmp_path / "bad.csv").write_text("text,label\nx,unknown\n", encoding="utf-8")
    (tmp_path / "bad.labels.json").write_text((tmp_path / "d.labels.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="tanımsız etiket"):
        load_dataset(str(tmp_path / "bad.csv"))
