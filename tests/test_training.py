from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import torch
from transformers import AutoModel, AutoModelForMaskedLM

from make_manual_notebook import notebook
from uralic_lm.common import read_json, read_jsonl
from uralic_lm.training import TrainConfig, run_experiment, evaluate_run, environment
from uralic_lm.inspection import Inspector


def test_saved_config_resume_and_model_loading(tiny_model, prepared, tmp_path):
    config = TrainConfig(model=str(tiny_model[0]), data=str(prepared[0]), device="cpu", precision="fp32",
                         max_length=16, batch_size=2, accumulation_steps=2, epochs=2, cpu_threads=1,
                         logging_steps=1, checkpoint_steps=2, eval_max_blocks=1, train_diagnostic_blocks=1)
    paused = tmp_path / "paused"
    continuous = tmp_path / "continuous"
    run_experiment(config, output=paused, stop_after_steps=1)
    assert read_json(paused / "status.json")["status"] == "paused"
    assert read_json(paused / "config.json") == asdict(config)
    run_experiment(config, resume=paused)
    run_experiment(config, output=continuous)
    assert read_json(paused / "status.json")["status"] == "completed"
    assert any(e["event"] == "resume" for e in read_jsonl(paused / "events.jsonl"))
    a = AutoModelForMaskedLM.from_pretrained(paused / "best_model").state_dict()
    b = AutoModelForMaskedLM.from_pretrained(continuous / "best_model").state_dict()
    assert all(torch.equal(a[key], b[key]) for key in a)
    assert AutoModel.from_pretrained(paused / "best_model").config.hidden_size == 16
    assert read_json(paused / "summary.json")["adapted"]["test"]["nio"]["masked_tokens"] > 0
    summary_before = (paused / "summary.json").read_bytes()
    assert read_json(evaluate_run(paused))["adapted"]["test"]["nio"]["masked_tokens"] > 0
    assert (paused / "summary.json").read_bytes() == summary_before
    inspector = Inspector(paused, device="cpu")
    try:
        assert len(inspector.predict("ӱ ӥ ŋuəɁ")["predictions"]) == 5
        assert inspector.candidates("word ", " end", ["ӱ ӥ"])[0]["subtokens"] == 2
        with pytest.raises(ValueError, match="no non-special"):
            inspector.predict("")
        with pytest.raises(ValueError, match="shorter excerpt"):
            inspector.predict("word " * 20)
    finally:
        inspector.close()


def test_resume_rejects_changed_experiment(tiny_model, prepared, tmp_path):
    config = TrainConfig(model=str(tiny_model[0]), data=str(prepared[0]), device="cpu", precision="fp32",
                         max_length=16, batch_size=2, accumulation_steps=2, epochs=2, cpu_threads=1,
                         eval_max_blocks=1, train_diagnostic_blocks=1)
    run = run_experiment(config, output=tmp_path / "run", stop_after_steps=1)
    config.learning_rate *= 2
    with pytest.raises(ValueError, match="settings changed"):
        run_experiment(config, resume=run)


def test_notebook_is_valid_and_cells_compile():
    nb = nbformat.from_dict(notebook())
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile("".join(cell.source), "notebook", "exec")


def test_environment_records_active_version_with_inherited_packages(monkeypatch):
    monkeypatch.setattr("importlib.metadata.distributions", lambda: [
        SimpleNamespace(metadata={"Name": "example"}, version="2.0"),
        SimpleNamespace(metadata={"Name": "example"}, version="1.0")])
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2.0")
    assert environment()["packages"] == {"example": "2.0"}
