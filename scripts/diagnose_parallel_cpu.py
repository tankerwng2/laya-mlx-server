"""Print evaluated stages of the tiny CPU parallel model after a native CI crash."""

import importlib.metadata
import platform
import runpy
from pathlib import Path

import mlx.core as mx

from laya_mlx.model import DecisionModel, parallel_option_masks


def evaluate(label, value):
    print(f"Evaluating {label}", flush=True)
    mx.eval(value)
    print(f"Completed {label}", flush=True)
    return value


def main():
    print(platform.platform(), flush=True)
    print("MLX", importlib.metadata.version("mlx"), flush=True)
    mx.set_default_device(mx.cpu)
    fixtures = runpy.run_path(str(Path(__file__).parents[1] / "tests/test_parallel_layout.py"))
    mx.random.seed(17)
    model = DecisionModel(fixtures["config"](), {"head_layers": 2, "option_layout": "parallel"})
    model.set_dtype(mx.float16)
    model.eval()
    evaluate("parameters", model.parameters())
    batch = {key: mx.array(value) for key, value in fixtures["sequence_batch"]().items()}
    evaluate("inputs", batch)
    x = evaluate("embedding", model.encoder.embeddings(batch["input_ids"]))
    x = evaluate("FP32 promotion", x.astype(mx.float32))
    masks = evaluate(
        "masks",
        parallel_option_masks(
            batch["attention_mask"],
            batch["position_ids"],
            batch["option_ids"],
            model.encoder.config.local_attention,
        ),
    )
    for index, layer in enumerate(model.encoder.layers):
        x = evaluate(
            f"encoder {index}", layer(x, masks[layer.attention_type], batch["position_ids"])
        )
    x = evaluate("encoder norm", model.encoder.final_norm(x))
    x = evaluate("type embedding", x + model.type_emb(batch["qtype"])[:, None, :])
    for index, layer in enumerate(model.head.layers):
        x = evaluate(f"head {index}", layer(x, batch["attention_mask"][:, None, None, :]))
    evaluate("full prediction", model(**batch))


if __name__ == "__main__":
    main()
