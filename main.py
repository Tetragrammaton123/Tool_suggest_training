import argparse
import glob
import json
import os
from dotenv import load_dotenv
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional
from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart

from tool_suggest import ToolSuggestClient, ToolSuggestConfig, LocalBackendConfig
from tool_suggest.services.repository import JSONFileRepository
from tool_suggest.services.formatter import SampleFormatter
from tool_suggest.services.suggester import KNNSuggester
from tool_suggest.services.embedder.openai import OpenAIEmbedder
from tool_suggest.services.selector import GreedySelector
from tool_suggest.models import Sample

def user_msg(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def assistant_msg(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _tqdm(it, **kwargs):
    if tqdm is None:
        return it
    return tqdm(it, **kwargs)


# ---------- SGD reading ----------
def iter_dialogue_files(split_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(split_dir, "dialogues_*.json")))


def load_json(path: str) -> Any:
    try:
        import orjson  # type: ignore
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def intents_from_user_turn(turn: dict[str, Any]) -> list[str]:
    intents: list[str] = []
    for frame in turn.get("frames", []):
        intent = (frame.get("state") or {}).get("active_intent")
        if intent and intent != "NONE":
            intents.append(intent)
    return sorted(set(intents))


def make_samples_from_dialogue(
    dialogue: dict[str, Any],
    *,
    max_history_messages: int,
    label_mode: str,  # "first" | "all"
    add_services_hint: bool,
) -> list[Any]:
    out: list[Any] = []
    ctx: list[Any] = []

    if add_services_hint:
        services = dialogue.get("services") or []
        if services:
            ctx.append(assistant_msg(f"[dialogue services] {', '.join(services)}"))

    for turn in dialogue.get("turns", []):
        speaker = turn.get("speaker")
        utt = (turn.get("utterance") or "").strip()
        if not utt:
            continue

        if speaker == "SYSTEM":
            ctx.append(assistant_msg(utt))
            continue

        if speaker == "USER":
            ctx.append(user_msg(utt))
            intents = intents_from_user_turn(turn)
            if not intents:
                continue

            tools = [intents[0]] if label_mode == "first" else intents

            out.append(
                Sample(
                    context=list(ctx),
                    tools=tools,
                    data={
                        "dialogue_id": dialogue.get("dialogue_id"),
                        "services": dialogue.get("services"),
                    },
                    parent_context=None,
                )
            )

    return out


async def build_client(
    *,
    repo_path: str,
    collection_name: str,
    openai_model: str,
    k: int,
    target_size: Optional[int],
) -> ToolSuggestClient:
    repo_file = Path(repo_path)
    repo_file.parent.mkdir(parents=True, exist_ok=True)

    repository = JSONFileRepository(repo_file, collection_name=collection_name)

    formatter = SampleFormatter(max_len=8000)

    embedder = OpenAIEmbedder(model=openai_model)

    suggester = KNNSuggester(formatter=formatter, embedder=embedder, k=k, aggregation="weighted")

    selector = None
    if target_size is not None:
        if GreedySelector is None:
            raise RuntimeError("GreedySelector is not available in your tool-suggest install.")
        selector = GreedySelector(formatter=formatter, embedder=embedder, target_size=target_size)

    return ToolSuggestClient(
        ToolSuggestConfig(
            collection_name=collection_name,
            local_backend=LocalBackendConfig(repository=repository, suggester=suggester, selector=selector),
        )
    )


async def record_split(
    client: ToolSuggestClient,
    split_dir: str,
    *,
    limit_dialogues: Optional[int],
    chunk_size: int,
    max_history_messages: int,
    label_mode: str,
    max_samples: Optional[int],
    add_services_hint: bool,
) -> Counter:
    files = iter_dialogue_files(split_dir)
    logger.info(
    "record split=%s files=%s chunk_size=%s",
    split_dir,
    len(files),
    chunk_size,
    )

    label_freq = Counter()
    chunk: list[Any] = []
    n_dialogs = 0
    n_samples = 0

    file_iter = _tqdm(files, desc=f"record files ({Path(split_dir).name})", unit="file")
    for fp in file_iter:
        dialogs = load_json(fp)
        dlg_iter = _tqdm(dialogs, desc=f"{Path(fp).name}", unit="dlg", leave=False)

        for d in dlg_iter:
            n_dialogs += 1
            samples = make_samples_from_dialogue(
                d,
                max_history_messages=max_history_messages,
                label_mode=label_mode,
                add_services_hint=add_services_hint,
            )

            for s in samples:
                for t in getattr(s, "tools", []):
                    label_freq[t] += 1
                chunk.append(s)
                n_samples += 1

                if len(chunk) >= chunk_size:
                    await client.record_bulk(chunk, wait=True)
                    chunk.clear()

                if max_samples is not None and n_samples >= max_samples:
                    break

            if tqdm is not None:
                dlg_iter.set_postfix_str(f"samples={n_samples} chunk={len(chunk)}")

            if max_samples is not None and n_samples >= max_samples:
                break
            if limit_dialogues is not None and n_dialogs >= limit_dialogues:
                break

        if max_samples is not None and n_samples >= max_samples:
            break
        if limit_dialogues is not None and n_dialogs >= limit_dialogues:
            break

    if chunk:
        await client.record_bulk(chunk, wait=True)

    print(f"[record] DONE dialogs={n_dialogs} samples={n_samples}")
    return label_freq


async def demo(client: ToolSuggestClient, split_dir: str, *, top_k: int, max_history_messages: int, label_mode: str, add_services_hint: bool):
    # pick random dialogue from first N (без загрузки всего split)
    candidates = []
    for i, fp in enumerate(iter_dialogue_files(split_dir)):
        dialogs = load_json(fp)
        candidates.extend(dialogs)
        if len(candidates) > 2000:
            break

    d = random.choice(candidates)
    samples = make_samples_from_dialogue(d, max_history_messages=max_history_messages, label_mode=label_mode, add_services_hint=add_services_hint)
    if not samples:
        print("[demo] no labeled samples in chosen dialogue")
        return
    s = random.choice(samples)

    suggestions = await client.suggest(context=s.context, top_k=top_k)

    print("\n[demo] trained:", client.is_trained)
    print("[demo] GT tools:", s.tools)
    print("[demo] suggestions:")
    for sug in suggestions:
        print(f"  {sug.id:30s} score={sug.score:.4f}")


async def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/sgd")
    ap.add_argument("--repo_path", type=str, default="data/tool_suggest_sgd_intents.json")
    ap.add_argument("--collection", type=str, default="sgd_intents")

    ap.add_argument("--openai_model", type=str, default="text-embedding-3-small")
    ap.add_argument("--k", type=int, default=7)

    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--max_history_messages", type=int, default=12)
    ap.add_argument("--label_mode", choices=["first", "all"], default="first")
    ap.add_argument("--add_services_hint", action="store_true")

    ap.add_argument("--limit_dialogues", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=None)

    ap.add_argument("--reset_repo", action="store_true")
    ap.add_argument("--record", choices=["train", "dev", "test"], default=None)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--demo", action="store_true")

    ap.add_argument("--target_size", type=int, default=None, help="Core-set selection (осторожно: тоже вызовет embeddings)")

    ap.add_argument("--eval", choices=["dev", "test"], default=None)
    ap.add_argument("--eval_dialogues", type=int, default=300)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--env_file", type=str, default=".env", help="Path to .env file with OPENAI_API_KEY and OPENAI_BASE_URL")

    args = ap.parse_args()


    if load_dotenv is None:
        raise ImportError("python-dotenv is required. Install: uv pip install python-dotenv")

    load_dotenv(args.env_file, override=False)

    # небольшая проверка, чтобы не ловить странные ошибки позже
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (check .env / env vars)")
    if not os.getenv("OPENAI_BASE_URL"):
        raise RuntimeError("OPENAI_BASE_URL is not set (check .env / env vars)")

    train_dir = os.path.join(args.data_root, "train")
    dev_dir = os.path.join(args.data_root, "dev")
    test_dir = os.path.join(args.data_root, "test")
    for p in (train_dir, dev_dir, test_dir):
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Split dir not found: {p}")

    if args._repo:
        rp = Path(args.repo_path)
        if rp.exists():
            rp.unlink()
            print(f"[init] removed repo file: {rp}")

    print(f"[init] embedder=openai model={args.openai_model} k={args.k}")

    client = await build_client(
        repo_path=args.repo_path,
        collection_name=args.collection,
        openai_model=args.openai_model,
        k=args.k,
        target_size=args.target_size,
    )

    if args.record is not None:
        split_map = {"train": train_dir, "dev": dev_dir, "test": test_dir}
        freq = await record_split(
            client,
            split_map[args.record],
            limit_dialogues=args.limit_dialogues,
            chunk_size=args.chunk_size,
            max_history_messages=args.max_history_messages,
            label_mode=args.label_mode,
            max_samples=args.max_samples,
            add_services_hint=args.add_services_hint,
        )

        print("\nTop intent-tools in recorded split:")
        for name, cnt in freq.most_common(30):
            print(f"{cnt:8d} {name}")

        toolset = await client.get_toolset()
        print("\nToolset size (from repo):", len(toolset))

    if args.train:
        toolset = await client.get_toolset()
        print(f"[train] starting... toolset_size={len(toolset)}")
        t0 = time.perf_counter()
        await client.train()
        dt = time.perf_counter() - t0
        print(f"[train] done in {dt:.1f}s; trained={client.is_trained}")

    if args.demo:
        await demo(
            client,
            dev_dir,
            top_k=10,
            max_history_messages=args.max_history_messages,
            label_mode=args.label_mode,
            add_services_hint=args.add_services_hint,
        )
    if args.eval is not None:
        split_map = {"dev": dev_dir, "test": test_dir}
        metrics = await eval_topk(
            client,
            split_map[args.eval],
            max_dialogues=args.eval_dialogues,
            top_k=args.top_k,
            max_history_messages=args.max_history_messages,
            label_mode=args.label_mode,
            add_services_hint=args.add_services_hint,
        )
        print(f"\n[eval {args.eval}]", metrics)

async def eval_topk(
    client: ToolSuggestClient,
    split_dir: str,
    *,
    max_dialogues: int,
    top_k: int,
    max_history_messages: int,
    label_mode: str,
    add_services_hint: bool,
) -> dict[str, float]:
    n = 0
    top1 = 0
    topk_hit = 0

    files = iter_dialogue_files(split_dir)
    dlg_seen = 0
    full_correct_dialogs = 0
    file_iter = _tqdm(files, desc=f"eval files ({Path(split_dir).name})", unit="file")
    for fp in file_iter:
        dialogs = load_json(fp)
        dlg_iter = _tqdm(dialogs, desc=f"{Path(fp).name}", unit="dlg", leave=False)

        for d in dlg_iter:
            if dlg_seen >= max_dialogues:
                break
            dlg_seen += 1

            samples = make_samples_from_dialogue(
                d,
                max_history_messages=max_history_messages,
                label_mode=label_mode,
                add_services_hint=add_services_hint,
            )
            all_samples_correct = True
            for s in samples:
                gt = set(s.tools)
                sugg = await client.suggest(context=s.context, top_k=top_k)
                pred = [x.id for x in sugg]

                n += 1
                if pred and pred[0] in gt:
                    top1 += 1
                if any(p in gt for p in pred):
                    topk_hit += 1
                else:
                    all_samples_correct = False
            if all_samples_correct and samples:
                full_correct_dialogs += 1
            if tqdm is not None and n:
                dlg_iter.set_postfix_str(
                    f"dlg={dlg_seen}/{max_dialogues} n={n} top1={top1/n:.3f} top{top_k}={topk_hit/n:.3f}"
                )

        if dlg_seen >= max_dialogues:
            break

    return {
        "n": float(n),
        "top1": (top1 / n if n else 0.0),
        f"top{top_k}": (topk_hit / n if n else 0.0),
        "full_correct_dialogs": (full_correct_dialogs / dlg_seen if dlg_seen else 0.0),
    }

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
