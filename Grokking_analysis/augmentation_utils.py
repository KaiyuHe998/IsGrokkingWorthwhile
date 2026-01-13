import json
import os
import re
import random
from collections import defaultdict, Counter
from statistics import median

import numpy as np


_TOKEN_RE = re.compile(r"<[^>]+>")


def _tokens(text: str):
    return _TOKEN_RE.findall(text or "")


def _tail_from_item(item: dict):
    """
    item: {"input_text": "<e><r>..." , "target_text": "<e><r>...<t></a>"}

    - atomic:    input=[h, r]          tail=t
    - inferred:  input=[h, r1, r2]     tail=t

    Returns: (input_tokens, tail_token_or_None)
    """
    if not isinstance(item, dict):
        return ([], None)

    inp_toks = _tokens(item.get("input_text", ""))
    tgt_toks = _tokens(item.get("target_text", ""))

    # The target must contain at least one more tail token than the input.
    if len(tgt_toks) <= len(inp_toks):
        return (inp_toks, None)

    # By convention, tail = target_tokens[len(input_tokens)].
    tail = tgt_toks[len(inp_toks)]
    return (inp_toks, tail)


def _item_key(item: dict):
    """Key used to match the same sample across different files."""
    if not isinstance(item, dict):
        return None
    return (item.get("input_text", None), item.get("target_text", None))


def _load_json_if_exists(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_hr_to_t_from_atomic_dict(atomic_dict):
    """
    atomic_dict: {h: [(r, t), ...]}

    Returns:
      hr_to_t[(h, r)] = t
        If multiple t exist for the same (h, r), keep the first and count collisions.
      collisions: number of conflicting entries
    """
    hr_to_t = {}
    collisions = 0
    for h, rts in (atomic_dict or {}).items():
        for (r, t) in rts:
            key = (h, r)
            if key in hr_to_t and hr_to_t[key] != t:
                collisions += 1
                continue
            hr_to_t.setdefault(key, t)
    return hr_to_t, collisions


def _build_hr_to_t_from_train_json(train_items):
    """
    Build hr_to_t from atomic samples in train.json (len(input)==2).

    Returns: (hr_to_t, collisions, atomic_count)
    """
    hr_to_t = {}
    collisions = 0
    atomic_count = 0
    for item in (train_items or []):
        inp, tail = _tail_from_item(item)
        if tail is None or len(inp) != 2:
            continue
        h, r = inp
        atomic_count += 1
        key = (h, r)
        if key in hr_to_t and hr_to_t[key] != tail:
            collisions += 1
            continue
        hr_to_t.setdefault(key, tail)
    return hr_to_t, collisions, atomic_count


def _recover_hops_from_inferred_item(item: dict, hr_to_t: dict):
    """
    Recover 2-hop facts from an inferred item:
      input:  <h><r1><r2>
      target: <h><r1><r2><t></a>

    Use hr_to_t[(h, r1)] to obtain bridge b, then verify hr_to_t[(b, r2)] == t.

    Returns: (hop1, hop2) or None
      hop1=(h, r1, b), hop2=(b, r2, t)
    """
    inp_toks, tail = _tail_from_item(item)
    if tail is None or len(inp_toks) != 3:
        return None
    h, r1, r2 = inp_toks

    b = hr_to_t.get((h, r1), None)
    if b is None:
        return None

    t2 = hr_to_t.get((b, r2), None)
    if t2 != tail:
        return None

    return (h, r1, b), (b, r2, tail)


def _select_test_ood_items(dataset_dir: str, train_items, prefer_valid=True):
    """
    Select the source of "test_ood" (compatible with older datasets):
    1) If prefer_valid and valid.json exists: use valid.json
    2) Else if test.json exists and contains type==test_inferred_ood: use those items
    3) Otherwise: return an empty list
    """
    valid_path = os.path.join(dataset_dir, "valid.json")
    test_path = os.path.join(dataset_dir, "test.json")

    if prefer_valid and os.path.exists(valid_path):
        return _load_json_if_exists(valid_path) or []

    test_items = _load_json_if_exists(test_path) or []
    typed_ood = [x for x in test_items if isinstance(x, dict) and x.get("type") == "test_inferred_ood"]
    if typed_ood:
        return typed_ood

    return []


def _derive_testood_fact_universe(test_ood_items, hr_to_t):
    """
    From test_ood inferred samples, derive the hop1/hop2 fact universe.

    Returns: (hop1_set, hop2_set, bad_parse_count)
    """
    hop1 = set()
    hop2 = set()
    bad = 0
    for it in (test_ood_items or []):
        rec = _recover_hops_from_inferred_item(it, hr_to_t)
        if rec is None:
            bad += 1
            continue
        h1, h2 = rec
        hop1.add(h1)
        hop2.add(h2)
    return hop1, hop2, bad


def _form_items(c, t):
    input_text = "".join(c)
    target_text = input_text + "".join([t, "</a>"])
    return {"input_text": input_text, "target_text": target_text}


def augment_train_inferred_with_fact_control(
    train_inferred_facts,
    test_inferred_ood_ds,
    atomic_dict,
    ID_facts,
    OOD_facts,
    hop1_ood_fact_ratio=0.0,
    hop1_samples_per_fact=0,
    hop2_ood_fact_ratio=0.0,
    hop2_samples_per_fact=0,
    avoid_exposing_ood_test_bridges_when_hop2_injection_off=True,
    seed=42,
    verbose=True,
):
    """
    Fact-level augmentation (using atomic fact triples as the unit)

    - Hop1 injection: expose OOD hop1 facts (h, r1, b), then combine with ID (b, r2, t)
      to synthesize inferred samples (h, r1, r2) -> t.
    - Hop2 injection: expose OOD hop2 facts (b, r2, t), then combine with ID (h, r1, b)
      to synthesize inferred samples (h, r1, r2) -> t.

    Note: the "controllable exposure" universe defaults to hop facts that actually appear in
    test_inferred_ood_ds inferred examples.
    """
    random.seed(seed)
    np.random.seed(seed)

    train_inferred_facts = list(train_inferred_facts or [])
    test_inferred_ood_ds = list(test_inferred_ood_ds or [])
    ID_facts = set(ID_facts or [])
    OOD_facts = set(OOD_facts or [])

    hr_to_t, collisions = _build_hr_to_t_from_atomic_dict(atomic_dict)

    # 1) Parse hop1/hop2 OOD atomic-fact universes from test_ood
    test_ood_hop1_facts = set()
    test_ood_hop2_facts = set()
    ood_test_bridges = set()
    bad_parse = 0

    for item in test_inferred_ood_ds:
        rec = _recover_hops_from_inferred_item(item, hr_to_t)
        if rec is None:
            bad_parse += 1
            continue
        hop1, hop2 = rec
        if hop1 in OOD_facts:
            test_ood_hop1_facts.add(hop1)
        if hop2 in OOD_facts:
            test_ood_hop2_facts.add(hop2)
        ood_test_bridges.add(hop1[2])

    # 2) Build ID indices for composing the other hop
    out_id = defaultdict(list)  # b -> [(r2, t)] where (b, r2, t) in ID
    in_id = defaultdict(list)   # b -> [(h, r1)] where (h, r1, b) in ID
    for (h, r, t) in ID_facts:
        out_id[h].append((r, t))
        in_id[t].append((h, r))

    def _sample_subset(s, ratio):
        s = list(s)
        if ratio <= 0 or not s:
            return []
        k = int(round(ratio * len(s)))
        k = max(0, min(k, len(s)))
        return random.sample(s, k) if k > 0 else []

    # 3) Deduplicate: avoid repeating inferred items (including existing train_inferred)
    used = set()
    for it in train_inferred_facts:
        inp, tail = _tail_from_item(it)
        if tail is None or len(inp) != 3:
            continue
        h, r1, r2 = inp
        used.add((h, r1, r2, tail))

    hop2_injection_off = (hop2_ood_fact_ratio <= 0) or (hop2_samples_per_fact <= 0)

    augmented = []
    added_hop1 = 0
    added_hop2 = 0

    # 4) Hop1 injection (expose OOD (h, r1, b))
    if hop1_samples_per_fact > 0 and hop1_ood_fact_ratio > 0:
        chosen = _sample_subset(test_ood_hop1_facts, hop1_ood_fact_ratio)
        for (h, r1, b) in chosen:
            if hop2_injection_off and avoid_exposing_ood_test_bridges_when_hop2_injection_off:
                if b in ood_test_bridges:
                    continue

            cand2 = out_id.get(b, [])
            if not cand2:
                continue
            k = min(hop1_samples_per_fact, len(cand2))
            for (r2, t) in random.sample(cand2, k):
                key = (h, r1, r2, t)
                if key in used:
                    continue
                augmented.append(_form_items([h, r1, r2], t))
                used.add(key)
                added_hop1 += 1

    # 5) Hop2 injection (expose OOD (b, r2, t))
    if hop2_samples_per_fact > 0 and hop2_ood_fact_ratio > 0:
        chosen = _sample_subset(test_ood_hop2_facts, hop2_ood_fact_ratio)
        for (b, r2, t) in chosen:
            inc = in_id.get(b, [])
            if not inc:
                continue
            k = min(hop2_samples_per_fact, len(inc))
            for (h, r1) in random.sample(inc, k):
                key = (h, r1, r2, t)
                if key in used:
                    continue
                augmented.append(_form_items([h, r1, r2], t))
                used.add(key)
                added_hop2 += 1

    if verbose:
        print("\n" + "=" * 90)
        print("FACT-LEVEL AUGMENTATION")
        print("=" * 90)
        print(f"seed={seed}")
        print(f"hr->t collisions (atomic_dict): {collisions}")
        print(
            f"test_ood parsed: hop1_ood_facts={len(test_ood_hop1_facts)}, "
            f"hop2_ood_facts={len(test_ood_hop2_facts)}, bad_parse={bad_parse}"
        )
        print("config:")
        print(f"  hop1_ood_fact_ratio={hop1_ood_fact_ratio}, hop1_samples_per_fact={hop1_samples_per_fact}")
        print(f"  hop2_ood_fact_ratio={hop2_ood_fact_ratio}, hop2_samples_per_fact={hop2_samples_per_fact}")
        print("added:")
        print(f"  +{added_hop1} (uses OOD hop1 fact)")
        print(f"  +{added_hop2} (uses OOD hop2 fact)")
        print(f"total new inferred: {len(augmented)}")
        print("=" * 90 + "\n")

    return train_inferred_facts + augmented


def _select_test_iid_items(dataset_dir: str):
    """
    Select IID test_inferred source (compatible with older datasets):
    - Prefer: test.json items with type==test_inferred_iid
    - If not present: return empty list (cannot reliably distinguish IID)
    """
    test_path = os.path.join(dataset_dir, "test.json")
    test_items = _load_json_if_exists(test_path) or []
    typed_iid = [x for x in test_items if isinstance(x, dict) and x.get("type") == "test_inferred_iid"]
    return typed_iid


def _summarize_exposure_against_universe(
    universe_hop1: set,
    universe_hop2: set,
    train_inferred_items: list,
    hr_to_t: dict,
):
    """
    Given a universe (hop1/hop2 facts), compute exposure and usage stats
    for train_inferred items.
    """
    universe_any = set(universe_hop1) | set(universe_hop2)

    train_hop1_used = set()
    train_hop2_used = set()

    parsed = 0
    parse_errors = 0
    uses_u_hop1 = 0
    uses_u_hop2 = 0
    uses_u_any = 0

    for it in train_inferred_items:
        rec = _recover_hops_from_inferred_item(it, hr_to_t)
        if rec is None:
            parse_errors += 1
            continue
        hop1, hop2 = rec
        parsed += 1
        train_hop1_used.add(hop1)
        train_hop2_used.add(hop2)

        u1 = hop1 in universe_any
        u2 = hop2 in universe_any
        uses_u_hop1 += int(u1)
        uses_u_hop2 += int(u2)
        uses_u_any = int(u1 or u2)
        uses_u_any += uses_u_any

    hop1_exposed = len(universe_hop1 & train_hop1_used)
    hop2_exposed = len(universe_hop2 & train_hop2_used)

    return {
        "universe_hop1": len(universe_hop1),
        "universe_hop2": len(universe_hop2),
        "universe_any": len(universe_any),
        "hop1_exposed": hop1_exposed,
        "hop2_exposed": hop2_exposed,
        "train_inferred_parsed": parsed,
        "train_inferred_parse_errors": parse_errors,
        "uses_hop1": uses_u_hop1,
        "uses_hop2": uses_u_hop2,
        "uses_any": uses_u_any,
    }


def enhanced_fact_exposure_analysis(dataset_dir: str, prefer_valid_as_test_ood: bool = True):
    """
    Fact exposure analysis compatible with both new and old datasets.

    - OOD definition: derive universe from test_ood (prefer valid.json; otherwise test.json[type==test_inferred_ood])
    - ID definition: derive universe from test.json[type==test_inferred_iid] (if missing, report N/A)
    """
    train_path = os.path.join(dataset_dir, "train.json")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")

    train = _load_json_if_exists(train_path) or []
    hr_to_t, collisions, atomic_count = _build_hr_to_t_from_train_json(train)

    # Extract train_inferred items
    train_inferred = []
    for item in train:
        inp, tail = _tail_from_item(item)
        if tail is None:
            continue
        if len(inp) == 3:
            train_inferred.append(item)

    # ---------- OOD universe (from valid or test[type==ood]) ----------
    test_ood_items = _select_test_ood_items(dataset_dir, train, prefer_valid=prefer_valid_as_test_ood)
    test_ood_hop1, test_ood_hop2, bad_testood = _derive_testood_fact_universe(test_ood_items, hr_to_t)

    ood_stats = _summarize_exposure_against_universe(
        universe_hop1=test_ood_hop1,
        universe_hop2=test_ood_hop2,
        train_inferred_items=train_inferred,
        hr_to_t=hr_to_t,
    )

    valid_exists = os.path.exists(os.path.join(dataset_dir, "valid.json"))
    ood_source = "valid.json" if (prefer_valid_as_test_ood and valid_exists) else "test.json[type==test_inferred_ood] or empty"

    # ---------- ID (IID) universe (from test[type==iid]) ----------
    test_iid_items = _select_test_iid_items(dataset_dir)
    test_iid_hop1, test_iid_hop2, bad_iid = _derive_testood_fact_universe(test_iid_items, hr_to_t)

    iid_stats = _summarize_exposure_against_universe(
        universe_hop1=test_iid_hop1,
        universe_hop2=test_iid_hop2,
        train_inferred_items=train_inferred,
        hr_to_t=hr_to_t,
    )

    # ---------- Print ----------
    print("\n" + "=" * 100)
    print("FACT EXPOSURE ANALYSIS (COMPAT + OOD + ID)")
    print("=" * 100)
    print(f"dataset_dir: {dataset_dir}")
    print(f"train atomics: {atomic_count}, hr->t collisions: {collisions}")
    print(f"train_inferred items: {len(train_inferred)}")
    print("")

    # OOD section
    print("[OOD / test_ood-derived universe]")
    print(f"  test_ood source: {ood_source}")
    print(f"  test_ood items: {len(test_ood_items)}, inferred-parse errors (universe build): {bad_testood}")
    print(f"  universe hop1/hop2/any: {ood_stats['universe_hop1']}/{ood_stats['universe_hop2']}/{ood_stats['universe_any']}")
    print(
        f"  exposed hop1: {ood_stats['hop1_exposed']}/{ood_stats['universe_hop1']}"
        f" ({(ood_stats['hop1_exposed']/ood_stats['universe_hop1']*100 if ood_stats['universe_hop1'] else 0):.2f}%)"
    )
    print(
        f"  exposed hop2: {ood_stats['hop2_exposed']}/{ood_stats['universe_hop2']}"
        f" ({(ood_stats['hop2_exposed']/ood_stats['universe_hop2']*100 if ood_stats['universe_hop2'] else 0):.2f}%)"
    )
    if ood_stats["train_inferred_parsed"] > 0:
        n = ood_stats["train_inferred_parsed"]
        print(f"  train_inferred uses OOD-universe via hop1: {ood_stats['uses_hop1']}/{n} ({ood_stats['uses_hop1']/n*100:.2f}%)")
        print(f"  train_inferred uses OOD-universe via hop2: {ood_stats['uses_hop2']}/{n} ({ood_stats['uses_hop2']/n*100:.2f}%)")
        print(f"  train_inferred uses OOD-universe any hop:  {ood_stats['uses_any']}/{n} ({ood_stats['uses_any']/n*100:.2f}%)")
    else:
        print("  train_inferred usage: N/A (no inferred parsed)")
    print("")

    # ID section
    print("[ID / test_inferred_iid-derived universe]")
    if len(test_iid_items) == 0:
        print("  source: test.json[type==test_inferred_iid] NOT FOUND -> N/A for old datasets without type")
    else:
        print("  source: test.json[type==test_inferred_iid]")
    print(f"  test_iid items: {len(test_iid_items)}, inferred-parse errors (universe build): {bad_iid}")
    print(f"  universe hop1/hop2/any: {iid_stats['universe_hop1']}/{iid_stats['universe_hop2']}/{iid_stats['universe_any']}")
    print(
        f"  exposed hop1: {iid_stats['hop1_exposed']}/{iid_stats['universe_hop1']}"
        f" ({(iid_stats['hop1_exposed']/iid_stats['universe_hop1']*100 if iid_stats['universe_hop1'] else 0):.2f}%)"
    )
    print(
        f"  exposed hop2: {iid_stats['hop2_exposed']}/{iid_stats['universe_hop2']}"
        f" ({(iid_stats['hop2_exposed']/iid_stats['universe_hop2']*100 if iid_stats['universe_hop2'] else 0):.2f}%)"
    )
    if iid_stats["train_inferred_parsed"] > 0 and iid_stats["universe_any"] > 0:
        n = iid_stats["train_inferred_parsed"]
        print(f"  train_inferred uses ID-universe via hop1: {iid_stats['uses_hop1']}/{n} ({iid_stats['uses_hop1']/n*100:.2f}%)")
        print(f"  train_inferred uses ID-universe via hop2: {iid_stats['uses_hop2']}/{n} ({iid_stats['uses_hop2']/n*100:.2f}%)")
        print(f"  train_inferred uses ID-universe any hop:  {iid_stats['uses_any']}/{n} ({iid_stats['uses_any']/n*100:.2f}%)")
    else:
        print("  train_inferred usage: N/A (no IID universe or no inferred parsed)")

    print("=" * 100 + "\n")


def annotate_test_json_fact_categories(
    dataset_dir: str,
    output_filename: str = "test_annotated.json",
    prefer_valid_as_test_ood: bool = True,
):
    """
    Annotate test.json with fact categories (compatible with older datasets that don't have id_facts/ood_facts).

    The output (written to output_filename) includes:
      - kind: atomic / inferred / unknown
      - atomic_fact: [h, r, t] (atomic only)
      - hop1_fact / hop2_fact (inferred only, if parseable)
      - uses_testood_fact_hop1/hop2/any: whether it uses the "test_ood fact universe"
      - Keeps original fields (e.g., the original 'type')
    """
    train_path = os.path.join(dataset_dir, "train.json")
    test_path = os.path.join(dataset_dir, "test.json")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    train = _load_json_if_exists(train_path) or []
    test_items = _load_json_if_exists(test_path) or []

    hr_to_t, _, _ = _build_hr_to_t_from_train_json(train)

    test_ood_items = _select_test_ood_items(dataset_dir, train, prefer_valid=prefer_valid_as_test_ood)
    test_ood_hop1, test_ood_hop2, _ = _derive_testood_fact_universe(test_ood_items, hr_to_t)
    test_ood_any = set(test_ood_hop1) | set(test_ood_hop2)

    annotated = []
    inferred_parse_errors = 0

    for it in test_items:
        if not isinstance(it, dict):
            annotated.append(it)
            continue

        inp, tail = _tail_from_item(it)
        out = dict(it)  # copy

        if tail is None:
            out["kind"] = "unknown"
            annotated.append(out)
            continue

        if len(inp) == 2:
            out["kind"] = "atomic"
            out["atomic_fact"] = [inp[0], inp[1], tail]
            out["is_testood_fact"] = (tuple(out["atomic_fact"]) in test_ood_any)
            annotated.append(out)
            continue

        if len(inp) == 3:
            out["kind"] = "inferred"
            rec = _recover_hops_from_inferred_item(it, hr_to_t)
            if rec is None:
                inferred_parse_errors += 1
                out["hop_parse_ok"] = False
                annotated.append(out)
                continue

            hop1, hop2 = rec
            out["hop_parse_ok"] = True
            out["hop1_fact"] = [hop1[0], hop1[1], hop1[2]]
            out["hop2_fact"] = [hop2[0], hop2[1], hop2[2]]
            out["uses_testood_fact_hop1"] = (hop1 in test_ood_any)
            out["uses_testood_fact_hop2"] = (hop2 in test_ood_any)
            out["uses_testood_fact_any"] = out["uses_testood_fact_hop1"] or out["uses_testood_fact_hop2"]
            annotated.append(out)
            continue

        out["kind"] = "unknown"
        annotated.append(out)

    out_path = os.path.join(dataset_dir, output_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f)

    print(f"Wrote: {out_path} (items={len(annotated)}, inferred_parse_errors={inferred_parse_errors})")


def _train_atomic_fact_counter(train_items):
    """
    Count occurrences of atomic triples in train.json (len(input)==2).
    Typically each atomic fact appears once, but we keep Counter for generality.
    """
    c = Counter()
    for it in train_items or []:
        inp, tail = _tail_from_item(it)
        if tail is None or len(inp) != 2:
            continue
        h, r = inp
        c[(h, r, tail)] += 1
    return c


def _percentiles(vals, ps=(0, 25, 50, 75, 90, 95, 99, 100)):
    if not vals:
        return {p: 0 for p in ps}
    arr = np.array(vals, dtype=np.int64)
    out = {}
    for p in ps:
        out[p] = int(np.percentile(arr, p))
    return out


def iid_fact_count_report(
    dataset_dir: str,
    split: str = "iid",
    prefer_valid_as_test_ood: bool = True,
    topk_missing: int = 20,
):
    """
    "Direct counting" report (no k needed):

    - split="iid": extract hop facts from test.json[type==test_inferred_iid]
    - split="ood": extract hop facts from valid.json (preferred) or test.json[type==test_inferred_ood]

    For extracted hop1/hop2 atomic facts, compute:
      A) How many times they appear as atomic facts in train (usually 0/1)
      B) How many times they are used as components inside train_inferred (Counter; often >> 1)
    """
    train_path = os.path.join(dataset_dir, "train.json")
    test_path = os.path.join(dataset_dir, "test.json")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    train = _load_json_if_exists(train_path) or []
    hr_to_t, collisions, atomic_count = _build_hr_to_t_from_train_json(train)

    # Train split
    train_inferred = []
    for it in train:
        inp, tail = _tail_from_item(it)
        if tail is None:
            continue
        if len(inp) == 3:
            train_inferred.append(it)

    train_atomic_counter = _train_atomic_fact_counter(train)
    train_hop1_counter, train_hop2_counter, bad_train = _count_hop_facts_in_inferred(train_inferred, hr_to_t)

    # Pick eval inferred items
    if split == "iid":
        eval_items = _select_test_iid_items(dataset_dir)
        source = "test.json[type==test_inferred_iid]"
    elif split == "ood":
        eval_items = _select_test_ood_items(dataset_dir, train, prefer_valid=prefer_valid_as_test_ood)
        valid_exists = os.path.exists(os.path.join(dataset_dir, "valid.json"))
        source = "valid.json" if (prefer_valid_as_test_ood and valid_exists) else "test.json[type==test_inferred_ood]"
    else:
        raise ValueError("split must be 'iid' or 'ood'")

    hop1_univ, hop2_univ, bad_eval = _derive_testood_fact_universe(eval_items, hr_to_t)

    # Counts for those universes
    hop1_as_inferred = [train_hop1_counter.get(f, 0) for f in hop1_univ]
    hop2_as_inferred = [train_hop2_counter.get(f, 0) for f in hop2_univ]

    hop1_as_atomic = [train_atomic_counter.get(f, 0) for f in hop1_univ]
    hop2_as_atomic = [train_atomic_counter.get(f, 0) for f in hop2_univ]

    # Missing lists (inferred-component perspective)
    hop1_missing = [f for f in hop1_univ if train_hop1_counter.get(f, 0) == 0]
    hop2_missing = [f for f in hop2_univ if train_hop2_counter.get(f, 0) == 0]

    def _summ(name, vals):
        if not vals:
            return f"{name}: N/A"
        ps = _percentiles(vals)
        return (
            f"{name}: mean={float(np.mean(vals)):.2f}, "
            f"p0={ps[0]}, p25={ps[25]}, p50={ps[50]}, p75={ps[75]}, "
            f"p90={ps[90]}, p95={ps[95]}, p99={ps[99]}, p100={ps[100]}"
        )

    print("\n" + "=" * 110)
    print("IID/OOD HOP FACT COUNT REPORT (NO k)")
    print("=" * 110)
    print(f"dataset_dir: {dataset_dir}")
    print(f"mode(split): {split}")
    print(f"eval source: {source}")
    print(f"train atomics: {atomic_count}, hr->t collisions: {collisions}")
    print(f"train_inferred items: {len(train_inferred)} (parse_errors={bad_train})")
    print(f"eval inferred items: {len(eval_items)} (parse_errors={bad_eval})")
    print("")
    print("[Universe sizes derived from eval inferred]")
    print(f"  hop1 facts: {len(hop1_univ)}")
    print(f"  hop2 facts: {len(hop2_univ)}")
    print("")
    print("[Counts in TRAIN as INFERRED components]  (what matters for inferred-augmentation exposure)")
    print("  " + _summ("hop1 count", hop1_as_inferred))
    print("  " + _summ("hop2 count", hop2_as_inferred))
    print(f"  hop1 missing (count==0): {len(hop1_missing)}/{len(hop1_univ)} ({(len(hop1_missing)/len(hop1_univ)*100 if hop1_univ else 0):.2f}%)")
    print(f"  hop2 missing (count==0): {len(hop2_missing)}/{len(hop2_univ)} ({(len(hop2_missing)/len(hop2_univ)*100 if hop2_univ else 0):.2f}%)")
    print("")
    print("[Counts in TRAIN as ATOMIC facts]  (usually 0/1; mainly sanity-check)")
    print("  " + _summ("hop1 atomic count", hop1_as_atomic))
    print("  " + _summ("hop2 atomic count", hop2_as_atomic))

    if topk_missing > 0 and (hop1_missing or hop2_missing):
        print("")
        if hop1_missing:
            print(f"[Examples] hop1 missing (showing up to {topk_missing})")
            for f in hop1_missing[:topk_missing]:
                print("  ", f)
        if hop2_missing:
            print(f"[Examples] hop2 missing (showing up to {topk_missing})")
            for f in hop2_missing[:topk_missing]:
                print("  ", f)

    print("=" * 110 + "\n")


def _count_hop_facts_in_inferred(inferred_items, hr_to_t):
    """
    Return hop1/hop2 component occurrence counts in train_inferred:
      hop1_counter[(h, r1, b)] = count
      hop2_counter[(b, r2, t)] = count
    """
    hop1_c = Counter()
    hop2_c = Counter()
    bad = 0
    for it in inferred_items or []:
        rec = _recover_hops_from_inferred_item(it, hr_to_t)
        if rec is None:
            bad += 1
            continue
        hop1, hop2 = rec
        hop1_c[hop1] += 1
        hop2_c[hop2] += 1
    return hop1_c, hop2_c, bad