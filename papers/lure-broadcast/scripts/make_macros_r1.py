"""Macros for the IEEE Access R1 revision's new measurements (plan WP3-WP9).

Kept apart from make_macros.py, whose single main() already carries 634 macros, and called from it,
so every number in the manuscript still originates in one build step (golden rule 1). The
conventions are the same:
  * counts below ten are spelled out by word();
  * precision is fixed here and never in the prose;
  * a quantity that appears in more than one artefact is cross-checked, and disagreement RAISES.

Sources (all under papers/lure-broadcast/results/):
  rule_interaction.json, frame_probe_<chain>.json, tail_sample_<chain>.json,
  tail_volume_<chain>.json, metadata_history_<chain>.json, negative_controls_<chain>.json,
  recipient_activity_robust_<chain>.json, bound_worked_example.json, forgery_chain_<chain>.json
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

FABRICATED = ("constant_nonzero", "address_derived")


def _count_or_word(m, name: str, v: int, comment: str = "") -> None:
    """A count as a word below ten and as a numeral otherwise, following the house style."""
    (m.word if 0 <= int(v) < 10 else m.int_)(name, int(v), comment)


def emit(m, load: Callable[[str], dict], chains: list[str], chain_tex: dict, paper: Path) -> None:
    # ---------------------------------------------------------------- C1 / WP9.3: the worked bound
    m.section("R1 · exact disagreement bound, worked calculation (bound_worked_example.json)")
    bw = load("bound_worked_example.json")
    for variant, tag in (("ignorability", "Ign"), ("worst_case", "Worst")):
        w = bw[variant]
        _count_or_word(m, f"Bound{tag}Kmax", w["k_max"])
        m.dec(f"Bound{tag}Tail", w["tail_at_allocation"], 4, "tail at an attaining allocation (> alpha)")
        m.dec(f"Bound{tag}NextTail", w["max_tail_at_k_max_plus_1"], 4, "largest tail at k_max+1 (<= alpha)")
    m.int_("BoundFlaggedPop", bw["flagged_population"])
    for variant, tag in (("ignorability", "Ign"), ("worst_case", "Worst")):
        m.int_(f"Bound{tag}Enum", bw[variant]["allocations_enumerated"], "allocations in B(x), enumerated")
        m.int_(f"Bound{tag}KmaxTab", bw[variant]["k_max"], "numeral for table cells")
    # Results and the appendix both say the ignorability maximum sits wholly in the most lightly sampled
    # flagged stratum.
    alloc = bw["ignorability"]["allocation"]
    frac = [mm / M for M, mm, _ in bw["ignorability"]["inputs_M_m_x"]]
    holders = [h for h, k in enumerate(alloc) if k]
    if not (len(holders) == 1 and frac[holders[0]] == min(frac)):
        raise SystemExit("the ignorability allocation is no longer wholly in the most lightly sampled "
                         "stratum; revise results.tex and model.tex")
    # The appendix says the worst case differs from ignorability only by counting each unlabelled flagged
    # contract as a sampled, observed disagreement.
    ign_in, worst_in = bw["ignorability"]["inputs_M_m_x"], bw["worst_case"]["inputs_M_m_x"]
    added_m = sum(w[1] - i[1] for i, w in zip(ign_in, worst_in))
    added_x = sum(w[2] - i[2] for i, w in zip(ign_in, worst_in))
    excluded = load("detector_design_based_base.json")["estimates"]["precision"][
        "false_positive_upper95_worst_case_exclusions"]["excluded_flagged_treated_as_false_positive"]
    if not added_m == added_x == excluded:
        raise SystemExit(f"worst case adds {added_m} sampled rows and {added_x} disagreements, but "
                         f"{excluded} flagged contracts were excluded; the appendix text would be wrong")

    # ---------------------------------------------------------------- WP2.3: v2 annotation study
    m.section("R1 · v2 annotation study, registered analyses (annotation_v2/evaluation.json)")
    ev2 = load("annotation_v2/evaluation.json")
    key2 = load("annotation_v2/answer_key.json")
    tier2 = ev2["tier"]
    if tier2["tier"] != "A" or any(r["author"] for r in tier2["raters"]) or tier2["adjudicator"]["author"]:
        raise SystemExit("the v2 evaluation is not a tier A run with non-author raters and adjudicator")
    m.int_("VTwoRows", key2["sample_size"])
    _count_or_word(m, "VTwoChains", len(key2["chains"]))
    for f, tag in (("destination", "Dest"), ("call_to_action", "Cta"), ("conj", "Conj"), ("purpose", "Purpose")):
        k = ev2["kappa"][f]
        m.dec(f"VTwoKappa{tag}", k["kappa"], 2)
        m.dec(f"VTwoKappa{tag}Lo", k["ci95"][0], 2)
        m.dec(f"VTwoKappa{tag}Hi", k["ci95"][1], 2)
    flagged_pop = None
    for name, tag in (("P1_conjunction", "Conj"), ("P2_purpose", "Purpose")):
        p = ev2["primary"][name]["pooled"]
        conf, est, bd = p["sampled_confusion"], p["estimates"], p["upper_bound_disagreeing_flagged"]
        m.int_(f"VTwo{tag}Agreed", p["agreed_rows"])
        for cell in ("TP", "FP", "FN", "TN"):
            m.int_(f"VTwo{tag}{cell}", conf[cell], "sampled agreed rows, pooled")
        _count_or_word(m, f"VTwo{tag}ExclFlagged", p["excluded_flagged"])
        m.dec(f"VTwo{tag}Precision", est["precision"]["point"], 4)
        m.dec(f"VTwo{tag}Recall", est["recall"]["point"], 4)
        m.dec(f"VTwo{tag}RecallLo", est["recall"]["ci95"][0], 4)
        m.dec(f"VTwo{tag}RecallHi", est["recall"]["ci95"][1], 4)
        for variant, vt in (("ignorability", "Ign"), ("worst_case", "Worst")):
            m.int_(f"VTwo{tag}Bound{vt}K", bd[variant]["k_max"])
            m.pct(f"VTwo{tag}Bound{vt}Pct", bd[variant]["theta"], 2)
        flagged_pop = bd["flagged_population"]
        # The text says no agreed flagged row disagreed, on any chain. Checked, not assumed.
        if conf["FP"] or any(c["sampled_confusion"]["FP"] for c in ev2["primary"][name]["per_chain"].values()):
            raise SystemExit(f"{name}: an agreed flagged row disagrees; the v2 text says none does")
        adj = ev2["adjudicated_reported_separately"][name]
        m.int_(f"VTwoAdj{tag}FN", adj["sampled_confusion"]["FN"])
        m.int_(f"VTwoAdj{tag}FP", adj["sampled_confusion"]["FP"])
        m.int_(f"VTwoAdj{tag}Agreed", adj["agreed_rows"])
        for variant, vt in (("ignorability", "Ign"), ("worst_case", "Worst")):
            m.int_(f"VTwoAdj{tag}Bound{vt}K", adj["upper_bound_disagreeing_flagged"][variant]["k_max"])
            m.pct(f"VTwoAdj{tag}Bound{vt}Pct", adj["upper_bound_disagreeing_flagged"][variant]["theta"], 2)
    m.int_("VTwoFlaggedPop", flagged_pop)
    # 872 is the five chains' flagged strata, Base restricted to the rows v1 did not sample.
    strong = {c: sum(v for s, v in info["population_by_stratum"].items() if s.startswith("strong"))
              for c, info in key2["chains"].items()}
    if sum(strong.values()) != flagged_pop:
        raise SystemExit(f"flagged strata sum to {sum(strong.values())}, the bound says {flagged_pop}")
    m.int_("VTwoBaseFlaggedOutOfSample", strong["base"], "Base flagged contracts v1 did not sample")
    # The text explains the wide worst case by Optimism: most excluded flagged rows, lightest sampling.
    sampled = {c: sum(v for s, v in info["sampled_by_stratum"].items() if s.startswith("strong"))
               for c, info in key2["chains"].items()}
    excl = {c: pc["excluded_flagged"] for c, pc in ev2["primary"]["P1_conjunction"]["per_chain"].items()}
    ratio = {c: strong[c] / sampled[c] for c in strong if sampled[c]}
    if not (max(ratio, key=ratio.get) == "optimism" and max(excl, key=excl.get) == "optimism"):
        raise SystemExit(f"Optimism no longer has both the lightest flagged sampling {ratio} and the most "
                         f"excluded flagged rows {excl}; revise the v2 bound paragraph")
    m.int_("VTwoOptFlaggedPop", strong["optimism"])
    m.int_("VTwoOptFlaggedSampled", sampled["optimism"])
    _count_or_word(m, "VTwoOptExclFlagged", excl["optimism"])

    # ---------------------------------------------------------------- WP2.3: deviation 4 and exploratory
    m.section("R1 · v2 robustness and author comparison (annotation_v2/robustness.json, deviation 4)")
    rob = load("annotation_v2/robustness.json")
    nonauthors, authors = rob["nonauthor_raters"], rob["author_raters"]
    # Table rows are named by role, so the roles must be the ones tier.json records.
    if nonauthors[:2] != [r["id"] for r in tier2["raters"]] or nonauthors[2:] != [tier2["adjudicator"]["id"]] \
            or len(authors) != 1:
        raise SystemExit(f"robustness roles {nonauthors} / {authors} do not match the registered tier record")
    pair_keys = [f"{a}-{b}" for i, a in enumerate(nonauthors) for b in nonauthors[i + 1:]]
    if sorted(pair_keys) != sorted(rob["R1_pairs"]):
        raise SystemExit(f"robustness pairs {sorted(rob['R1_pairs'])} are not the three non-author pairs")
    # The registered pair's row reuses tab:vtwo's macros, so the two computations must agree exactly.
    reg = rob["R1_pairs"][pair_keys[0]]
    if reg["kappa"] != ev2["kappa"] or any(reg[n] != ev2["primary"][n]["pooled"]
                                           for n in ("P1_conjunction", "P2_purpose")):
        raise SystemExit("robustness.json's registered pair differs from evaluation.json")
    rows_r = {"PairTwo": rob["R1_pairs"][pair_keys[1]], "PairThree": rob["R1_pairs"][pair_keys[2]],
              "Maj": rob["R3_majority"]}
    for tag, res in rows_r.items():
        if "kappa" in res:
            m.dec(f"VTwo{tag}KappaConj", res["kappa"]["conj"]["kappa"], 2)
        for name, lab in (("P1_conjunction", "Conj"), ("P2_purpose", "Purpose")):
            p = res[name]
            m.dec(f"VTwo{tag}{lab}Recall", p["estimates"]["recall"]["point"], 4)
        wc = res["P1_conjunction"]["upper_bound_disagreeing_flagged"]["worst_case"]
        m.int_(f"VTwo{tag}ConjBoundWorstK", wc["k_max"])
        m.pct(f"VTwo{tag}ConjBoundWorstPct", wc["theta"], 2)
    # The text says no conclusion depends on the pair: no flagged disagreement for any pair or the majority.
    every = [reg, *rows_r.values()]
    if any(r[n]["sampled_confusion"]["FP"] for r in every for n in ("P1_conjunction", "P2_purpose")):
        raise SystemExit("a non-author pair or the majority has a flagged disagreement; revise sec:res-v2")
    kconj = [rob["R1_pairs"][k]["kappa"]["conj"]["kappa"] for k in pair_keys]
    m.dec("VTwoPairKappaConjMin", min(kconj), 2)
    m.dec("VTwoPairKappaConjMax", max(kconj), 2)
    # The text says the tie-breaker's pairs agree least, which is why its adjudication moves FN.
    if kconj[0] != max(kconj):
        raise SystemExit("the registered pair no longer agrees best; revise the robustness paragraph")
    # The text attributes the widest worst case to the pair with the most flagged rows left unagreed.
    excl_f = {k: rob["R1_pairs"][k]["P1_conjunction"]["excluded_flagged"] for k in pair_keys}
    worst = {k: rob["R1_pairs"][k]["P1_conjunction"]["upper_bound_disagreeing_flagged"]["worst_case"]["k_max"]
             for k in pair_keys}
    if max(worst, key=worst.get) != max(excl_f, key=excl_f.get):
        raise SystemExit(f"widest worst case {worst} is not the pair with most excluded flagged rows {excl_f}")
    _count_or_word(m, "VTwoPairThreeConjExclFlagged", excl_f[pair_keys[2]])
    for f, tag in (("conj", "Conj"), ("purpose", "Purpose")):
        m.dec(f"VTwoFleiss{tag}", rob["R2_fleiss"][f]["kappa"], 2)
    maj = rob["R3_majority"]
    m.int_("VTwoMajConjRows", maj["rows_with_majority"]["conj"])
    _count_or_word(m, "VTwoMajConjExclFlagged", maj["P1_conjunction"]["excluded_flagged"])
    # The text says the majority's worst case is narrower than any pair's.
    if maj["P1_conjunction"]["upper_bound_disagreeing_flagged"]["worst_case"]["k_max"] >= min(worst.values()):
        raise SystemExit(f"the majority's worst case is not narrower than every pair's {worst}")
    # The text says it equals the adjudicated worst case (the two label sets differ only on rows the
    # pair left unresolved, so this can change with the data).
    if maj["P1_conjunction"]["upper_bound_disagreeing_flagged"]["worst_case"] != \
            ev2["adjudicated_reported_separately"]["P1_conjunction"]["upper_bound_disagreeing_flagged"]["worst_case"]:
        raise SystemExit("the majority's worst case no longer equals the adjudicated one; revise sec:res-v2")

    au = authors[0]
    a1, a2 = rob["A1_author_kappa"][au], rob["A2_author_vs_majority"][au]
    akc = [a1[x]["conj"]["kappa"] for x in nonauthors]
    m.dec("VTwoAuthorKappaConjMin", min(akc), 2)
    m.dec("VTwoAuthorKappaConjMax", max(akc), 2)
    m.dec("VTwoAuthorKappaConjMaj", a1["majority"]["conj"]["kappa"], 2)
    m.dec("VTwoAuthorKappaPurposeMaj", a1["majority"]["purpose"]["kappa"], 2)
    m.int_("VTwoAuthorMajRows", a2["rows_with_majority_conjunction"])
    m.int_("VTwoAuthorMajMatch", a2["author_matches_majority"])
    # The text says the detector matches the majority on the same number of rows as the author.
    if a2["detector_matches_majority"] != a2["author_matches_majority"]:
        raise SystemExit("A2: the author and the detector no longer match the majority equally often")
    # The text says the author agrees with each primary rater more closely than they agree with each other.
    if not all(a1[x]["conj"]["kappa"] > kconj[0] for x in nonauthors[:2]):
        raise SystemExit("the author no longer agrees with each primary rater above the primary pair's kappa")
    # The text says the author is one of the first study's two raters.
    v1_files = load("interrater_base.json")["rater_files"]
    if not any(f.removesuffix(".csv").endswith("_" + au.lower()) for f in v1_files):
        raise SystemExit(f"{au} is not among the first study's rater files {v1_files}; revise sec:res-v2")

    # Sensitivity says the non-author raters disagree with no agreed flagged contract "either", i.e. the
    # first study had none on its agreed set too.
    if load("detector_design_based_base.json")["confusion_matrix_sampled_rows"]["FP"]:
        raise SystemExit("the first study has an agreed flagged disagreement; revise sensitivity.tex")

    m.section("R1 · exploratory, not registered (annotation_v2/exploratory.json)")
    exp = load("annotation_v2/exploratory.json")
    if not exp["status"].startswith("exploratory"):
        raise SystemExit("exploratory.json is not labelled exploratory")
    e1 = exp["E1_detector_agreement"]
    m.int_("VTwoAuthorDetAgree", e1[au]["agrees_with_detector"])
    m.int_("VTwoAuthorDetResolved", e1[au]["resolved"])
    na_share = [e1[x]["share"] for x in nonauthors]
    m.pct("VTwoNonAuthorDetPctMin", min(na_share), 1)
    m.pct("VTwoNonAuthorDetPctMax", max(na_share), 1)
    m.pct("VTwoAuthorDetPct", e1[au]["share"], 1)
    # The text says the author agrees with the detector more often than any non-author.
    if e1[au]["share"] <= max(na_share):
        raise SystemExit("the author no longer agrees with the detector more often than every non-author")
    e2 = exp["E2_mismatch_overlap"][au]
    if e2["author_mismatches"] != a2["rows_with_majority_conjunction"] - a2["author_matches_majority"]:
        raise SystemExit("exploratory.json and robustness.json disagree on the author's mismatches")

    # ---------------------------------------------------------------- C2 / WP1.2: recall as drawn
    m.section("R1 · detector as frozen when the sample was drawn (detector_eval_base.json)")
    drawn = load("detector_eval_base.json")
    if drawn["detector_source"] != "frozen in the answer key at sample-draw time":
        raise SystemExit("detector_eval_base.json was not computed from the verdicts frozen at draw time; "
                         "it cannot stand as the pre-revision recall")
    est = drawn["estimates_population_weighted"]
    m.dec("DetRecallAsDrawn", est["recall"], 4, "recall before the post-sample detector revision")
    m.dec("DetFoneAsDrawn", est["f1"], 4)
    weak = drawn["miss_location"]["by_detector_band"]
    _count_or_word(m, "DetAsDrawnWeakLures", weak["weak"]["human_lure"], "labelled lures in the frozen weak band")
    _count_or_word(m, "DetAsDrawnWeakLabelled", weak["weak"]["labelled"])
    _count_or_word(m, "DetAsDrawnNoneLures", weak["none"]["human_lure"], "labelled lures in the frozen none band")
    # Results says the revision moved every reclassified sampled contract from weak to strong.
    moved = load("detector_design_based_base.json")["reclassified_since_draw"]["rows"]
    if not all(r["frozen"] == "weak" and r["now"] == "strong" for r in moved):
        raise SystemExit("a reclassified contract did not move from weak to strong; revise results.tex")

    # ---------------------------------------------------------------- WP11.1: replication record
    m.section("R1 · replication record (replication.json)")
    rep = load("replication.json")
    for c in chains:
        tex = chain_tex[c][0]
        r = rep["chains"][c]
        m.int_(f"ReplLastBlock{tex}", r["last_block"], "last block of the half-open corpus range")
        m.int_(f"ReplProbeBlock{tex}", r["end_block_exclusive"], "first block after the corpus")
        m.int_(f"ReplLastLogged{tex}", r["last_block_with_transfer_log"])
        m.raw(f"ReplLastLoggedDate{tex}", r["last_block_with_transfer_log_utc"][:10])
        m.raw(f"ReplTipDate{tex}", r["tip_fixed_at_utc"][:10])
        m.raw(f"ReplHosts{tex}", ", ".join(rf"\url{{{h}}}" for h in r["metadata_hosts"]))
        # The text says every state probe reads the first block after the corpus. Check each artefact.
        pins = {f"frame_probe_{c}.json": load(f"frame_probe_{c}.json")["block"],
                f"metadata_history_{c}.json": load(f"metadata_history_{c}.json")["corpus_end_block"]}
        if c == "base":
            pins["forgery_probe_base.json"] = load("forgery_probe_base.json")["block"]
        wrong = {f: b for f, b in pins.items() if b != r["end_block_exclusive"]}
        if wrong:
            raise SystemExit(f"{c}: probes pinned elsewhere than block {r['end_block_exclusive']}: {wrong}")
    for lib in ("hypersync", "numpy", "scipy"):
        m.raw(f"Repl{lib.capitalize()}", rep["versions"][lib])
    # The caption says only Polygon zkEVM's last transfer log precedes the pull, by about a month.
    from datetime import date
    gaps = {c: (date.fromisoformat(rep["chains"][c]["tip_fixed_at_utc"][:10])
                - date.fromisoformat(rep["chains"][c]["last_block_with_transfer_log_utc"][:10])).days
            for c in chains}
    if not (all(g <= 1 for c, g in gaps.items() if c != "polygon-zkevm") and 25 <= gaps["polygon-zkevm"] <= 40):
        raise SystemExit(f"last-logged gaps {gaps} no longer match the replication caption")

    # ---------------------------------------------------------------- WP5: rule paths, allowlist, scripts
    m.section("R1 · how the rule behaves on the frames (rule_interaction.json)")
    ri = load("rule_interaction.json")
    tot_strong = tot_short = tot_nonlatin = tot_look = tot_added = 0
    for c in chains:
        tex = chain_tex[c][0]
        r = ri[c]
        strong = r["bands"].get("strong", 0)
        channel = load(f"forgery_chain_{c}.json")["lure_transfers_by_contract"]
        if strong != len(channel):
            raise SystemExit(f"{c}: rule_interaction counts {strong} strong but the channel census has "
                             f"{len(channel)} flagged contracts; the two artefacts disagree")
        _count_or_word(m, f"RuleShortcuts{tex}", r["shortcut_contracts"])
        m.pct(f"RuleShortcutVol{tex}", r["shortcut_transfer_share"] or 0.0, 2)
        _count_or_word(m, f"RuleDestOnlyWeak{tex}", r["lexicon_bound"]["destination_only_weak"])
        _count_or_word(m, f"RuleNonLatin{tex}", r["script_census"].get("any_non_latin", 0))
        _count_or_word(m, f"RuleLookalike{tex}", r["non_latin_lookalike_only"])
        tot_strong += strong
        tot_short += r["shortcut_contracts"]
        tot_nonlatin += r["script_census"].get("any_non_latin", 0)
        tot_look += r["non_latin_lookalike_only"]
        tot_added += len(r["allowlist_off"]["added"])
        if r["strong_without_transfer_count"]:
            raise SystemExit(f"{c}: flagged contracts without a transfer count {r['strong_without_transfer_count'][:3]}")
    m.int_("RuleFlaggedTotal", tot_strong)
    _count_or_word(m, "RuleShortcutsTotal", tot_short)
    _count_or_word(m, "RuleAllowlistAddedTotal", tot_added, "strong contracts added with the allowlist emptied")
    m.int_("RulePopulationTotal", sum(ri[c]["population"] for c in chains),
           "examined frame contracts with resolved metadata, all chains")
    m.int_("RuleNonLatinTotal", tot_nonlatin)
    m.int_("RuleLookalikeTotal", tot_look)
    cjk = sum(ri[c]["script_census"].get("cjk", 0) for c in chains)
    _count_or_word(m, "RuleCjkTotal", cjk)
    dest_nonlatin = sum(ri[c]["lexicon_bound"]["destination_only_weak_with_non_latin"] for c in chains)
    dest_imitating = sum(ri[c]["lexicon_bound"]["destination_only_weak_non_latin_lookalike_only"] for c in chains)
    if dest_imitating != dest_nonlatin:
        raise SystemExit(f"{dest_nonlatin - dest_imitating} destination-only payloads carry letters of another "
                         "language; the text says none does. Review them before regenerating.")
    _count_or_word(m, "RuleDestOnlyNonLatinTotal", dest_nonlatin,
                   "destination-only payloads with non-Latin letters, every one imitating Latin")
    # List sizes are read from the frozen detector, never typed.
    import sys
    sys.path.insert(0, str(paper.parents[1] / "analysis"))
    from src import metadata_lure as ml
    _count_or_word(m, "RuleBenignDomains", len(ml.BENIGN_DOMAINS))
    m.int_("RuleAggregatorHosts", len(ml.AGGREGATOR_HOSTS))
    m.int_("RuleMultiSuffixes", len(ml.MULTI_SUFFIXES))
    m.int_("RuleIanaTlds", len(ml.LURE_TLDS))

    # ---------------------------------------------------------------- WP3: frame probe
    m.section("R1 · ledger probe over every examined contract (frame_probe_<chain>.json)")
    flagged_fab = flagged_led = unflag_fab = unflag_led = 0
    for c in chains:
        tex = chain_tex[c][0]
        d = load(f"frame_probe_{c}.json")
        if d.get("partial"):
            raise SystemExit(f"{c}: frame probe is PARTIAL; refusing to emit its numbers")
        s = d["flagged_vs_not"]
        reverts = sum(1 for r in d["per_contract"].values() if r["verdict"] == "balance_reverts")
        for name, v in (("ProbeFlagFab", s["flagged"]["fabricated"]), ("ProbeFlagLedger", s["flagged"]["zero_ledger"]),
                        ("ProbeUnflagFab", s["not_flagged"]["fabricated"]),
                        ("ProbeUnflagLedger", s["not_flagged"]["zero_ledger"]),
                        ("ProbeNoResponse", d["no_response"]), ("ProbeReverts", reverts)):
            _count_or_word(m, f"{name}{tex}", v)
            m.int_(f"{name}Tab{tex}", v, "numeral form, for tables")
        flagged_fab += s["flagged"]["fabricated"]
        flagged_led += s["flagged"]["zero_ledger"]
        unflag_fab += s["not_flagged"]["fabricated"]
        unflag_led += s["not_flagged"]["zero_ledger"]
        _count_or_word(m, f"ProbeControls{tex}", d["controls"]["n"])
        if any(v in FABRICATED for v in d["controls"]["verdicts"]):
            raise SystemExit(f"{c}: a registry control looks fabricated; the probe would not discriminate")
    m.int_("ProbeFlagFabTotal", flagged_fab)
    m.int_("ProbeFlagLedgerTotal", flagged_led)
    m.int_("ProbeUnflagFabTotal", unflag_fab)
    m.int_("ProbeUnflagLedgerTotal", unflag_led)
    # Composition of the unflagged fabricators, read from the probe and the frame metadata.
    zk = load("frame_probe_zksync-era.json")
    _count_or_word(m, "ProbeUnflagFabWeakZKsyncEra",
                   sum(1 for a in zk["fabricated_not_flagged"] if zk["per_contract"][a]["band"] == "weak"))
    # How much of each chain's lexicon bound the ledger probe shows to be fabricating.
    ri_dest = load("rule_interaction.json")
    realised, understated = {}, {}
    for c in chains:
        probe = load(f"frame_probe_{c}.json")
        dest = set(ri_dest[c]["lexicon_bound"]["destination_only_weak_addresses"])
        realised[c] = len(dest & set(probe["fabricated_not_flagged"]))
        understated[c] = realised[c] / ri_dest[c]["bands"]["strong"]
        _count_or_word(m, f"ProbeFabDestOnly{chain_tex[c][0]}", realised[c],
                       "destination-only payloads (lexicon bound) that fabricate balances")
    # The sensitivity text says most of zkSync Era's bound is realised and that its flagged count is the
    # most understated. Both are read from the data here.
    zk_bound = ri_dest["zksync-era"]["lexicon_bound"]["destination_only_weak"]
    if not (2 * realised["zksync-era"] > zk_bound and max(understated, key=understated.get) == "zksync-era"):
        raise SystemExit("zkSync Era no longer realises most of its lexicon bound, or is no longer the most "
                         "understated chain; revise the lexicon paragraph in sensitivity.tex")
    arb = load("frame_probe_arbitrum-one.json")
    arb_md = json.loads((paper / "data" / "token_metadata_arbitrum-one.json").read_text())
    usdc = sum(1 for a in arb["fabricated_not_flagged"]
               if (arb_md.get(a) or {}).get("name") == "USDC" and (arb_md.get(a) or {}).get("symbol") == "USDC")
    m.int_("ProbeArbCounterfeitUsdc", usdc, "unflagged Arbitrum fabricators named exactly USDC/USDC")
    _count_or_word(m, "ProbeArbUnflagFabNone",
                   sum(1 for a in arb["fabricated_not_flagged"] if arb["per_contract"][a]["band"] == "none"))

    # ---------------------------------------------------------------- WP4: tail sample and volume
    m.section("R1 · probability sample below the frame (tail_sample / tail_volume)")
    per = {load(f"tail_sample_{c}.json")["per_stratum"] for c in chains}
    bands = {len(load(f"tail_sample_{c}.json")["bands"]) for c in chains}
    if len(per) != 1 or len(bands) != 1:
        raise SystemExit(f"tail design differs across chains: per_stratum {per}, bands {bands}")
    m.int_("TailPerStratum", per.pop(), "contracts drawn per rank band (design constant)")
    _count_or_word(m, "TailBands", bands.pop(), "log-spaced rank bands below the frame")
    for c in chains:
        tex = chain_tex[c][0]
        t = load(f"tail_sample_{c}.json")
        v = load(f"tail_volume_{c}.json")
        frame = sum(int(n) for n in load(f"forgery_chain_{c}.json")["lure_transfers_by_contract"].values())
        if v["frame_flagged_transfers"] != frame:
            raise SystemExit(f"{c}: tail_volume's frame volume disagrees with the channel census")
        m.int_(f"TailContracts{tex}", t["tail_contracts"])
        m.int_(f"TailSampled{tex}", sum(p["resolved"] for p in t["per_band"].values()))
        _count_or_word(m, f"TailFlaggedSampled{tex}", t["observed_strong"])
        m.int_(f"TailFlaggedSampledTab{tex}", t["observed_strong"], "numeral form, for tables")
        _count_or_word(m, f"TailNewKeys{tex}", len(t["new_campaign_keys"]))
        m.pct(f"TailCoverage{tex}", t["metadata_coverage"], 1)
        ht = v["ht_flagged_transfers"]
        m.int_(f"TailHtVol{tex}", round(ht["point"]))
        m.int_(f"TailHtVolLo{tex}", round(ht["ci95"][0]))
        m.int_(f"TailHtVolHi{tex}", round(ht["ci95"][1]))
        m.dec(f"TailHtVolM{tex}", ht["point"] / 1e6, 2, "millions, for tables")
        m.dec(f"TailHtVolLoM{tex}", ht["ci95"][0] / 1e6, 2)
        m.dec(f"TailHtVolHiM{tex}", ht["ci95"][1] / 1e6, 2)
        m.dec(f"FrameVolM{tex}", frame / 1e6, 2, "frame flagged transfers, millions")
        hc = v["ht_flagged_contracts"]
        m.int_(f"TailHtContracts{tex}", round(hc["point"]))
        m.int_(f"TailHtContractsLo{tex}", round(hc["ci95"][0]))
        m.int_(f"TailHtContractsHi{tex}", round(hc["ci95"][1]))
        m.int_(f"TailUpperContracts{tex}", hc["exact_bonferroni_upper95"])
        m.dec(f"TailRatio{tex}", v["tail_to_frame_volume_ratio"]["point"], 2)
        m.dec(f"TailRatioLo{tex}", v["tail_to_frame_volume_ratio"]["ci95"][0], 2)
        m.dec(f"TailRatioHi{tex}", v["tail_to_frame_volume_ratio"]["ci95"][1], 2)
        m.dec(f"TailWorstRatio{tex}", v["worst_case_tail_to_frame_ratio"], 3)
        # Frame share of flagged volume: frame / (frame + tail). The point and the interval come
        # from the HT ratio; the worst case comes from the deterministic bound.
        r_pt, r_lo, r_hi = (v["tail_to_frame_volume_ratio"]["point"], *v["tail_to_frame_volume_ratio"]["ci95"])
        m.pct(f"FrameShare{tex}", 1 / (1 + r_pt), 1)
        m.pct(f"FrameShareLo{tex}", 1 / (1 + r_hi), 1)
        m.pct(f"FrameShareHi{tex}", 1 / (1 + r_lo), 1)
        m.pct(f"FrameShareWorst{tex}", 1 / (1 + v["worst_case_tail_to_frame_ratio"]), 1)

    # ---------------------------------------------------------------- WP7: historical metadata
    m.section("R1 · historical metadata on every chain (metadata_history_<chain>.json)")
    comp = stable = 0
    for c in chains:
        tex = chain_tex[c][0]
        d = load(f"metadata_history_{c}.json")
        verdicts = d["by_verdict"]
        changed = verdicts.get("changed", 0)
        n_stable = verdicts.get("stable", 0)
        if n_stable + changed != d["comparable_contracts"]:
            raise SystemExit(f"{c}: metadata_history verdicts do not sum to the comparable count")
        _count_or_word(m, f"MetaHistFlagged{tex}", d["flagged_contracts"])
        _count_or_word(m, f"MetaHistComparableAll{tex}", d["comparable_contracts"])
        _count_or_word(m, f"MetaHistChangedAll{tex}", changed)
        comp += d["comparable_contracts"]
        stable += n_stable
    m.int_("MetaHistComparableTotal", comp)
    m.int_("MetaHistStableTotal", stable)
    if stable != comp:              # the findings table and conclusions say no answering contract changed
        raise SystemExit(f"{comp - stable} answering contracts changed their payload; revise the text")

    # ---------------------------------------------------------------- WP6: negative controls
    m.section("R1 · negative controls (negative_controls_<chain>.json)")
    nft = reg = nft_flag = reg_flag = fab = nft_below2 = nft_measured = 0
    for c in chains:
        tex = chain_tex[c][0]
        s = load(f"negative_controls_{c}.json")["summary"]
        for kind, store in (("nft", "Nft"), ("registry", "Reg")):
            if kind not in s:
                continue
            k = s[kind]
            fab_here = sum(n for v, n in k["ledger"].items() if v in FABRICATED)
            fab += fab_here
            if kind == "nft":
                nft += k["metadata_resolved"]
                nft_flag += k["flagged_strong"]
                nft_below2 += k["fanout_below_2"]
                nft_measured += k["fanout_measured"]
                m.dec(f"NegNftFanoutMedian{tex}", k["fanout_median"], 2)
            else:
                reg += k["metadata_resolved"]
                reg_flag += k["flagged_strong"]
    m.int_("NegNftTotal", nft)
    m.int_("NegRegTotal", reg)
    _count_or_word(m, "NegNftFlagged", nft_flag)
    _count_or_word(m, "NegRegFlagged", reg_flag)
    # Sensitivity and the response say no control returned a fabricated balance. Checked, not assumed.
    if fab:
        raise SystemExit(f"{fab} negative controls returned a fabricated balance; revise sensitivity.tex")
    _count_or_word(m, "NegNftFanoutBelowTwo", nft_below2)
    m.int_("NegNftFanoutMeasured", nft_measured)

    # ---------------------------------------------------------------- WP8: activity robustness
    m.section("R1 · recipient-activity robustness (recipient_activity_robust_<chain>.json)")
    act_chains = ("arbitrum-one", "base", "optimism", "polygon-zkevm")
    shift = max(abs(load(f"recipient_activity_robust_{c}.json")["V3_cleaned_control"]["difference"]["difference"]
                    - load(f"recipient_activity_robust_{c}.json")["V1_published"]["difference"]["difference"])
                for c in act_chains)
    m.dec("ActCleanShiftMax", shift * 100, 2, "largest |V3 - V1| difference, percentage points")
    aligned_positive = all(load(f"recipient_activity_robust_{c}.json")["V2_time_aligned"]["difference"]["ci95_lo"] > 0
                           for c in act_chains)
    if not aligned_positive:
        raise SystemExit("a time-aligned difference interval includes zero; the prose claiming a positive "
                         "aligned contrast on every chain would be false")
    for c in act_chains:
        tex = chain_tex[c][0]
        d = load(f"recipient_activity_robust_{c}.json")
        pub = load(f"recipient_activity_{c}.json")
        v1 = d["V1_published"]
        if (v1["recipients"]["hits"], v1["control"]["hits"]) != (pub["active"]["hits"], pub["control"]["active"]["hits"]):
            raise SystemExit(f"{c}: robustness V1 does not reproduce the published activity counts")
        for tag, key in (("Pub", "V1_published"), ("Aligned", "V2_time_aligned"), ("Clean", "V3_cleaned_control")):
            diff = d[key]["difference"]
            m.dec(f"Act{tag}Diff{tex}", diff["difference"] * 100, 2, "percentage points")
            m.dec(f"Act{tag}DiffLo{tex}", diff["ci95_lo"] * 100, 2)
            m.dec(f"Act{tag}DiffHi{tex}", diff["ci95_hi"] * 100, 2)
            m.dec(f"Act{tag}Ratio{tex}", diff["ratio"], 2)
        m.pct(f"ActAlignedRecip{tex}", d["V2_time_aligned"]["recipients"]["estimate"], 2)
        m.pct(f"ActAlignedControl{tex}", d["V2_time_aligned"]["control"]["estimate"], 2)
        m.pct(f"ActFreshShare{tex}", 1 - d["V2_time_aligned"]["recipients"]["n"] / v1["recipients"]["n"], 0,
              "named recipients never seen in a non-lure row before their first naming")
