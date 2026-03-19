"""
Comprehensive pipeline test — run before training.

Tests:
  1. FEATURE_DIM consistency (C++ 77 == Python 77)
  2. C++ feature shape & value sanity
  3. Board texture edge cases (rainbow / two-suited / monotone)
  4. PostflopAdvantageNet / PostflopStrategyNet input/output shapes
  5. ReservoirBuffer add_batch + sample shape
  6. Discard CFR features, net, strategy
  7. Preflop tabular CFR canonicalize + is_ready
  8. run_traversals_batched (warmup): buffer shapes
  9. Checkpoint save / load round-trip
  10. Phase1/2/3 is_complete logic
"""

import sys, os, re, ctypes, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

PASS  = "\033[32m✔\033[0m"
FAIL  = "\033[31m✘\033[0m"
errors = []

def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}  {detail}")
        errors.append(name)

def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE_DIM consistency
# ─────────────────────────────────────────────────────────────────────────────
section("1. FEATURE_DIM consistency")

from game.constants import FEATURE_DIM as PY_DIM, NUM_ACTIONS
from game.features import _c_lib

check("Python FEATURE_DIM == 77", PY_DIM == 77, f"got {PY_DIM}")

hdr = open(os.path.join(os.path.dirname(__file__), "cpp", "core", "constants.h")).read()
m = re.search(r"FEATURE_DIM\s*=\s*(\d+)", hdr)
cpp_dim = int(m.group(1)) if m else None
check("C++ FEATURE_DIM == 77", cpp_dim == 77, f"got {cpp_dim}")
check("Python == C++", PY_DIM == cpp_dim)

from postflop_cfr.buffers import _FEAT_DIM as BUF_DIM
check("ReservoirBuffer _FEAT_DIM == 77", BUF_DIM == 77, f"got {BUF_DIM}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. C++ feature shape & value sanity
# ─────────────────────────────────────────────────────────────────────────────
section("2. C++ c_state_features shape & values")

def call_features(hand2_ints, comm_ints, n_comm, my_bet, opp_bet,
                  street, is_bb, my_disc=(-1,-1,-1), opp_disc=(-1,-1,-1)):
    f = np.zeros(PY_DIM, dtype=np.float32)
    h2 = (ctypes.c_int * 2)(*hand2_ints)
    bc = (ctypes.c_int * 5)(*[comm_ints[i] if i<len(comm_ints) else -1 for i in range(5)])
    md = (ctypes.c_int * 3)(*my_disc)
    od = (ctypes.c_int * 3)(*opp_disc)
    bf = (ctypes.c_int * 8)(*[0]*8)
    _c_lib.c_state_features(
        h2, bc, ctypes.c_int(n_comm),
        ctypes.c_int(my_bet), ctypes.c_int(opp_bet),
        ctypes.c_int(street), ctypes.c_int(int(is_bb)),
        md, od,
        f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bf, None, None, ctypes.c_int(0), ctypes.c_int(0),
    )
    return f

# Flop, to_call=4, is_bb=True
f = call_features([0, 9], [3, 12, 21], 3, 4, 8, 1, True,
                   opp_disc=(1, 2, 10))

check("output len == 77", len(f) == 77)
check("no NaN/Inf",       np.all(np.isfinite(f)), f"bad indices={np.where(~np.isfinite(f))}")
check("my_cat sums to 1",       abs(f[0:17].sum()  - 1.) < 1e-4, f"sum={f[0:17].sum():.4f}")
check("my_range_cats sums to 1",abs(f[17:34].sum() - 1.) < 1e-4, f"sum={f[17:34].sum():.4f}")
check("opp_range_cats sums to 1",abs(f[34:51].sum()- 1.) < 1e-4, f"sum={f[34:51].sum():.4f}")
check("my_cat is one-hot (max>0.5)", f[0:17].max() > 0.5)
# my_range_cats must NOT be uniform when opp_disc is provided (we passed my_disc=(-1,-1,-1) but opp_disc=(1,2,10))
# Verify with explicit my_disc != (-1,-1,-1) to confirm discard-based update
f_with_mydisc = call_features([0, 9], [3, 12, 21], 3, 4, 8, 1, True,
                               my_disc=(5, 6, 7), opp_disc=(1, 2, 10))
check("my_range_cats non-uniform when my_disc known",
      f_with_mydisc[17:34].max() > 1./17 * 1.5,
      f"max={f_with_mydisc[17:34].max():.3f} vs uniform={1./17:.3f}")
check("my_range_cats sums to 1 with discards",
      abs(f_with_mydisc[17:34].sum() - 1.) < 1e-4)
check("street[73]=1 (flop)",  abs(f[73] - 1.) < 1e-4, f"street={f[73:76]}")
check("position is_bb=1",     abs(f[76] - 1.) < 1e-4, f"pos={f[76]}")
check("bet_facing=1",         abs(f[61] - 1.) < 1e-4)   # to_call = 8-4 = 4 > 0
check("can_check=0",          abs(f[62] - 0.) < 1e-4)

pot = 4 + 8; raise_room = 100 - 8
check("raise_room/pot",
      abs(f[68] - raise_room/pot) < 1e-3,
      f"expected {raise_room/pot:.3f} got {f[68]:.3f}")

# Check node where to_call=0 → can_check=1
f2 = call_features([0, 9], [3, 12, 21], 3, 8, 8, 1, True)
check("can_check=1 when no bet facing", abs(f2[62] - 1.) < 1e-4, f"val={f2[62]}")
check("bet_facing=0 when no bet",       abs(f2[61] - 0.) < 1e-4, f"val={f2[61]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Board texture edge cases
# ─────────────────────────────────────────────────────────────────────────────
section("3. Board texture (rainbow / two-suited / monotone)")

def board_tex(board_ints):
    f = call_features([4, 5], board_ints, len(board_ints), 0, 0, 1, False)
    return f[51:59]   # 8-dim slice: [paired, fc, fd, conn, hi, mono, two_s, coord]

# Cards: suit*9 + rank.  ♦=0,♥=1,♠=2
# Rainbow: 2♦(0) 2♥(9) 2♠(18) — different suits
rain = board_tex([0, 9, 18])
check("rainbow fd_present=0",   abs(rain[2] - 0.) < 1e-4, f"{rain[2]:.2f}")
check("rainbow two_suited=0",   abs(rain[6] - 0.) < 1e-4, f"{rain[6]:.2f}")
check("rainbow monotone=0",     abs(rain[5] - 0.) < 1e-4, f"{rain[5]:.2f}")

# Two-suited: 2♦(0) 3♦(1) 2♥(9) — 2 diamonds + 1 heart
two = board_tex([0, 1, 9])
check("two-suited fd_present=1",  abs(two[2] - 1.) < 1e-4, f"{two[2]:.2f}")
check("two-suited two_suited=1",  abs(two[6] - 1.) < 1e-4, f"{two[6]:.2f}")
check("two-suited monotone=0",    abs(two[5] - 0.) < 1e-4, f"{two[5]:.2f}")

# Monotone: 2♦(0) 3♦(1) 4♦(2) — all diamonds
mono = board_tex([0, 1, 2])
check("monotone fd_present=1",    abs(mono[2] - 1.) < 1e-4, f"{mono[2]:.2f}")
check("monotone two_suited=0",    abs(mono[6] - 0.) < 1e-4, f"{mono[6]:.2f}")
check("monotone monotone=1",      abs(mono[5] - 1.) < 1e-4, f"{mono[5]:.2f}")
check("monotone flush_complete=1",abs(mono[1] - 1.) < 1e-4, f"{mono[1]:.2f}")

# Paired board
paired = board_tex([0, 9, 1])   # 2♦ 2♥ 3♦  — ranks 0,0,1  → paired
check("paired board: paired=1",   abs(paired[0] - 1.) < 1e-4, f"{paired[0]:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Net input/output shapes
# ─────────────────────────────────────────────────────────────────────────────
section("4. Net shapes (adv + strategy)")

from postflop_cfr import PostflopAdvantageNet, PostflopStrategyNet

adv = PostflopAdvantageNet()
stn = PostflopStrategyNet()
x   = torch.randn(16, PY_DIM)

with torch.no_grad():
    a_out = adv(x)
    s_out = stn(x)

check("adv input_dim == 77",    adv.embed[0].in_features == PY_DIM, f"got {adv.embed[0].in_features}")
check("adv output (16,8)",      a_out.shape == (16, NUM_ACTIONS),   f"got {a_out.shape}")
check("str input_dim == 77",    stn.embed[0].in_features == PY_DIM, f"got {stn.embed[0].in_features}")
check("str output (16,8)",      s_out.shape == (16, NUM_ACTIONS),   f"got {s_out.shape}")
check("no NaN in adv output",   torch.all(torch.isfinite(a_out)).item())
check("no NaN in str output",   torch.all(torch.isfinite(s_out)).item())

# ─────────────────────────────────────────────────────────────────────────────
# 5. ReservoirBuffer add_batch + sample
# ─────────────────────────────────────────────────────────────────────────────
section("5. ReservoirBuffer")

from postflop_cfr.buffers import ReservoirBuffer

buf   = ReservoirBuffer(10_000)
feats = np.random.randn(64, PY_DIM).astype(np.float32)
vals  = np.random.randn(64, NUM_ACTIONS).astype(np.float32)
iters = np.ones(64, dtype=np.float32)
masks = np.ones((64, NUM_ACTIONS), dtype=np.float32)
strs  = np.full(64, 1, dtype=np.int32)   # all flop

buf.add_batch(feats, vals, iters, masks, strs)
check("len(buf) == 64",         len(buf) == 64)
check("street_bufs[1] truthy",  bool(buf.street_bufs[1]))
check("street_bufs[2] falsy",   not bool(buf.street_bufs[2]))

result = buf.sample(32)
check("sample returns tuple",   result is not None and len(result) == 4)
sf, sv, si, sm = result
check("sample feats shape",     sf.shape[1] == PY_DIM,      f"got {sf.shape}")
check("sample vals shape",      sv.shape[1] == NUM_ACTIONS, f"got {sv.shape}")
check("sample size <= 32",      sf.shape[0] <= 32)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Discard CFR pipeline
# ─────────────────────────────────────────────────────────────────────────────
section("6. Discard CFR pipeline")

from discard_cfr.cfr import DiscardCFR
from discard_cfr.features import FEAT_DIM as D_DIM

check("discard FEAT_DIM == 44", D_DIM == 44, f"got {D_DIM}")

from discard_cfr.features import PAIR_DIM, CTX_DIM
dc = DiscardCFR()
check("discard pair_layer in == PAIR_DIM (23)",
      dc.net.pair_layer.in_features == PAIR_DIM,
      f"got {dc.net.pair_layer.in_features}")
check("discard ctx_layer in == CTX_DIM (21)",
      dc.net.ctx_layer.in_features == CTX_DIM,
      f"got {dc.net.ctx_layer.in_features}")
check("discard output is scalar per pair",
      dc.net.output[-1].out_features == 1,
      f"got {dc.net.output[-1].out_features}")

p0h5  = np.array([[0,1,2,3,4]], dtype=np.int32)
p1h5  = np.array([[9,10,11,12,13]], dtype=np.int32)
comms = np.array([[5,6,7,8,14]], dtype=np.int32)
dc.run_iter(p0h5, p1h5, comms)
check("discard buf non-empty after iter", len(dc.buf) > 0)
loss = dc.train()
check("discard train loss is finite", np.isfinite(float(loss)), f"loss={loss}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Preflop CFR
# ─────────────────────────────────────────────────────────────────────────────
section("7. Preflop tabular CFR")

from preflop_cfr.canonical import canonicalize
from preflop_cfr.cfr import PreflopCFR

hand5 = [0, 1, 2, 3, 4]
canon = canonicalize(hand5)
check("canonical is tuple",       isinstance(canon, tuple))
check("canonical len == 5",       len(canon) == 5)
check("canonical is sorted",      list(canon) == sorted(canon))
check("suit-perm same canonical",
      canonicalize([0,1,2,3,4]) == canonicalize([9,10,11,12,13]))

class _FakeTrainer:
    preflop_strategy_sum = {}

cfr_empty = PreflopCFR(trainer_state=_FakeTrainer())
check("PreflopCFR.is_ready(empty) = False", not cfr_empty.is_ready())

# ─────────────────────────────────────────────────────────────────────────────
# 8. run_traversals_batched → buffer shapes & content
# ─────────────────────────────────────────────────────────────────────────────
section("8. run_traversals_batched (warmup phase)")

from deep_cfr import DeepCFR
from postflop_cfr.traversal import run_traversals_batched, _postflop_ready

from postflop_cfr.traversal import MIN_WARMUP_SAMPLES
from postflop_cfr.buffers import ReservoirBuffer as RB

trainer = DeepCFR()

# ── Warmup traversals: only fills preflop CFR, not postflop buffers ──────────
run_traversals_batched(trainer, traversals_per_iter=100,
                       traversing_player=0, discard_trainer=None)
check("warmup: preflop_regrets filled",     len(trainer.preflop_regrets) > 0)
check("warmup: strategy_sum filled",        len(trainer.preflop_strategy_sum) > 0)
check("warmup: adv_buffers empty (correct)",
      sum(len(b) for b in trainer.adv_buffers) == 0)

# ── Force neural mode by pre-seeding buffers past threshold ──────────────────
dummy_f  = np.random.randn(MIN_WARMUP_SAMPLES + 1, PY_DIM).astype(np.float32)
dummy_v  = np.random.randn(MIN_WARMUP_SAMPLES + 1, NUM_ACTIONS).astype(np.float32)
dummy_i  = np.ones(MIN_WARMUP_SAMPLES + 1, dtype=np.float32)
dummy_m  = np.ones((MIN_WARMUP_SAMPLES + 1, NUM_ACTIONS), dtype=np.float32)
for s in [1, 2, 3]:
    for p in range(2):
        trainer.adv_buffers[p].add_batch(
            dummy_f, dummy_v, dummy_i, dummy_m,
            np.full(MIN_WARMUP_SAMPLES + 1, s, dtype=np.int32))

check("neural mode ready (_postflop_ready)", _postflop_ready(trainer))

# Now run neural traversal and verify buffers grow
buf_before = sum(len(b) for b in trainer.adv_buffers)
run_traversals_batched(trainer, traversals_per_iter=50,
                       traversing_player=0, discard_trainer=None)
buf_after = sum(len(b) for b in trainer.adv_buffers)
check("neural: adv_buffers grew",       buf_after > buf_before,
      f"before={buf_before} after={buf_after}")
check("neural: strategy_buffer non-empty", len(trainer.strategy_buffer) > 0)

# Verify feature dim in newly collected samples
for p in range(2):
    for s in [1, 2, 3]:
        if trainer.adv_buffers[p].street_bufs[s]:
            r = trainer.adv_buffers[p].sample_street(s, 4)
            if r:
                check(f"adv_buf[{p}] street {s} feat dim == 77",
                      r[0].shape[1] == PY_DIM, f"got {r[0].shape[1]}")
            break

# ─────────────────────────────────────────────────────────────────────────────
# 9. Checkpoint save / load round-trip
# ─────────────────────────────────────────────────────────────────────────────
section("9. Checkpoint save / load")

from postflop_cfr.checkpoint import save_checkpoint, load_checkpoint
from postflop_cfr.training import train_adv_networks

trainer.iteration = 7
run_traversals_batched(trainer, traversals_per_iter=100, traversing_player=0)
train_adv_networks(trainer)

with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
    ckpt = tmp.name

save_checkpoint(trainer, ckpt, iteration=7, save_buffers=False)

trainer2      = DeepCFR()
iter_loaded   = load_checkpoint(trainer2, ckpt)
check("loaded iteration == 7", iter_loaded == 7, f"got {iter_loaded}")

for p in range(2):
    w1 = trainer.adv_nets[p].head.weight.cpu()
    w2 = trainer2.adv_nets[p].head.weight.cpu()
    check(f"adv_net[{p}] weights match",
          torch.allclose(w1, w2), f"max_diff={(w1-w2).abs().max():.6f}")

w1 = trainer.strategy_net.head.weight.cpu()
w2 = trainer2.strategy_net.head.weight.cpu()
check("strategy_net weights match", torch.allclose(w1, w2))

os.unlink(ckpt)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Phase transition logic
# ─────────────────────────────────────────────────────────────────────────────
section("10. Phase transition logic")

from training_phase import Phase1, Phase2, Phase3, PhaseStats

# Phase1 thresholds: MIN_PF_INFOSETS=20, MIN_PF_VISITS=5000
s_small = PhaseStats(iteration=10, pf_infosets=5,  pf_visits=100.)
s_big   = PhaseStats(iteration=50, pf_infosets=25, pf_visits=6000.)
check("Phase1 not complete (few infosets)", not Phase1().is_complete(s_small))
check("Phase1 complete (enough infosets)",       Phase1().is_complete(s_big))

# Phase2: plateau detection (PATIENCE=10, DELTA=0.02, MIN_ITERS=20)
s_flat  = PhaseStats(iteration=25,
                     discard_loss_history=[0.50]*10 + [0.499]*10)  # flat
s_nocon = PhaseStats(iteration=25,
                     discard_loss_history=[1.0, 0.8, 0.6, 0.4, 0.2])  # still dropping
check("Phase2 not complete (still converging)", not Phase2().is_complete(s_nocon))
check("Phase2 complete (plateau detected)",          Phase2().is_complete(s_flat))

# Phase3 never terminates
check("Phase3 never complete", not Phase3().is_complete(s_big))

# ─────────────────────────────────────────────────────────────────────────────
# 11. Range propagation semantic correctness
# ─────────────────────────────────────────────────────────────────────────────
section("11. Range propagation correctness")

from range_finder import RangeFinder, N_HANDS

ALL_PAIRS_TEST = [(a, b) for a in range(27) for b in range(a + 1, 27)]

# ── 11a. c_range_update_discard removes impossible hands (no fast_score) ──────
disc3 = [1, 2, 10]
disc_set = set(disc3)

rf_test = RangeFinder()
rf_test.init(dead_cards=[])                    # all 351 uniform
rf_test.update_discard(disc3, [5, 6, 7])       # board3 now ignored

probs = rf_test.get_range_array()
bad_hands = [(a, b) for i, (a, b) in enumerate(ALL_PAIRS_TEST)
             if (a in disc_set or b in disc_set) and probs[i] > 1e-9]
check("discard_update: impossible hands zeroed",
      len(bad_hands) == 0, f"{len(bad_hands)} non-zero impossible hands")
check("discard_update: range still sums to 1",
      abs(probs.sum() - 1.) < 1e-5, f"sum={probs.sum():.6f}")
valid_n = sum(1 for a, b in ALL_PAIRS_TEST
              if a not in disc_set and b not in disc_set)
alive_n = int((probs > 1e-9).sum())
check(f"discard_update: exactly {valid_n} valid hands survive",
      alive_n == valid_n, f"alive={alive_n} expected={valid_n}")

# ── 11b. C++ opp_range_cats changes after seeing discards ─────────────────────
f_opp_disc  = call_features([0, 9], [3, 12, 21], 3, 0, 0, 1, False,
                              opp_disc=(1, 2, 10))
f_opp_nodisc = call_features([0, 9], [3, 12, 21], 3, 0, 0, 1, False,
                               opp_disc=(-1,-1,-1))

cats_with    = f_opp_disc[34:51]
cats_nodisc  = f_opp_nodisc[34:51]
check("opp_range_cats: non-uniform after discard",
      not np.allclose(cats_with, cats_nodisc, atol=1e-3),
      f"max_diff={np.abs(cats_with-cats_nodisc).max():.4f}")
check("opp_range_cats: sums to 1", abs(cats_with.sum() - 1.) < 1e-4)

# Cross-check: RangeFinder gives same categories
rf_cross = RangeFinder()
rf_cross.init(dead_cards=[0, 9, 3, 12, 21])   # my_hand + community
rf_cross.update_discard([1, 2, 10], [3, 12, 21])
cats_rf = rf_cross.category_probs([3, 12, 21, -1, -1], 0.)
check("opp_range_cats: matches RangeFinder cross-check",
      np.allclose(cats_with, cats_rf, atol=1e-3),
      f"max_diff={np.abs(cats_with-cats_rf).max():.4f}")

# ── 11c. my_range_cats excludes my own discarded cards ────────────────────────
# opp perspective: dead = community only, update = my discards
rf_myrange = RangeFinder()
rf_myrange.init(dead_cards=[3, 12, 21])        # community only
rf_myrange.update_discard([1, 2, 10], [3, 12, 21])
probs_my = rf_myrange.get_range_array()

bad_my = [(a, b) for i, (a, b) in enumerate(ALL_PAIRS_TEST)
           if (a in disc_set or b in disc_set) and probs_my[i] > 1e-9]
check("my_range: impossible hands zeroed after my discard",
      len(bad_my) == 0, f"{len(bad_my)} non-zero impossible hands")

f_my_disc = call_features([0, 9], [3, 12, 21], 3, 0, 0, 1, False,
                           my_disc=(1, 2, 10))
cats_my_cpp = f_my_disc[17:34]
check("my_range_cats: non-uniform after seeing my discards",
      cats_my_cpp.max() > 1./PY_DIM,   # above uniform baseline
      f"max={cats_my_cpp.max():.3f}")

# ── 11d. _recompute_discards_with_cfr: cards partition correctly ──────────────
from postflop_cfr.traversal import _recompute_discards_with_cfr
from discard_cfr.cfr import DiscardCFR
from game.features import batch_deal_discard

dc_test = DiscardCFR()
N_TEST = 30
r_test  = batch_deal_discard(N_TEST)
p0h_o, p1h_o, _, _, comms_t, p0h5_t, p1h5_t = r_test
p0h_n, p1h_n, p0d_n, p1d_n = _recompute_discards_with_cfr(
    p0h5_t, p1h5_t, comms_t, dc_test)

ok_p0 = ok_p1 = True
for i in range(N_TEST):
    orig_A = set(int(x) for x in p0h5_t[i])
    kept_A = set(int(x) for x in p0h_n[i])
    disc_A = set(int(x) for x in p0d_n[i])
    if kept_A | disc_A != orig_A or len(kept_A) != 2 or len(disc_A) != 3:
        ok_p0 = False; break
    orig_B = set(int(x) for x in p1h5_t[i])
    kept_B = set(int(x) for x in p1h_n[i])
    disc_B = set(int(x) for x in p1d_n[i])
    if kept_B | disc_B != orig_B or len(kept_B) != 2 or len(disc_B) != 3:
        ok_p1 = False; break
check(f"recompute_discards p0: kept+discarded = original 5 cards ({N_TEST} games)", ok_p0)
check(f"recompute_discards p1: kept+discarded = original 5 cards ({N_TEST} games)", ok_p1)

# ── 11e. Real traversal buffer features have valid range distributions ─────────
# Fresh trainer: run warmup + neural, sample ONLY neural-mode features
_fresh = DeepCFR()
# Seed to just above threshold with REALISTIC range data (from actual C++ features)
_seed_f = np.zeros((MIN_WARMUP_SAMPLES + 1, PY_DIM), dtype=np.float32)
for _si in range(MIN_WARMUP_SAMPLES + 1):
    _seed_f[_si, 17:34] = np.ones(17, dtype=np.float32) / 17   # uniform my_range
    _seed_f[_si, 34:51] = np.ones(17, dtype=np.float32) / 17   # uniform opp_range
    _seed_f[_si, 73] = 1.   # flop one-hot
_dummy_v = np.zeros((MIN_WARMUP_SAMPLES+1, NUM_ACTIONS), dtype=np.float32)
_dummy_i = np.ones(MIN_WARMUP_SAMPLES+1, dtype=np.float32)
_dummy_m = np.ones((MIN_WARMUP_SAMPLES+1, NUM_ACTIONS), dtype=np.float32)
for _s in [1,2,3]:
    for _p in range(2):
        _fresh.adv_buffers[_p].add_batch(
            _seed_f, _dummy_v, _dummy_i, _dummy_m,
            np.full(MIN_WARMUP_SAMPLES+1, _s, dtype=np.int32))
check("fresh: neural mode ready", _postflop_ready(_fresh))

# Run neural traversal — these samples have real C++ computed range features
run_traversals_batched(_fresh, traversals_per_iter=100, traversing_player=0)
_ns = _fresh.strategy_buffer.sample(16)   # strategy buffer only has real samples
if _ns is not None:
    _sf = _ns[0]   # (n, 77)
    _my  = _sf[:, 17:34]   # my_range_cats
    _opp = _sf[:, 34:51]   # opp_range_cats
    # range cats must sum to 1 per sample
    check("buffer[real] my_range_cats sums to 1",
          np.allclose(_my.sum(axis=1),  1., atol=1e-3))
    check("buffer[real] opp_range_cats sums to 1",
          np.allclose(_opp.sum(axis=1), 1., atol=1e-3))
    # opp_range_cats: different samples should have different distributions
    # (because discards differ per game)
    _opp_var = _opp.var(axis=0).sum()
    check("buffer[real] opp_range_cats: varies across samples",
          _opp_var > 1e-6, f"var={_opp_var:.6f}")
else:
    check("strategy buffer has real samples", False, "empty")

# ── 11f. Current range ≠ fast_score heuristic (proves heuristic removed) ──────
import math as _math

def _fast_score_py(c0, c1, board3):
    """Python mirror of cpp/heuristic/discard.h::fast_score."""
    r0, r1 = c0 % 9, c1 % 9
    s0, s1 = c0 // 9, c1 // 9
    sc = 0.
    if r0 == r1: sc += 10.
    sc += max(r0, r1) * 0.5
    if s0 == s1: sc += 3.
    for b in board3:
        if b < 0: continue
        br, bs = b % 9, b // 9
        if br == r0 or br == r1: sc += 5.
        if abs(br-r0) <= 1 or abs(br-r1) <= 1: sc += 1.
        if bs == s0 and s0 == s1: sc += 2.
    return sc

_KP = [(i,j) for i in range(5) for j in range(i+1,5)]

def _range_heuristic(dead_cards, disc3, board3, temp=0.05):
    """OLD fast_score-based range update (for heuristic comparison only)."""
    dead = set(c for c in dead_cards if c >= 0)
    disc_set = set(c for c in disc3 if c >= 0)
    b3 = (list(board3) + [-1,-1,-1])[:3]
    probs = np.array([0. if (a in dead or b in dead) else 1.
                      for a, b in ALL_PAIRS_TEST], dtype=np.float64)
    s = probs.sum(); probs /= s
    for i, (a, b) in enumerate(ALL_PAIRS_TEST):
        if probs[i] <= 1e-12: continue
        if a in disc_set or b in disc_set: probs[i] = 0.; continue
        hand5 = [a, b] + [c for c in disc3 if c >= 0]
        if len(hand5) < 5: continue
        scores = [_fast_score_py(hand5[ki], hand5[kj], b3) for ki,kj in _KP]
        mx = max(scores)
        ws = [_math.exp((sc - mx)/temp) for sc in scores]
        sw = sum(ws)
        sa, sb = min(a,b), max(a,b)
        kw = 0.
        for idx,(ki,kj) in enumerate(_KP):
            ca,cb = sorted([hand5[ki],hand5[kj]])
            if (ca,cb)==(sa,sb): kw = ws[idx]/sw; break
        probs[i] *= kw
    s = probs.sum()
    return (probs/s).astype(np.float32) if s > 0 else probs.astype(np.float32)

_disc3 = [1, 2, 10]
_board3 = [5, 6, 7]
_dead  = [0, 9] + _board3          # my_hand + community

# Current (card removal only)
rf_cur = RangeFinder()
rf_cur.init(dead_cards=_dead)
rf_cur.update_discard(_disc3, _board3)
_probs_cur = rf_cur.get_range_array()

# Old fast_score heuristic
_probs_heuristic = _range_heuristic(_dead, _disc3, _board3)

# Expected: uniform over (not dead, not disc) pairs
_all_dead_set = set(_dead) | set(_disc3)
_valid_idx = [i for i,(a,b) in enumerate(ALL_PAIRS_TEST)
              if a not in _all_dead_set and b not in _all_dead_set]
_n_valid = len(_valid_idx)
_probs_expected = np.zeros(N_HANDS, dtype=np.float32)
for _vi in _valid_idx: _probs_expected[_vi] = 1./_n_valid

check("current ≠ fast_score heuristic (heuristic was removed)",
      not np.allclose(_probs_cur, _probs_heuristic, atol=1e-3),
      f"max_diff={np.abs(_probs_cur-_probs_heuristic).max():.5f}")

check("current == uniform over valid pairs (card removal only)",
      np.allclose(_probs_cur, _probs_expected, atol=1e-5),
      f"max_diff={np.abs(_probs_cur-_probs_expected).max():.7f}")

check("fast_score range != uniform (sanity: heuristic would be different)",
      not np.allclose(_probs_heuristic, _probs_expected, atol=1e-3),
      f"max_diff={np.abs(_probs_heuristic-_probs_expected).max():.5f}")

# ── 11g. Verify cfr/features.h include chain has no fast_score in range path ──
_feat_h = open(os.path.join(os.path.dirname(__file__), "cpp", "cfr", "features.h")).read()
check("cfr/features.h does NOT include heuristic/discard.h",
      "heuristic/discard" not in _feat_h,
      "found heuristic/discard include in cfr/features.h")
check("cfr/features.h does NOT reference fast_score",
      "fast_score" not in _feat_h,
      "found fast_score reference in cfr/features.h")

_rfcpp = open(os.path.join(os.path.dirname(__file__), "cpp", "range", "rangefinder.cpp")).read()
check("range/rangefinder.cpp does NOT include heuristic/discard.h",
      "heuristic/discard" not in _rfcpp)
check("range/rangefinder.cpp does NOT call fast_score",
      "fast_score(" not in _rfcpp,
      "found fast_score call in rangefinder.cpp")

# ── 11h. Remaining known heuristic: action_probs_heuristic in range_tracker.py ─
_rt = open(os.path.join(os.path.dirname(__file__), "postflop_cfr", "range_tracker.py")).read()
check("range_tracker.py uses sigmoid heuristic for BETTING (not discard)",
      "action_probs_heuristic" in _rt,
      "missing expected sigmoid heuristic for betting range")
check("range_tracker.py does NOT use fast_score for range",
      "fast_score" not in _rt,
      "found fast_score in range_tracker.py (heuristic leak!)")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
if errors:
    print(f"  \033[31m{len(errors)} FAILED:\033[0m")
    for e in errors:
        print(f"    ✘ {e}")
    sys.exit(1)
else:
    print(f"  \033[32mAll checks passed — ready to train.\033[0m")
print(f"{'═'*60}\n")
