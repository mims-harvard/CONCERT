"""BUG D verification: did f_pert actually train on the pmap multi-slide retrain?

Loads ONLY the checkpoint state_dict (no model build, no data). Checks:
  (a) f_pert core weight L2 moved off std-0.01 init (~7.155 for [num_genes,emb]).
  (b) per-pert CORE response = E @ W.T differentiates: basal-slot near-null, 3 KO distinct.
  (c) fpert_field.ls_head / mlp trained.
A TRAINED f_pert (old 2-KO ref): basal ||5.8||/2 genes vs Tgfbr2 ||57||/1166 genes.
A FROZEN f_pert (27468445): all perts ||~7||, ~1500 genes, pairwise cos ~0 (random proj).
"""
import sys, torch, numpy as np

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    '/n/netscratch/mzitnik_lab/Lab/xiang/CONCERT/outputs/ms_pmap_gp_pp_bayesTrue/model.pt'
sd = torch.load(CKPT, map_location='cpu')
if isinstance(sd, dict) and 'model_state_dict' in sd:
    sd = sd['model_state_dict']
if isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']

# find the relevant keys
def find(sub):
    return [k for k in sd.keys() if sub in k]

print("=== f_pert / fpert_field / perturbation keys ===")
for k in sd.keys():
    if any(s in k for s in ['f_pert', 'fpert_field', 's_encoder.perturbation', 'mu_action']):
        v = sd[k]
        try:
            print(f"  {k:<55} shape={tuple(v.shape)} L2={v.float().norm():.4f}")
        except Exception:
            print(f"  {k:<55} {v}")

# core weight W: the Linear inside f_pert. Try common names.
W = None
for cand in ['f_pert.weight', 'f_pert.core.weight', 'f_pert.linear.weight', 'f_pert.net.weight']:
    if cand in sd:
        W = sd[cand]; wk = cand; break
if W is None:
    fk = find('f_pert')
    print("\n[!] f_pert.weight not found; f_pert keys:", fk)
# embedding E
E = None
for cand in ['lord_encoder.s_encoder.perturbation.weight',
             'lord_encoder.s_encoder.perturbation.embedding.weight']:
    if cand in sd:
        E = sd[cand]; ek = cand; break
if E is None:
    ek_l = find('s_encoder.perturbation')
    print("\n[!] perturbation embedding not found; keys:", ek_l)

INIT = None
if W is not None:
    INIT = (0.01 * np.sqrt(W.numel()))  # std-0.01 init L2 estimate
    print(f"\n(a) core weight {wk}: L2={W.float().norm():.4f}  (std-0.01 init ~ {INIT:.3f})")

if W is not None and E is not None:
    E = E.float(); W = W.float()
    # per-pert core response = E @ W.T  -> (n_pert, num_genes)
    # W shape may be (num_genes, emb) or (emb, num_genes)
    if W.shape[1] == E.shape[1]:
        core = E @ W.t()          # (n_pert, num_genes)
    elif W.shape[0] == E.shape[1]:
        core = E @ W              # (n_pert, num_genes)
    else:
        print(f"[!] shape mismatch E{tuple(E.shape)} W{tuple(W.shape)}"); sys.exit(1)
    core = core.numpy()
    n_pert = core.shape[0]
    print(f"\n(b) per-pert CORE response  (n_pert={n_pert}, genes={core.shape[1]}):")
    norms = np.linalg.norm(core, axis=1)
    active = (np.abs(core) > 0.5).sum(axis=1)
    if n_pert <= 12:
        for i in range(n_pert):
            print(f"  code {i}:  ||core||={norms[i]:8.3f}   |shift|>0.5 genes={active[i]:5d}")
        print("  pairwise cosine:")
        for i in range(n_pert):
            for j in range(i+1, n_pert):
                a, b = core[i], core[j]
                cos = float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))
                print(f"    ({i},{j}) cos={cos:+.3f}")
    else:
        # large vocab (spac ~547): summarize instead of dumping every code
        print(f"  code 0 (__basal__ slot):  ||core||={norms[0]:8.3f}   |shift|>0.5 genes={active[0]:5d}")
        print(f"  KO codes 1..{n_pert-1}:  ||core|| min/median/max = "
              f"{norms[1:].min():.3f} / {np.median(norms[1:]):.3f} / {norms[1:].max():.3f}")
        order = np.argsort(norms[1:])[::-1] + 1
        print("  top-8 KO by ||core||:", [f"code{int(i)}={norms[i]:.1f}" for i in order[:8]])
        print("  bottom-5 KO by ||core||:", [f"code{int(i)}={norms[i]:.2f}" for i in order[-5:]])
        print(f"  basal-vs-KO ratio (median KO / basal) = {np.median(norms[1:])/max(norms[0],1e-9):.2f}x")
    print("\nVERDICT: f_pert TRAINED if one code is near-null (basal) and KO codes have large ||core||"
          " with distinct gene programs; FROZEN if all ~equal ||~7|| & orthogonal (random projection).")
