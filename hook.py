"""The three SAE-coefficient interventions on a single feature, as a resid hook.

  +  (amplify):  ã_i = a_i + s * a_max          (paper Eq. 6)
  0  (ablate):   ã_i = 0        (c = 0)         in-distribution off-state
  -  (negative): ã_i = -c * m_i (c > 0)         OOD past-off probe

Single-feature delta steering: we move the residual only along feature i's
decoder direction, leaving all other features AND the SAE reconstruction error
untouched, so the no-intervention baseline is exactly the original model.

  resid' = resid + (ã_i - a_i) * W_dec[:, i]
"""
import torch


def make_steer_hook(sae, feature_idx: int, mode: str, *,
                    s: float = 10.0, c: float = 0.0, m_i: float = 1.0,
                    a_max_char: float = None):
    """Return a transformer_lens forward hook fn for blocks.L.hook_resid_post.

    mode: 'amplify' | 'coeff' | 'baseline'
      amplify -> ã_i = a_i + s*a_max_char
      coeff   -> ã_i = -c*m_i   (c=0 ablation, c>0 negative)
      baseline-> no change (sanity)

    a_max_char is the feature's CHARACTERISTIC max activation (Eq.6's a_max), a
    fixed scalar from data — NOT the current prompt's activation. This is required
    for the neutral-prompt "appears-from-nothing" demo, where the feature is OFF
    (current max = 0) so a current-prompt a_max would amplify by zero.
    """
    w_dec_i = sae.W_dec[feature_idx]            # [d_model]

    def hook_fn(resid, hook):
        # resid: [batch, pos, d_model]
        a = sae.encode(resid)                   # [batch, pos, d_sae]
        a_i = a[..., feature_idx]               # [batch, pos]

        if mode == "baseline":
            return resid
        elif mode == "amplify":
            ref = a_max_char if a_max_char is not None else float(a_i.amax())
            a_i_new = a_i + s * ref
        elif mode == "coeff":
            a_i_new = torch.full_like(a_i, -c * m_i)
        else:
            raise ValueError(f"unknown mode {mode}")

        delta = (a_i_new - a_i).unsqueeze(-1)  # [batch, pos, 1]
        return resid + delta * w_dec_i

    return hook_fn


def make_random_hook(sae, feature_idx: int, *, c: float, m_i: float,
                     seed: int = 0):
    """Magnitude-matched random-direction control for the negative push (guard #2).

    The negative intervention writes a delta of L2 norm
        ||(ã_i - a_i) * W_dec[:,i]|| = |(-c*m_i) - a_i| * ||W_dec[:,i]||
    along the feature's decoder direction. This control writes the SAME per-position
    L2 magnitude along a FIXED random unit direction in residual space instead.

    Comparing neg(c) vs rand(c) disentangles "the negative regime is special" from
    "a large-magnitude perturbation is a large-magnitude perturbation": only if neg
    suppresses the concept MORE than rand, at no worse coherence cost, does going
    negative buy something beyond raw perturbation size.

    The direction is seeded by the feature so it is fixed across the c-sweep (the
    magnitude scales with c; the direction does not).
    """
    w_dec_i = sae.W_dec[feature_idx]            # [d_model]
    w_norm = float(w_dec_i.detach().norm())
    g = torch.Generator(device="cpu").manual_seed(seed + int(feature_idx))
    r = torch.randn(w_dec_i.shape[0], generator=g)
    r_hat = (r / r.norm()).to(device=w_dec_i.device, dtype=w_dec_i.dtype)

    def hook_fn(resid, hook):
        a = sae.encode(resid)
        a_i = a[..., feature_idx]                       # [batch, pos]
        a_i_new = -c * m_i
        mag = (a_i_new - a_i).abs() * w_norm            # [batch, pos], L2 of the neg delta
        return resid + mag.unsqueeze(-1) * r_hat

    return hook_fn


@torch.no_grad()
def feature_activation(model, sae, hook_name: str, feature_idx: int, tokens):
    """Return feature i's activation per position for given tokens: [batch, pos]."""
    _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    resid = cache[hook_name]
    a = sae.encode(resid)
    return a[..., feature_idx]
