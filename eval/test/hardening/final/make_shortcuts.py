"""Build the new shortcut artifacts replayed in the final hardening pass.

Every case uses only what a future method can see: public training data, the
public train evaluator's validity rule, and the method's own outputs. Outputs
go to eval/test/hardening/final/raw_predictions/<case>/predictions/.
"""
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'raw_predictions'
N, T, K, V, S = 16384, 24, 8, 32, 16
speakers = torch.load('data/synthetic_codec/test_conditions.pt', weights_only=True)['speaker'].long()
train = torch.load('data/synthetic_codec/train.pt', weights_only=True)
tok, spk = train['tokens'].long(), train['speaker'].long()


def save(case, tensors):
    folder = OUT / case / 'predictions'
    if all((folder / f'generator_{stack}x.pt').is_file() for stack in tensors):
        return  # already built (outputs are committed evidence; do not rewrite)
    folder.mkdir(parents=True, exist_ok=True)
    for stack, tokens in tensors.items():
        torch.save({'pred_tokens': tokens.to(torch.uint8), 'config': {'stack_factor': stack}, 'split': 'test'},
                   folder / f'generator_{stack}x.pt')


def reference(stack, path='checkpoints/predictions'):
    return torch.load(f'{path}/generator_{stack}x.pt', weights_only=True)['pred_tokens'].long()


# 1. Analytic sampler inferred from training data with the public validity rule
#    (residual = c_k - 5 c_{k-1} - 7 spk - 11 k - pos mod 32): empirical coarse
#    steps per speaker, empirical style-bit rates per (speaker, codebook).
def analytic(seed):
    g = torch.Generator().manual_seed(seed)
    steps = (tok[:, 1:, 0] - tok[:, :-1, 0]) % V
    k = torch.arange(1, K).view(1, 1, -1)
    pos = torch.arange(T).view(1, T, 1)
    resid = (tok[..., 1:] - 5 * tok[..., :-1] - 7 * spk.view(-1, 1, 1) - 11 * k - pos) % V
    style_rate = torch.stack([resid[spk == s, 0].float().mean(0) for s in range(S)])
    step_hist = torch.stack([torch.bincount(steps[spk == s].flatten(), minlength=V).float() for s in range(S)])
    out = torch.zeros(N, T, K, dtype=torch.long)
    out[:, 0, 0] = torch.randint(V, (N,), generator=g)
    for t in range(1, T):
        out[:, t, 0] = (out[:, t - 1, 0] + torch.multinomial(step_hist[speakers], 1, generator=g).squeeze(1)) % V
    style = (torch.rand(N, K - 1, generator=g) < style_rate[speakers]).long()
    p = torch.arange(T).view(1, T)
    for kk in range(1, K):
        out[..., kk] = (5 * out[..., kk - 1] + 7 * speakers.view(-1, 1) + 11 * kk + p + style[:, kk - 1].view(-1, 1)) % V
    return out


save('shortcut_analytic_sampler', {2: analytic(11), 4: analytic(12)})

# 2. The 2x samples submitted again as the 4x row (false 6-step declaration).
save('shortcut_reuse_2x_as_4x', {2: reference(2), 4: reference(2)})

# 3. A second, independent sample of the 2x model declared as the 4x row.
second = 'eval/test/hardening/raw_predictions/reference_sample1/predictions'
save('shortcut_resample_2x_as_4x', {2: reference(2), 4: reference(2, second)})

# 4./5. Copies of training utterances (same speaker) that evade exact
#       complete-utterance matching.
by_speaker = [torch.nonzero(spk == s).flatten() for s in range(S)]


def pick(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([by_speaker[s][torch.randint(len(by_speaker[s]), (1,), generator=g)].squeeze() for s in speakers.tolist()])


def splice(seed):
    a, b = tok[pick(seed)], tok[pick(seed + 1)]
    return torch.cat([a[:, :12], b[:, 12:]], dim=1)


def edit(seed):
    out = tok[pick(seed)].clone()
    # Flip the style bit of the last residual codebook in the last frame:
    # the within-frame rule still holds, only the tuple differs from the copy.
    base = (5 * out[:, -1, -2] + 7 * speakers + 11 * (K - 1) + (T - 1)) % V
    out[:, -1, -1] = torch.where(out[:, -1, -1] == base, (base + 1) % V, base)
    return out


save('shortcut_splice_halves', {2: splice(21), 4: splice(23)})
save('shortcut_copy_edit', {2: edit(31), 4: edit(33)})


# 6. False 4x declaration that evades a scored-rows-only reuse check: the 4x
#    file's scored rows hold the 2x model's outputs for *unscored* rows of the
#    same speaker. The public train evaluator reveals which rows are unscored.
def train_rows_as_4x():
    order = torch.randperm(N, generator=torch.Generator().manual_seed(250919592))
    train_rows, test_rows = order[:N // 2].sort().values, order[N // 2:].sort().values
    two, four = reference(2), reference(4).clone()
    for s in range(S):
        dst, src = test_rows[speakers[test_rows] == s], train_rows[speakers[train_rows] == s]
        four[dst] = two[src[torch.arange(len(dst)) % len(src)]]
    return four


save('shortcut_train_rows_2x_as_4x', {2: reference(2), 4: train_rows_as_4x()})
# 7. REDTEAM constant output (all zeros) at both legal operating points.
save('shortcut_constant', {2: torch.zeros(N, T, K, dtype=torch.long), 4: torch.zeros(N, T, K, dtype=torch.long)})
print('built', sorted(p.name for p in OUT.iterdir()))
