"""Hidden synthetic speech-codec process (held only by eval/test/).

A random nonlinear state-space model stands in for speech, and a fixed
finite-scalar quantizer (FSQ, as in the paper's NanoCodec) turns its latent
trajectory into codec tokens:

* each utterance has a speaker (16) and a persistent utterance style;
* a hidden phone-like regime (16 states) switches with speaker-, phone- and
  state-dependent durations and transition probabilities;
* a 64-d hidden state evolves nonlinearly, driven by the regime, speaker,
  style and fresh noise every frame;
* a random nonlinear readout gives a 24-d latent, which FSQ quantizes: each of
  the 8 codebooks owns 3 latent dimensions, bounded by tanh and rounded to a
  4 x 4 x 2 grid, so one codebook token is one of 32 cells.

All codebooks of a frame read out the same hidden state, so they are strongly
dependent given the past (what parallel prediction cannot capture); the
regime and hidden state carry structure across frames (what frame stacking
must decode within one primary step). The parameters are drawn from a seed
that exists only in this file; nothing outside eval/test/ can regenerate the
process, the training data, or the hidden test references. Only the
quantizer grid (how a token maps back to a codec latent) is public, as for a
real codec.
"""
import math

import torch

SEQ_LEN, NUM_CODEBOOKS, VOCAB, NUM_SPEAKERS = 24, 8, 32, 16
LEVELS = (4, 4, 2)                      # FSQ levels of one codebook's 3 dimensions
CODEBOOK_DIMS = len(LEVELS)
LATENT_DIM = NUM_CODEBOOKS * CODEBOOK_DIMS
HIDDEN, PHONES, SPEAKER_DIM, STYLE_DIM, READOUT = 64, 16, 16, 8, 96
SIGMA_HIDDEN, SIGMA_LATENT = 0.5, 0.05
PROCESS_SEED = 590_377_119


def _params(seed=PROCESS_SEED):
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, dtype=torch.float64) * scale

    p = {'spk': r(NUM_SPEAKERS, SPEAKER_DIM), 'phone_emb': r(PHONES, HIDDEN)}
    trans = r(PHONES, PHONES, scale=2.0)
    trans.fill_diagonal_(-1e9)          # a switch always moves to another regime
    p['trans'] = trans
    p['trans_spk'] = torch.einsum('se,epq->spq', p['spk'], r(SPEAKER_DIM, PHONES, PHONES, scale=0.5))
    p['trans_h'] = r(PHONES, HIDDEN, scale=1.0 / math.sqrt(HIDDEN))
    p['init'] = r(NUM_SPEAKERS, PHONES)
    p['stay'] = r(PHONES, scale=0.5) + 1.0
    p['stay_spk'] = r(NUM_SPEAKERS, scale=0.5)
    p['stay_h'] = r(HIDDEN, scale=1.0 / math.sqrt(HIDDEN))
    q, _ = torch.linalg.qr(r(HIDDEN, HIDDEN))
    p['A'] = 0.8 * q
    p['B'] = r(HIDDEN, SPEAKER_DIM, scale=0.4)
    p['style'] = r(HIDDEN, STYLE_DIM, scale=0.4)
    p['W1'] = r(READOUT, HIDDEN, scale=1.5 / math.sqrt(HIDDEN))
    p['b1'] = r(READOUT, scale=0.5)
    p['W2'] = r(LATENT_DIM, READOUT, scale=2.0 / math.sqrt(READOUT))
    p['b2'] = r(LATENT_DIM, scale=0.3)
    return p


_P = _params()
_LEVELS = torch.tensor(LEVELS)
_RADIX = torch.tensor([LEVELS[1] * LEVELS[2], LEVELS[2], 1])


def sample_latents(speakers, seed):
    """(b, 24, 24) float64 latent trajectories for the given speakers."""
    g = torch.Generator().manual_seed(seed)
    speakers = speakers.long()
    b = speakers.shape[0]
    style = torch.randn(b, STYLE_DIM, generator=g, dtype=torch.float64)
    const = _P['spk'][speakers] @ _P['B'].T + style @ _P['style'].T
    phone = torch.multinomial(_P['init'][speakers].softmax(-1), 1, generator=g).squeeze(1)
    h = torch.zeros(b, HIDDEN, dtype=torch.float64)
    z = torch.zeros(b, SEQ_LEN, LATENT_DIM, dtype=torch.float64)
    for t in range(SEQ_LEN):
        if t > 0:
            stay_logit = _P['stay'][phone] + _P['stay_spk'][speakers] + h @ _P['stay_h']
            stay = torch.rand(b, generator=g, dtype=torch.float64) < torch.sigmoid(stay_logit)
            logits = _P['trans'][phone] + _P['trans_spk'][speakers, phone] + h @ _P['trans_h'].T
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=g).squeeze(1)
            phone = torch.where(stay, phone, nxt)
        noise = torch.randn(b, HIDDEN, generator=g, dtype=torch.float64)
        h = torch.tanh(h @ _P['A'].T + _P['phone_emb'][phone] + const + SIGMA_HIDDEN * noise)
        readout = torch.tanh(h @ _P['W1'].T + _P['b1']) @ _P['W2'].T + _P['b2']
        z[:, t] = readout + SIGMA_LATENT * torch.randn(b, LATENT_DIM, generator=g, dtype=torch.float64)
    return z


def quantize(z):
    """FSQ: per codebook, bound its 3 dims with tanh and round to the 4x4x2 grid."""
    shape = z.shape[:-1]
    unit = (torch.tanh(z.reshape(*shape, NUM_CODEBOOKS, CODEBOOK_DIMS)) + 1) / 2
    digits = torch.minimum((unit * _LEVELS).floor().long(), _LEVELS - 1)
    return (digits * _RADIX).sum(-1)


def sample_tokens(speakers, seed, chunk=8192):
    """(b, 24, 8) long codec tokens drawn from the process (deterministic in seed)."""
    out = []
    for i, start in enumerate(range(0, speakers.shape[0], chunk)):
        out.append(quantize(sample_latents(speakers[start:start + chunk], seed * 1009 + i)))
    return torch.cat(out) if out else torch.zeros(0, SEQ_LEN, NUM_CODEBOOKS, dtype=torch.long)


def make_split(num_sequences, seed):
    speakers = torch.randint(NUM_SPEAKERS, (num_sequences,), generator=torch.Generator().manual_seed(seed + 7919))
    return {'tokens': sample_tokens(speakers, seed), 'speaker': speakers}
