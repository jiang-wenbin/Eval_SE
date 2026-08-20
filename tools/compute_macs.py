#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch
from thop import profile
from thop.profile import register_hooks

from espnet2.tasks.enh import EnhancementTask
from espnet2.tasks.enh_tse import TargetSpeakerExtractionTask
from espnet2.torch_utils.model_summary import model_summary
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool


def count_mha(m, x, y):
    """torch.nn.MultiheadAttention

    Args:
        m: module
        x: input
        y: output
    """
    q, k, v = x[:3]
    # Input projection before attention
    total_proj = q.numel() * m.embed_dim + k.numel() * m.kdim + v.numel() * m.vdim

    if m.batch_first:
        # (B, L, N) -> (L, B, N)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
    L_q, B, N_q = q.size()
    L_k, B, N_k = k.size()
    L_v, B, N_v = v.size()
    N_q, N_k, N_v = m.embed_dim, m.kdim, m.vdim
    assert N_k == N_q, (N_k, N_q)
    assert L_k == L_v, (L_k, L_v)

    # Attention: (Q @ K^T) / sqrt(d_k) @ V
    total_att = B * L_q * L_k * N_k + B * N_v * L_q * L_v

    # Output projection
    total_proj_out = B * L_q * m.embed_dim**2

    total_ops = total_proj + total_att + total_proj_out
    m.total_ops += torch.DoubleTensor([int(total_ops)])


register_hooks[torch.nn.MultiheadAttention] = count_mha


def compute_macs(model, fs=8000, is_tse=False):
    dur = 10
    dur_aux = 4

    device = next(model.parameters()).device
    input = torch.rand((1, fs * dur)).to(device)
    lengths = torch.LongTensor([fs * dur]).to(device)

    macs_enc, _ = profile(model=model.encoder, inputs=(input, lengths))
    feature, flen = model.encoder(input, lengths)

    if is_tse:
        # feature_aux = torch.rand((1, model.extractor.spk_embed_dim)).to(device)
        # flen_aux = torch.LongTensor([1]).to(device)

        macs_enc = macs_enc * 2
        aux = torch.rand((1, fs * dur_aux)).to(device)
        feature_aux, flen_aux = model.encoder(aux, lengths)
        macs_sep, _, ret_dict = profile(
            model=model.extractor,
            inputs=(feature, flen, feature_aux, flen_aux),
            ret_layer_info=True,
        )
    else:
        macs_sep, _, ret_dict = profile(
            model=model.separator, inputs=(feature, flen), ret_layer_info=True
        )

    def divide_by(dic, val):
        for k in dic:
            m_ops, m_params, next_dict = dic[k]
            if len(next_dict) > 0:
                divide_by(next_dict, val)
            dic[k] = (m_ops / val, m_params, next_dict)
    divide_by(ret_dict, dur)

    macs_dec, _ = profile(model=model.decoder, inputs=(feature, lengths))

    ret_dict = dict(sorted(ret_dict.items(), key=lambda x: x[1][0], reverse=True))
    return macs_enc / dur, macs_sep / dur, macs_dec / dur, ret_dict


def get_meta(config, fs, is_tse=False):
    # model_file = f'{exp_dir}/{model}'
    max_depth = torch.inf

    if is_tse:
        enh_model, enh_train_args = TargetSpeakerExtractionTask.build_model_from_file(
            config, None
        )
    else:
        enh_model, enh_train_args = EnhancementTask.build_model_from_file(config, None)
    enh_model.to(dtype=getattr(torch, "float32")).eval()

    message = model_summary(enh_model)
    *macs, ret_dict = compute_macs(model=enh_model, fs=fs, is_tse=is_tse)
    macs = [m / 1024 / 1024 / 1024 for m in macs]
    message += f"\n   Enc Macs: {macs[0]} G/s\n"
    message += f"   Sep Macs: {macs[1]} G/s\n"
    message += f"   Dec Macs: {macs[2]} G/s\n"
    message += f"   Tol Macs: {sum(macs)} G/s\n"

    print(message)


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Frontend inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--enh_config", type=str, help="Model config file")
    group.add_argument("--fs", type=int, help="samplerate", default=8000)
    group.add_argument("--is_tse", type=str2bool, help="Whether to use TSE model")

    return parser


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    print(args)
    get_meta(config=args.enh_config, fs=args.fs, is_tse=args.is_tse)


if __name__ == "__main__":
    main()
