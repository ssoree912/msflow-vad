import numpy as np
import torch
import torch.nn.functional as F

def post_process(c, size_list, outputs_list):
    print('Multi-scale sizes:', size_list)
    # concatenate per level (all on CPU)
    outputs_cat = [torch.cat(outputs, 0).cpu() for outputs in outputs_list]
    total = outputs_cat[0].shape[0]
    chunk_size = max(1, min(256, total))  # cap chunk to avoid huge memory

    anomaly_scores = []
    anomaly_maps_add = []
    anomaly_maps_mul = []

    top_k = int(c.input_size[0] * c.input_size[1] * c.top_k)

    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        # upsample and accumulate for this chunk
        logp_maps_chunk = []
        prop_maps_chunk = []
        for out in outputs_cat:
            chunk = out[start:end]
            logp = F.interpolate(chunk.unsqueeze(1),
                                 size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)
            logp_maps_chunk.append(logp)
            output_norm = chunk - chunk.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
            prob_map = torch.exp(output_norm)
            prop = F.interpolate(prob_map.unsqueeze(1),
                                 size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)
            prop_maps_chunk.append(prop)

        # fusion (mul)
        logp_map = sum(logp_maps_chunk)
        logp_map -= logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        prop_map_mul = torch.exp(logp_map)
        anomaly_mul = prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prop_map_mul
        scores = anomaly_mul.reshape(anomaly_mul.shape[0], -1).topk(top_k, dim=-1)[0]
        anomaly_scores.append(np.mean(scores.detach().cpu().numpy(), axis=1))
        anomaly_maps_mul.append(anomaly_mul.detach().cpu().numpy())

        # fusion (add)
        prop_map_add = sum(prop_maps_chunk).detach().cpu().numpy()
        anomaly_map_add = prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add
        anomaly_maps_add.append(anomaly_map_add)

    anomaly_score = np.concatenate(anomaly_scores, axis=0)
    anomaly_score_map_add = np.concatenate(anomaly_maps_add, axis=0)
    anomaly_score_map_mul = np.concatenate(anomaly_maps_mul, axis=0)

    return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul
