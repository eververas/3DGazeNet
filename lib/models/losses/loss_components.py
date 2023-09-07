import torch
import numpy as np


# calculate 3D vectors from phi-theta angels
def pitchyaws_to_vector_torch(pitchyaws):
    n = pitchyaws.shape[0]
    sin = torch.sin(pitchyaws)
    cos = torch.cos(pitchyaws)
    out = torch.empty((n, 3), dtype=pitchyaws.dtype, device=pitchyaws.device)
    out[:, 0] = torch.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = torch.multiply(cos[:, 0], cos[:, 1])
    return out

# calculate 3D phi-theta angels from vectors
def vector_to_pitchyaws_torch(vectors):
    n = vectors.shape[0]
    out = torch.empty((n, 2))
    vectors = torch.divide(vectors, torch.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = torch.arcsin(vectors[:, 1])  # theta
    out[:, 1] = torch.arctan(vectors[:, 0] / vectors[:, 2])  # phi
    return out

def eyeballs_to_angles(eyeballs, iris_idxs481, normalized_vector_pred=0):
    normalized_vectors_lefteye = points_to_vector(eyeballs[0] * (-1), iris_idxs481)
    normalized_vectors_righteye = points_to_vector(eyeballs[1] * (-1), iris_idxs481)
    vecs = normalized_vectors_lefteye + normalized_vectors_righteye + normalized_vector_pred
    pred_vecs = vecs / torch.norm(vecs, dim=1, keepdim=True)
    pred_angles = vector_to_pitchyaws_torch(pred_vecs)
    return pred_angles

# calculate normalized gaze 3D vectors from eyeballs
def points_to_vector(points, iris_lms_idx):
    back = points[:, np.arange(32)].mean(axis=1, keepdim=True) # (B, 1, 3)
    front = points[:, iris_lms_idx].mean(axis=1, keepdim=True) # (B, 1, 3)
    vec = front - back
    vec = vec / torch.norm(vec, dim=2, keepdim=True)  # (B, 1, 3)
    return torch.squeeze(vec)

# arccos loss from normalized 3D vectors
def vector_acos_loss(vec_a, vec_b):
    loss = torch.acos(torch.clamp(torch.sum(vec_a * vec_b, dim=1), -1. + 10e-7, 1 - 10e-7))
    return loss

# reconstruction loss
def l1_recon_loss(w_vertex_reg, verts_pred, verts_trg):
    batch_size = verts_pred.shape[0]
    diff_x = abs(verts_pred[:, :, 0] - verts_trg[:, :, 0]).T
    diff_y = abs(verts_pred[:, :, 1] - verts_trg[:, :, 1]).T
    diff_z = abs(verts_pred[:, :, 2] - verts_trg[:, :, 2]).T
    l_x = diff_x.sum() / batch_size
    l_y = diff_y.sum() / batch_size
    l_z = diff_z.sum() / batch_size
    loss_l1 = w_vertex_reg * (l_x + l_y + l_z)

    return loss_l1, w_vertex_reg * l_x, w_vertex_reg * l_y, w_vertex_reg * l_z

# edge lenght loss
def edge_length_loss(F, w_edge_length, verts_pred, verts_trg):
    batch_size = verts_pred.shape[0]
    # Regularization based on edge lengths.
    # Define num_triangles*3 links.
    V0 = torch.cat((F[:, 0], F[:, 0], F[:, 1]))
    V1 = torch.cat((F[:, 1], F[:, 2], F[:, 2]))

    dif = verts_pred[:, V0] - verts_pred[:, V1]
    x_dif, y_dif, z_dif = dif[:, :, 0], dif[:, :, 1], dif[:, :, 2]
    gt_dif = verts_trg[:, V0] - verts_trg[:, V1]
    x_gt_dif, y_gt_dif, z_gt_dif = gt_dif[:, :, 0], gt_dif[:, :, 1], gt_dif[:, :, 2]
    eucl_dist = lambda x, y, z: torch.sqrt((x ** 2) + (y ** 2) + (z ** 2))
    edge_length_diff = eucl_dist(x_dif, y_dif, z_dif) - eucl_dist(x_gt_dif, y_gt_dif, z_gt_dif)

    edge_length_diff = edge_length_diff.T
    edge_length_loss = w_edge_length * abs(edge_length_diff).sum()
    edge_length_loss = edge_length_loss / batch_size

    return edge_length_loss

# gaze loss from source and target 3D vectors
def gaze_loss_from_vectors(w_gaze_acos, vec, vec_trg):
    batch_size = vec.shape[0]
    vec = vec / torch.norm(vec, dim=1, keepdim=True)
    vec_trg = vec_trg / torch.norm(vec_trg, dim=1, keepdim=True)
    loss_arr = vector_acos_loss(vec, vec_trg)
    loss = loss_arr.sum() / batch_size
    loss = w_gaze_acos * (loss / np.pi) * 180.
    return loss, loss_arr * (180./np.pi)

# gaze loss from source and target pitchyaws
def gaze_loss_from_pitchyaws(w_gaze_acos, pitchyaws_pred, pitchyaws_trg):
    batch_size = pitchyaws_pred.shape[0]
    diff_pitch = abs(pitchyaws_pred[:, 0] - pitchyaws_trg[:, 0]).T
    diff_yaw = abs(pitchyaws_pred[:, 1] - pitchyaws_trg[:, 1]).T
    L_pitch = diff_pitch.sum() / batch_size
    L_yaw = diff_yaw.sum() / batch_size
    loss = w_gaze_acos * (L_pitch + L_yaw)
    return loss
    
# method to unpack input and output
def unbox_output_eyes(output, meta, img_size):
    device = output[0].device
    # target verts
    verts_trg_l = meta[0]['verts'].to(device).float() / (img_size / 2)
    verts_trg_l[:, :, [0, 1]] -= 1
    verts_trg_r = meta[1]['verts'].to(device).float() / (img_size / 2)
    verts_trg_r[:, :, [0, 1]] -= 1 
    # predicted verts
    verts_pred = output[0]
    verts_pred_l = verts_pred[:, :481]
    verts_pred_r = verts_pred[:, 481:]
    # predicted gaze
    vec_pred = output[-1]
    gaze_pred = {'pitchyaws': vector_to_pitchyaws_torch(vec_pred).to(output[0].device),
                 'vector': vec_pred}

    return verts_pred_l, verts_pred_r, verts_trg_l, verts_trg_r, gaze_pred

# method to unpack input and output
def unbox_output_gaze(output, meta):
    gaze_pred = output[0]

    # target gaze of left right eyes and face
    gaze_trg_l = {'pitchyaws': meta[0]['gaze']['pitchyaws'].to(output[0].device).float(),
                  'vector': meta[0]['gaze']['vector'].to(output[0].device).float()}
    gaze_trg_r = {'pitchyaws': meta[1]['gaze']['pitchyaws'].to(output[0].device).float(),
                  'vector': meta[1]['gaze']['vector'].to(output[0].device).float()}
    gaze_trg_f = {'pitchyaws': meta[2]['gaze']['pitchyaws'].to(output[0].device).float(),
                  'vector': meta[2]['gaze']['vector'].to(output[0].device).float()}
    gaze_pred = {'pitchyaws': vector_to_pitchyaws_torch(gaze_pred).to(output[0].device),
                 'vector': gaze_pred}

    return gaze_pred, gaze_trg_l, gaze_trg_r, gaze_trg_f


# method to unpack input and output
def unbox_output_face(output, meta, img_size):
    points_pred_face = output[1]

    # face
    target_x_f = meta[2]['vert_x'].to(output[0].device).float() / (img_size / 2) - 1  # -> [-1, 1]
    target_y_f = meta[2]['vert_y'].to(output[0].device).float() / (img_size / 2) - 1  # -> [-1, 1]
    target_z_f = meta[2]['vert_z'].to(output[0].device).float() / (img_size / 2)
    targets_f = (target_x_f, target_y_f, target_z_f)

    return points_pred_face, targets_f

