import torch as th

def box_distance(box_a, box_b):
    center_a, offset_a = box_a
    center_b, offset_b = box_b
    dist = th.abs(center_a - center_b) - offset_a - offset_b
    return dist

def box_intersection(box_a, box_b):
    center_a, offset_a = box_a
    center_b, offset_b = box_b
    
    lower = th.maximum(center_a - offset_a, center_b - offset_b)
    upper = th.minimum(center_a + offset_a, center_b + offset_b)
    centers = (lower + upper) / 2
    offsets = th.abs(upper - lower) / 2
    intersection = (centers, offsets)
    return intersection, lower, upper

def inclusion_loss(box_a, box_b, gamma):
    dist_a_b = box_distance(box_a, box_b)
    _, offset_a = box_a
    loss = th.linalg.norm(th.relu(dist_a_b + 2*offset_a - gamma), dim=1)
    return loss

def gci0_bot_loss(data, class_offset):
    offset_c = class_offset(data[:, 0])
    loss = th.linalg.norm(offset_c, dim=1)

    return loss

def gci0_loss(data, sub_center, sub_offset, sup_center, sup_offset, gamma,
              bump=None, bump_weight=0.1, neg=False):
    center_c = sub_center(data[:, 0])
    offset_c = sub_offset(data[:, 0])
    center_d = sup_center(data[:, 1])
    offset_d = sup_offset(data[:, 1])
    box_c = (center_c, offset_c)
    box_d = (center_d, offset_d)
    loss = inclusion_loss(box_c, box_d, gamma)
    if bump:
        loss = loss + bump_weight * th.norm(bump(data[:,0]) - bump(data[:,1]))
    return loss

def gci1_loss(data, class_center, class_offset, gamma, neg=False):
    center_c = class_center(data[:, 0])
    center_d = class_center(data[:, 1])
    center_e = class_center(data[:, 2])
    offset_c = class_offset(data[:, 0])
    offset_d = class_offset(data[:, 1])
    offset_e = class_offset(data[:, 2])

    box_c = (center_c, offset_c)
    box_d = (center_d, offset_d)
    box_e = (center_e, offset_e)

    intersection, lower, upper = box_intersection(box_c, box_d)
    box_incl_loss = inclusion_loss(intersection, box_e, gamma)

    additional_loss = th.linalg.norm(th.relu(lower - upper), dim=1)
    loss = box_incl_loss + additional_loss
    return loss


def gci1_bot_loss(data, c_center, c_offset, d_center, d_offset, gamma,
                  neg=False):
    center_c = c_center(data[:, 0])
    center_d = d_center(data[:, 1])

    offset_c = c_offset(data[:, 0])
    offset_d = d_offset(data[:, 1])

    box_c = (center_c, offset_c)
    box_d = (center_d, offset_d)

    box_dist = box_distance(box_c, box_d)
    loss = th.linalg.norm(th.relu(-box_dist - gamma), dim=1)
    return loss

def gci2_loss(data, class_center, class_offset, head_center, head_offset,
              tail_center, tail_offset, bump, gamma, delta, reg_factor,
              offset_reg_factor, neg=False, training=True):
    if neg:
        return gci2_loss_neg(data, class_center, class_offset, head_center,
                             head_offset, tail_center, tail_offset, bump,
                             gamma, delta, reg_factor, offset_reg_factor,
                             training=training)
    else:
        center_c = class_center(data[:, 0])
        offset_c = class_offset(data[:, 0])

        center_head = head_center(data[:, 1])
        offset_head = head_offset(data[:, 1])

        center_tail = tail_center(data[:, 1])
        offset_tail = tail_offset(data[:, 1])

        center_d = class_center(data[:, 2])
        offset_d = class_offset(data[:, 2])

        bump_c = bump(data[:, 0])
        bump_d = bump(data[:, 2])
        
        box_c = (center_c, offset_c)
        box_head = (center_head, offset_head)
        box_tail = (center_tail, offset_tail)
        box_d = (center_d, offset_d)

        bumped_c = (center_c + bump_d, offset_c)
        bumped_d = (center_d + bump_c, offset_d)

        inclussion_1 = inclusion_loss(bumped_c, box_head, gamma)
        inclussion_2 = inclusion_loss(bumped_d, box_tail, gamma)

        loss = inclussion_1 + inclussion_2
        reg_loss = reg_factor * th.linalg.norm(bump.weight, dim=1).sum() \
            if training else 0
        offset_reg = offset_reg_factor * (th.linalg.norm(offset_c, dim=1) +
                                          th.linalg.norm(offset_d, dim=1) +
                                          th.linalg.norm(offset_head, dim=1) +
                                          th.linalg.norm(offset_tail, dim=1)) \
            if training else 0
        
        return reg_loss + loss/2 + offset_reg


def gci2_loss_neg(data, class_center, class_offset, head_center, head_offset, tail_center, tail_offset, bump, gamma, delta, reg_factor, offset_reg_factor, training=True):

    def minimal_distance(box_a, box_b, gamma):
        dist = box_distance(box_a, box_b)
        min_dist = th.linalg.norm(th.relu(dist + gamma), dim=1)
        return min_dist
    
    center_c = class_center(data[:, 0])
    offset_c = class_offset(data[:, 0])
    center_head = head_center(data[:, 1])
    offset_head = head_offset(data[:, 1])
    center_tail = tail_center(data[:, 1])
    offset_tail = tail_offset(data[:, 1])
    center_d = class_center(data[:, 2])
    offset_d = class_offset(data[:, 2])
    bump_c = bump(data[:, 0])
    bump_d = bump(data[:, 2])

    box_c = (center_c, offset_c)
    box_head = (center_head, offset_head)
    box_tail = (center_tail, offset_tail)
    box_d = (center_d, offset_d)

    bumped_c = (center_c + bump_d, offset_c)
    bumped_d = (center_d + bump_c, offset_d)
    
    fist_part = (delta - minimal_distance(bumped_c, box_head, gamma))**2
    second_part = (delta - minimal_distance(bumped_d, box_tail, gamma))**2

    loss = fist_part + second_part
    reg_loss = reg_factor * th.linalg.norm(bump.weight, dim=1).sum() \
        if training else 0
    
    offset_reg = offset_reg_factor * (th.linalg.norm(offset_c, dim=1) +
                                      th.linalg.norm(offset_d, dim=1) +
                                      th.linalg.norm(offset_head, dim=1) +
                                      th.linalg.norm(offset_tail, dim=1)) \
            if training else 0
    
    return loss + reg_loss + offset_reg


def gci3_loss(data, class_center, class_offset, head_center, head_offset, tail_center, tail_offset, bump, gamma, reg_factor, neg=False, training=True):

    center_d = class_center(data[:, 2])
    offset_d = th.abs(class_offset(data[:, 2]))

    
    center_head = head_center(data[:, 0])
    offset_head = th.abs(head_offset(data[:, 0]))

    bump_c = bump(data[:, 1])

    bumped_head = (center_head - bump_c, offset_head)
    box_d = (center_d, offset_d)
    loss = inclusion_loss(bumped_head, box_d, gamma)
    reg_loss = reg_factor * th.linalg.norm(bump.weight, dim=1).sum() \
        if training else 0
    return loss + reg_loss


def gci3_bot_loss(data, head_offset):

    offset_head = th.abs(head_offset(data[:, 0]))
    loss = th.linalg.norm(offset_head, dim=1)
    return loss
