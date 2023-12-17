"""
Paper: Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Production Federated Learning
Customized PGA attack for several aggregators, searching best gama within [1, max_gama] with gama = gama * (1 + tol) each step
"""

import torch
import numpy as np
import copy
from model import ServerModel
from wquantiles import median as weighted_median
import sys
sys.path.append('../../MEBwO/src')
from meb.ball import MEBwO

def pga_attack_mebra(updates, weights, mean_up, is_corrupted,mebwo_alg="shenmaier", dev_type='sign',max_gama=1e3):
    """
    Paper: Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Production Federated Learning
    differ from pga_attack_mebra() on the search method of lamda, this is according to the pseudocode in the paper
    s
    updates: updates of all clients, both benign and malicious
    weights: weights of all clients, both benign and malicious
    mean_up: the mean update of benign clients
    is_corrupted: indicator of corrupted/malicious clients, with 1 denoting corrupted and 0 denoting benign
    """
    
    if dev_type == 'unit_vec':
        deviation = mean_up / np.linalg.norm(mean_up)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = np.sign(mean_up)

    # lamda = torch.Tensor([lamda]).float().cuda()  # the search range of coeffcient gama in algorithm2
    threshold_diff = 1e-5
    gama = 1
    gama_succ = 0
    max_min_d = 0
    tol = 0.4
    count = 0 # recording the time peroid of search procss
    mal_ups = copy.deepcopy(updates)
    corp_client_rate = sum(is_corrupted)/len(is_corrupted)

    while gama <= max_gama:
        count = count + 1
        mal_update = (mean_up - gama * deviation)
        for i in range(len(updates)):
            if is_corrupted[i]:
                mal_ups[i] = mal_update
        # print("in pgaAtcUtils.py line210: mal_ups[:5]: ",mal_ups[:5])

        ball1 = MEBwO().fit(data=np.array(mal_ups), method=mebwo_alg, eta=(1-corp_client_rate)) # inner
        inner_points = np.asarray([p for p in np.asarray(mal_ups) if ball1.contains(p)])  
        # print("shape of inner points: ", len(inner_points), " shape of weights: ", len(weights))
        agr_up1 = ServerModel.weighted_average_oracle(inner_points, weights)

        ball2 = MEBwO().fit(data=np.array(mal_ups), method=mebwo_alg, eta=(corp_client_rate)) # outer
        outer_points = np.asarray([p for p in np.asarray(mal_ups) if not ball2.contains(p)])
        # print("shape of outer points: ", len(outer_points), " shape of weights: ", len(weights))
        agr_up2 = ServerModel.weighted_average_oracle(outer_points, weights)

        dis1 = np.linalg.norm(agr_up1 - mean_up)
        dis2 = np.linalg.norm(agr_up2 - mean_up)
        min_d = min(dis1, dis2)
        
        if min_d > max_min_d: # maximizing the attak objective
            gama_succ = gama
            max_min_d = min_d

        gama = gama * (1 + tol)

    print("search count: ",count," PGA Attack: final gama_succ: ",gama_succ," norm(max_min_d): ", np.linalg.norm(max_min_d))
    mal_update = (mean_up - gama_succ * deviation)
    return mal_update

def pga_attack_mkrum(updates, weights, mean_up, is_corrupted, fraction_to_discard, dev_type='unit_vec', max_gama=20):
    """
    mean_up: mean gradient of benign workers
    """
    if dev_type == 'unit_vec':
        deviation = mean_up / np.linalg.norm(mean_up)  # unit vector, dir to good dir
    elif dev_type == 'sign':
        deviation = np.sign(mean_up)  # P8: w = sign(benign_mean_up)
        
    gama = 1
    tol = 0.1
    gama_succ = 0
    max_dis = 0
    count = 0 # recording the time peroid of search procss
    mal_ups = copy.deepcopy(updates)

    while gama < max_gama:
        count = count +1
        mal_update = (mean_up - gama * deviation) # P8:DYN-OPT mal-up = benign_mean_up - rw
        for i in range(len(updates)):
            if is_corrupted[i]:
                mal_ups[i] = mal_update

        agg_grads = ServerModel.multikrum_update(mal_ups, weights, fraction_to_discard)
        dis = np.linalg.norm(agg_grads - mean_up)

        if dis > max_dis: 
            gama_succ = gama
            max_dis = dis

        gama = gama * (1 + tol)
   
    print("PGA Attack: search count: ",count," final gama_succ: ",gama_succ," norm(max_dis): ", np.linalg.norm(max_dis))
    mal_update = (mean_up - gama_succ * deviation)
    return mal_update

def pga_attack_trmean(updates, weights, mean_up, is_corrupted, fraction_to_discard, dev_type='sign', max_gama=20):
    """
    all_updates: updates of all clients, both benign and malicious
    weights: weights of all clients, both benign and malicious
    mean_up: the mean update of benign clients
    """
    if dev_type == 'sign':
        deviation = np.sign(mean_up)
    elif dev_type == 'unit_vec':
        deviation = mean_up / np.linalg.norm(mean_up)  # unit vector, dir to good dir

    gama = 1
    tol = 0.1
    max_loss = 0
    gama_succ = 0
    count = 0 # recording the time peroid of search procss
    mal_ups = copy.deepcopy(updates)

    while gama < max_gama:
        count = count + 1
        mal_update = (mean_up - gama * deviation)
        for i in range(len(updates)):
            if is_corrupted[i]:
                mal_ups[i] = mal_update
        # print("mal_ups[:5]: ", mal_ups[:5])

        agg_grads = ServerModel.trimmed_mean_update(mal_ups, weights, fraction_to_discard)
        # print("agg_grads: ", agg_grads)
        loss = np.linalg.norm(agg_grads - mean_up)
        # print("PGA Attack: loss: ", loss)

        if loss > max_loss:
            gama_succ = gama
            # print("one gama_succ: ", gama_succ)
            max_loss = loss

        gama = gama * (1 + tol)
    
    print("PGA Attack: search count: ",count," final gama_succ: ",gama_succ," norm(max_loss): ", np.linalg.norm(max_loss))
    mal_update = (mean_up - gama_succ * deviation)
    return mal_update

def pga_attack_norm(updates, weights, mean_up, is_corrupted, norm_bound, dev_type='sign', max_gama=2000):
    """
    all_updates: updates of all clients, both benign and malicious
    weights: weights of all clients, both benign and malicious
    mean_up: the mean update of benign clients
    """
    if dev_type == 'sign':
        deviation = np.sign(mean_up)
    elif dev_type == 'unit_vec':
        deviation = mean_up / np.linalg.norm(mean_up)  # unit vector, dir to good dir

    gama = 1
    tol = 0.2
    max_loss = 0
    gama_succ = 0
    count = 0 # recording the time peroid of search procss
    mal_ups = copy.deepcopy(updates)

    while gama < max_gama:
        count = count + 1
        mal_update = (mean_up - gama * deviation)

        if count == 1: # scaling to the norm_bound at first
            mal_update = mal_update * norm_bound / np.linalg.norm(mal_update)
            # print("Norm bound/norm(mal_up): ",norm_bound / np.linalg.norm(mal_update))
        
        for i in range(len(updates)):
            if is_corrupted[i]:
                mal_ups[i] = mal_update
        agg_grads = ServerModel.norm_bounded_mean_update(mal_ups, weights, norm_bound)
        loss = np.linalg.norm(agg_grads - mean_up)
        # print("PGA Attack: loss: ", loss)

        if loss > max_loss:
            gama_succ = gama
            # print("one gama_succ: ", gama_succ)
            max_loss = loss

        gama = gama * (1 + tol)
    
    print("PGA Attack: search count: ",count," final gama_succ: ",gama_succ," norm(max_loss): ", np.linalg.norm(max_loss))
    mal_update = (mean_up - gama_succ * deviation)
    return mal_update

def pga_attack_gm(updates, weights, mean_up, is_corrupted, maxiter, dev_type='sign', max_gama=2000):
    """
    mean_up: mean gradient of benign workers
    """
    if dev_type == 'unit_vec':
        deviation = mean_up / np.linalg.norm(mean_up)  # unit vector, dir to good dir
    elif dev_type == 'sign':
        deviation = np.sign(mean_up)  # P8: w = sign(benign_mean_up)
        
    gama = 1
    tol = 0.2
    gama_succ = 0
    max_dis = 0
    count = 0 # recording the time peroid of search procss
    mal_ups = copy.deepcopy(updates)

    while gama < max_gama:
        count = count +1
        mal_update = (mean_up - gama * deviation) # P8:DYN-OPT mal-up = benign_mean_up - rw
        for i in range(len(updates)):
            if is_corrupted[i]:
                mal_ups[i] = mal_update

        agg_grads, _ , _  = ServerModel.geometric_median_update(mal_ups, weights, maxiter=maxiter)
        dis = np.linalg.norm(agg_grads - mean_up)
        # print("PGA Attack: dis=norm(agg_grads - mean_up): ", dis)

        if dis > max_dis: 
            gama_succ = gama
            max_dis = dis

        gama = gama * (1 + tol)
   
    print("PGA Attack: search count: ",count," final gama_succ: ",gama_succ," norm(max_dis): ", np.linalg.norm(max_dis))
    mal_update = (mean_up - gama_succ * deviation)
    return mal_update

def pga_attack_median(updates, weights, mean_up, is_corrupted, dev_type='unit_vec', max_gama=2000):
    """
    mean_up: mean gradient of benign workers
    """
    if dev_type == 'unit_vec':
        deviation = mean_up / np.linalg.norm(mean_up)  # unit vector, dir to good dir
    elif dev_type == 'sign':
        deviation = np.sign(mean_up)  # P8: w = sign(benign_mean_up)
        
    gama = 1
    tol = 0.2
    gama_succ = 0
    max_dis = 0
    count = 0 # recording the time peroid of search procss
    mal_ups = copy.deepcopy(updates)

    while gama < max_gama:
        count = count +1
        mal_update = (mean_up - gama * deviation) # P8:DYN-OPT mal-up = benign_mean_up - rw
        for i in range(len(updates)):
            if is_corrupted[i]:
                mal_ups[i] = mal_update

        agg_grads = weighted_median(np.array(mal_ups).T, np.array(weights)).T
        dis = np.linalg.norm(agg_grads - mean_up)
        # print("PGA Attack: dis=norm(agg_grads - mean_up): ", dis)

        if dis > max_dis: 
            gama_succ = gama
            max_dis = dis

        gama = gama * (1 + tol)
   
    print("PGA Attack: search count: ",count," final gama_succ: ",gama_succ," norm(max_dis): ", np.linalg.norm(max_dis))
    mal_update = (mean_up - gama_succ * deviation)
    return mal_update

def pga_attack_mebra1(updates, weights, mean_up, is_corrupted,dev_type='sign',lamda=20):
    """
    search method of lamda related to the one from 
    https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning

    updates: updates of all clients, both benign and malicious
    weights: weights of all clients, both benign and malicious
    mean_up: the mean update of benign clients
    is_corrupted: indicator of corrupted/malicious clients, with 1 denoting corrupted and 0 denoting benign
    """
    
    if dev_type == 'unit_vec':
        deviation = mean_up / np.linalg.norm(mean_up)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = np.sign(mean_up)

    # lamda = torch.Tensor([lamda]).float().cuda()  # the search range of coeffcient gama in algorithm2
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    max_min_d = 0
    count = 0 # recording the time peroid of search procss
    mal_ups = copy.deepcopy(updates)
    corp_client_rate = sum(is_corrupted)/len(is_corrupted)

    while abs(lamda_succ - lamda) > threshold_diff:
        count = count + 1
        mal_update = (mean_up - lamda * deviation)
        for i in range(len(updates)):
            if is_corrupted[i]:
                mal_ups[i] = mal_update
        # print("in pgaAtcUtils.py line210: mal_ups[:5]: ",mal_ups[:5])

        ball1 = MEBwO().fit(data=np.array(mal_ups), method=mebwo_alg, eta=(1-corp_client_rate)) # inner
        inner_points = np.asarray([p for p in np.asarray(mal_ups) if ball1.contains(p)])  
        # print("shape of inner points: ", len(inner_points), " shape of weights: ", len(weights))
        agr_up1 = ServerModel.weighted_average_oracle(inner_points, weights)

        ball2 = MEBwO().fit(data=np.array(mal_ups), method=mebwo_alg, eta=(corp_client_rate)) # outer
        outer_points = np.asarray([p for p in np.asarray(mal_ups) if not ball2.contains(p)])
        # print("shape of outer points: ", len(outer_points), " shape of weights: ", len(weights))
        agr_up2 = ServerModel.weighted_average_oracle(outer_points, weights)

        dis1 = np.linalg.norm(agr_up1 - mean_up)
        dis2 = np.linalg.norm(agr_up2 - mean_up)
        min_d = min(dis1, dis2)
        
        if min_d > max_min_d: # maximizing the attak objective
            lamda_succ = lamda
            max_min_d = min_d
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2 
        lamda_fail = lamda_fail / 2 # decresing the search step

    print("search count: ",count," PGA Attack: final lamda_succ: ",lamda_succ," norm(max_min_d): ", np.linalg.norm(max_min_d))
    mal_update = (mean_up - lamda_succ * deviation)
    return mal_update

def pga_attack_trmean1(updates, weights, mean_up, is_corrupted, fraction_to_discard, dev_type='sign', threshold=5.0):
    """
    search method of lamda related to the one from 
    https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning
    
    all_updates: updates of all clients, both benign and malicious
    weights: weights of all clients, both benign and malicious
    mean_up: the mean update of benign clients
    """
    if dev_type == 'sign':
        deviation = np.sign(mean_up)
    elif dev_type == 'unit_vec':
        deviation = mean_up / np.linalg.norm(mean_up)  # unit vector, dir to good dir

    # lamda = torch.Tensor([threshold]).cuda()
    lamda = threshold
    threshold_diff = 1e-5
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0
    count = 0 # recording the time peroid of search procss
    mal_ups = copy.deepcopy(updates)

    while abs(lamda_succ - lamda) > threshold_diff:
        count = count +1
        mal_update = (mean_up - lamda * deviation)
        for i in range(len(updates)):
            if is_corrupted[i]:
                mal_ups[i] = mal_update

        agg_grads = ServerModel.trimmed_mean_update(mal_ups, weights, fraction_to_discard)
        loss = np.linalg.norm(agg_grads - mean_up)

        if prev_loss < loss:
            lamda_succ = lamda
            final_loss = loss
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss
    
    print("search count: ",count," PGA Attack: final lamda_succ: ",lamda_succ," norm(final loss): ", np.linalg.norm(final_loss))
    mal_update = (mean_up - lamda_succ * deviation)
    return mal_update
