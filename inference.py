import copy
import itertools
from functools import reduce
from operator import mul

import numpy as np

from constants import NETWORKS_FILE
from helpers import load_networks


def get_evidence_from_casecard(card):
    risk_ev = [
        [k["concept"]["id"], "True" if k["presence"] == "PRESENT" else "False"]
        for k in card["card"]["risk_factors"]
        if k["label"] == "Risk"
    ]
    symptom_ev = [
        [k["concept"]["id"], "False" if k["severity"] == "NOT_PRESENT" else "True"]
        for k in card["card"]["symptoms"]
        if (k["label"] != "Super") & (k["concept"]["id"] is not None)
    ]
    int_ev = [{"id": el[0], "state": el[1]} for el in risk_ev + symptom_ev]
    return int_ev


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator. so define a list L and then call pL = [x for x in powerset(l)]
    """
    if len(seq) == 0:
        yield []
    elif len(seq) == 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item


def disease_ids(network_data):
    return [i for i, j in list(network_data.items()) if j["label"] == "Disease"]


def risk_factor_ids(network_data):
    return [i for i, j in list(network_data.items()) if j["label"] == "Risk"]


def symptom_ids(network_data):
    return [i for i, j in list(network_data.items()) if j["label"] == "Symptom"]


def childvec(network_data):
    disease_nodes = disease_ids(network_data)
    risk_factor_dict = dict((el, []) for el in risk_factor_ids(network_data))
    for _id in disease_nodes:

        for parent_id in network_data[_id]["parents"]:
            risk_factor_dict[parent_id].append(_id)
    return risk_factor_dict


def childvec_diseases(network_data):
    disease_nodes = disease_ids(network_data)
    symptom_nodes = symptom_ids(network_data)
    disease_dict = dict((el, []) for el in disease_nodes)
    for _id in symptom_nodes:

        for parent_id in network_data[_id]["parents"]:
            disease_dict[parent_id].append(_id)
    return disease_dict


def get_lambda(s, d, network_data):
    if d not in network_data[s]["parents"]:
        return 1
    else:
        return network_data[s]["cpt"][network_data[s]["parents"].index(d)]


def get_coeff(disease, dur, nd):
    if disease in nd["cd4c5fb4-21bc-4e13-ac50-6eee8d24e769"]["parents"]:
        return max(
            0.01,
            nd["cd4c5fb4-21bc-4e13-ac50-6eee8d24e769"]["cpt"][
                nd["cd4c5fb4-21bc-4e13-ac50-6eee8d24e769"]["parents"].index(disease)
            ][dur - 1],
        )
    else:
        return 0.5


def get_parents(y, joint_dict, parent_ids):
    if y in joint_dict.keys():
        return [k for k in joint_dict[y].keys() if k in parent_ids]
    else:
        return []


def DOS_multiply(duration, results, nd):
    cui_to_duration = {
        "C0436361": 1,
        "C0436362": 2,
        "C0436363": 3,
        "C0436364": 4,
        "C0436365": 5,
    }
    dur_num = cui_to_duration[duration]
    return dict([[k, val * get_coeff(k, dur_num, nd)] for k, val in results.items()])


def counterfactual_correction_sufficiency(
    disease_children, d, Z, positive_symptoms, network_data
):
    if len(Z) == len(positive_symptoms):  # sum is trivial when over no terms
        return 0
    else:
        return len(list(set(positive_symptoms) - set(Z))) - sum(
            [
                get_lambda(_s, d, network_data)
                for _s in list(set(positive_symptoms) - set(Z))
            ]
        )


def counterfactual_correction_disablement(
    disease_children, d, A, on_symptoms, network_data
):
    if len(A) == 0:
        return 0
    else:
        lambda_vec = [get_lambda(_s, d, network_data) for _s in A]
        return (
            len(A)
            - sum([1 / x for x in lambda_vec if x > 0])
            - sum([1 for x in lambda_vec if x == 0])
        )


def update_marginals(
    network_data, risk_factor_evidence, disease_marginals, marginal_matrix
):
    disease_numbers = dict(
        [[d, num] for num, d in enumerate(disease_ids(network_data))]
    )
    risk_factor_children = childvec(network_data)
    disease_marginal_transform = np.ones(len(disease_marginals))
    disease_matrix_transform = np.ones([len(disease_marginals), len(disease_marginals)])
    for item in risk_factor_evidence.items():
        r, val = item
        p_r = network_data[r]["cpt"][0]
        r_children = risk_factor_children[r]
        lambda_matrix = np.ones([len(disease_marginals), len(disease_marginals)])
        for child in r_children:
            child_number = disease_numbers[child]
            child_lambda = network_data[child]["cpt"][
                network_data[child]["parents"].index(r)
            ]
            lambda_matrix[child_number, :] = (
                lambda_matrix[child_number, :] * child_lambda
            )
            lambda_matrix[:, child_number] = (
                lambda_matrix[:, child_number] * child_lambda
            )
            if val == 1:
                disease_marginal_transform[child_number] = (
                    disease_marginal_transform[child_number]
                    * child_lambda
                    / (child_lambda * (1 - p_r) + p_r)
                )
            else:
                disease_marginal_transform[child_number] = (
                    disease_marginal_transform[child_number]
                    * 1
                    / (child_lambda * (1 - p_r) + p_r)
                )
        if val == 1:
            disease_matrix_transform = disease_matrix_transform * (
                lambda_matrix / ((1 - p_r) * lambda_matrix + p_r)
            )  # addition of scalar applies to all elements
        else:
            disease_matrix_transform = disease_matrix_transform * (
                np.ones([len(disease_marginals), len(disease_marginals)])
                / ((1 - p_r) * lambda_matrix + p_r)
            )
    transformed_marginal_vector = disease_marginals * disease_marginal_transform
    transformed_marginal_matrix = marginal_matrix * disease_matrix_transform

    sigma = np.sqrt(transformed_marginal_vector * (1 - transformed_marginal_vector))
    Phi = (
        transformed_marginal_matrix
        - np.triu(
            np.outer(transformed_marginal_vector, transformed_marginal_vector), k=1
        )
    ) / np.outer(sigma, sigma)
    updated_marginals_trans = 1 - ((1 - transformed_marginal_vector) + 0.002)
    sigma_new = np.sqrt(updated_marginals_trans * (1 - updated_marginals_trans))
    updated_marginal_matrix_trans = np.triu(
        np.outer(updated_marginals_trans, updated_marginals_trans), k=1
    ) + Phi * np.triu(np.outer(sigma_new, sigma_new), k=1)

    return updated_marginals_trans, updated_marginal_matrix_trans


def approximate_inference(network_data, evidence, network_name, dos, datapath):
    evidence = [[key, value] for key, value in evidence.items()]
    risk_evidence = {}
    positive_symptoms = []
    negative_symptoms = []
    for ev in evidence:
        if ev[0] not in list(network_data.keys()):
            print("Error: evidence not in network")
            break
        counter = 0
        if ev[1] == "True":
            counter = 1
        else:
            counter = 0
        if network_data[ev[0]]["label"] == "Symptom":
            if ev[1] == "False":
                negative_symptoms += [ev[0]]
            else:
                positive_symptoms += [ev[0]]
        else:
            risk_evidence[ev[0]] = counter
    disease_children = childvec_diseases(network_data)

    disease_marginals = np.load(
        datapath / f"{network_name}_single_disease_marginals.npy"
    )
    marginal_matrix = np.load(
        datapath / f"{network_name}_bipartite_disease_marginals.npy"
    )

    updated_marginals, updated_marginal_matrix = update_marginals(
        network_data, risk_evidence, disease_marginals, marginal_matrix
    )

    single_disease_marginals = [
        [k, updated_marginals[num]] for num, k in enumerate(disease_ids(network_data))
    ]
    correlation_matrix = updated_marginal_matrix - np.triu(
        np.outer(updated_marginals, updated_marginals), k=1
    )
    correlation_matrix[np.abs(correlation_matrix) <= 1e-16] = 0
    correlation_matrix[correlation_matrix < 0] = 0
    counter_res_sufficiency, counter_res_disablement, obs_res = posteriors_and_CFs(
        network_data,
        positive_symptoms,
        negative_symptoms,
        single_disease_marginals,
        correlation_matrix,
        disease_children,
    )

    if any(np.isnan(obs_res)):
        print("obs Nan", end="\n")
        return False, False
    obs_res[np.isnan(obs_res)] = 0

    output_obs = {}
    for num, d in enumerate(single_disease_marginals):

        output_obs[d[0]] = obs_res[num]

    if any(np.isnan(counter_res_sufficiency)):
        print("counter Nan", end="\n")
        return False, False
    counter_res_sufficiency[np.isnan(counter_res_sufficiency)] = 0

    output_counter_sufficiency = {}
    for num, d in enumerate(single_disease_marginals):

        output_counter_sufficiency[d[0]] = counter_res_sufficiency[num]

    if any(np.isnan(counter_res_disablement)):
        print("counter Nan", end="\n")
        return False, False
    counter_res_disablement[np.isnan(counter_res_disablement)] = 0

    output_counter_disablement = {}
    for num, d in enumerate(single_disease_marginals):

        output_counter_disablement[d[0]] = counter_res_disablement[num]

    output_counter_sufficiency, output_counter_disablement, output_obs = (
        DOS_multiply(dos, output_counter_sufficiency, network_data),
        DOS_multiply(dos, output_counter_disablement, network_data),
        DOS_multiply(dos, output_obs, network_data),
    )
    return output_counter_sufficiency, output_counter_disablement, output_obs


def posteriors_and_CFs(
    network_data,
    on_symptoms,
    off_symptoms,
    disease_marginals,
    correlation_matrix,
    disease_children,
):
    inference_results_sufficiency = np.zeros(len(disease_marginals))
    inference_results_disablement = np.zeros(len(disease_marginals))
    obs_inference_results = np.zeros(len(disease_marginals))
    Smarg = 0

    # 'list' is necessary, as it returns [ [] ] when you have no on-symptoms, which means you still calculate smarg for all evidence false
    for s in list(powerset(on_symptoms)):
        s_join = s + off_symptoms
        arg_0 = ((-1) ** len(s)) * reduce(
            mul, [network_data[symp]["cpt"][-1] for symp in s_join], 1
        )
        parent_diseases = set(
            list(
                itertools.chain.from_iterable(
                    [network_data[i]["parents"] for i in s_join]
                )
            )
        )
        biglambda = 1
        av_terms, lambda_vec, unav = [], [], []
        #  construct big lambda and aggrigate parent lambdas
        for k_d, p_d in disease_marginals:
            if k_d in parent_diseases:
                symptoms_of_d = set(disease_children[k_d]) & set(s_join)
                prod_lambda = reduce(
                    mul,
                    [
                        network_data[s]["cpt"][network_data[s]["parents"].index(k_d)]
                        for s in symptoms_of_d
                    ],
                    1,
                )
                biglambda *= prod_lambda * (1 - p_d) + p_d
                unav += [(prod_lambda * (1 - p_d)) / (prod_lambda * (1 - p_d) + p_d)]
                lambda_vec += [(prod_lambda - 1)]
                av_terms += [1 / (prod_lambda * (1 - p_d) + p_d)]

            else:
                unav += [(1 - p_d)]
                lambda_vec += [0]
                av_terms += [1]

        # append 1st order term to results
        Smarg += arg_0 * biglambda
        temp_results = (
            arg_0 * biglambda * np.array(unav)
        )  # part of p(S_- = 0, Z = 0, d_k = 1|R) that comes from product term in correlator expansion
        obs_inference_results += arg_0 * biglambda * np.array(unav)
        outervec = np.array(
            [(lambda_vec[i]) * av_terms[i] for i in range(len(disease_marginals))]
        )
        M = correlation_matrix * np.outer(outervec, outervec)
        Msum = np.sum(M)
        correlator_correction = np.array(
            [
                arg_0
                * biglambda
                * unav[num]
                * (Msum - np.sum(M[num, :]) - np.sum(M[:, num]))
                + arg_0
                * biglambda
                * (lambda_vec[num] + 1)
                * av_terms[num]
                * np.sum(correlation_matrix[:, num] * outervec)
                + arg_0
                * biglambda
                * (lambda_vec[num] + 1)
                * av_terms[num]
                * np.sum(correlation_matrix[num, :] * outervec)
                for num in range(len(disease_marginals))
            ]
        )

        temp_results += correlator_correction
        obs_inference_results += correlator_correction
        full_av_vector = np.array(lambda_vec) * np.array(av_terms)
        Smarg += (
            arg_0
            * biglambda
            * np.sum(correlation_matrix * np.outer(full_av_vector, full_av_vector))
        )
        inference_results_sufficiency += temp_results * np.array(
            [
                counterfactual_correction_sufficiency(
                    disease_children, d, s, on_symptoms, network_data
                )
                for d in disease_ids(network_data)
            ]
        )
        inference_results_disablement += temp_results * np.array(
            [
                counterfactual_correction_disablement(
                    disease_children, d, s, on_symptoms, network_data
                )
                for d in disease_ids(network_data)
            ]
        )

    return (
        inference_results_sufficiency / Smarg,
        inference_results_disablement / Smarg,
        obs_inference_results / Smarg,
    )


def create_marginals_files(args):
    networks = load_networks(datapath=args.datapath, filename=NETWORKS_FILE)

    for network_name, network in networks.items():
        construct_marginals(
            network_data=network, network_name=network_name, datapath=args.datapath
        )


def construct_marginals(network_data, network_name, datapath, max_p=1 - 1e-12):
    print(f"> Consructing marginals for {network_name}")

    def get_p(d, network_data):
        return min(network_data[d]["cpt"][0], max_p)

    diseases = disease_ids(network_data)
    risk_factor_nodes = risk_factor_ids(network_data)
    risk_children_dict = childvec(network_data)
    disease_marginals = []
    for d in diseases:
        # leak lambda multiplies everything
        prod_lambda = 0
        if network_data[d]["parents"] == []:  #  intialise prodlambda with leak lambda
            prod_lambda = get_p(
                d,
                network_data,
            )
        else:
            prod_lambda = network_data[d]["cpt"][-1]
            # load risk factors and calculate marginal
            for num, _id in enumerate(network_data[d]["parents"]):
                prob_r = copy.deepcopy(
                    network_data[_id]["cpt"]
                )  # simple hack to prevent us from overwriting risk factor probabilities
                prod_lambda = prod_lambda * (
                    prob_r[0] + prob_r[1] * network_data[d]["cpt"][num]
                )
        disease_marginals += [prod_lambda]
    # now calculate the matrix of joint zero states
    disease_marginal_matrix = np.zeros([len(disease_marginals), len(disease_marginals)])
    for i, d in enumerate(diseases):
        for j, d_prime in enumerate(diseases):
            if j <= i:  # only need to fill top triangle as symmetric
                continue
            else:
                prod_lambda = 1
                if network_data[d]["parents"] == []:
                    prod_lambda = prod_lambda * get_p(
                        d,
                        network_data,
                    )
                else:
                    prod_lambda = prod_lambda * network_data[d]["cpt"][-1]
                if network_data[d_prime]["parents"] == []:
                    prod_lambda = prod_lambda * get_p(
                        d_prime,
                        network_data,
                    )
                else:
                    prod_lambda = prod_lambda * network_data[d_prime]["cpt"][-1]
                for num, _id in enumerate(
                    list(
                        set(
                            network_data[d]["parents"]
                            + network_data[d_prime]["parents"]
                        )
                    )
                ):
                    prob_r = copy.deepcopy(network_data[_id]["cpt"])
                    lamb_1, lamb_2 = 1, 1
                    if _id in network_data[d]["parents"]:
                        lamb_1 = network_data[d]["cpt"][
                            network_data[d]["parents"].index(_id)
                        ]
                    if _id in network_data[d_prime]["parents"]:
                        lamb_2 = network_data[d_prime]["cpt"][
                            network_data[d_prime]["parents"].index(_id)
                        ]
                    prod_lambda = prod_lambda * (
                        prob_r[0] + prob_r[1] * lamb_1 * lamb_2
                    )
                disease_marginal_matrix[i, j] = prod_lambda

    single_path = datapath / f"{network_name}_single_disease_marginals.npy"
    bipart_path = datapath / f"{network_name}_bipartite_disease_marginals.npy"

    print(f"  > Writing {single_path}")
    np.save(single_path, np.array(disease_marginals))
    print(f"  > Writing {bipart_path}")
    np.save(bipart_path, disease_marginal_matrix)
