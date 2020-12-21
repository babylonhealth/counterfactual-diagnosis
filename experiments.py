import numpy as np
from tqdm import tqdm

from constants import (DATA_PATH, NETWORKS_FILE, RESULTS_CF_DISSABLEMENT_FILE,
                       RESULTS_CF_SUFFICIENCY_FILE, RESULTS_FILE,
                       RESULTS_OBS_FILE, VIGNETTES_FILE)
from helpers import load_networks
from inference import approximate_inference, get_evidence_from_casecard
from utils import load_from_json, write_to_pickle


def run_single_vignette(*, card, networks=None, datapath=DATA_PATH):
    if networks is None:
        networks = load_networks(datapath)

    network_name = card["card"]["network_name"]
    network_data = networks[network_name]
    true_id = card["card"]["diseases"][0]["id"]
    evidence = get_evidence_from_casecard(card)
    evidence = dict(
        [
            [ev["id"], ev["state"]]
            for ev in evidence
            if ev["id"] != "cd4c5fb4-21bc-4e13-ac50-6eee8d24e769"
        ]
    )
    dos = card["card"]["duration"]
    counter_suff, counter_diss, obs = approximate_inference(
        network_data, evidence, network_name, dos, datapath
    )

    return counter_suff, counter_diss, obs, true_id


def run_vignettes_experiment(*, args):
    if args.reproduce is False:
        # run over the test_networks.json file and perform inference calculation
        networks = load_from_json(args.datapath / NETWORKS_FILE)
        casecards = load_from_json(args.datapath / VIGNETTES_FILE)
        inference_output = None
    else:
        # use pre-calcd inference output
        networks = None
        inference_output = load_from_json(args.datapath / RESULTS_FILE)
        casecards = load_from_json(args.datapath / VIGNETTES_FILE)

    topn_results_obs = []
    topn_results_counter_suff = []
    topn_results_counter_diss = []

    count_all = 0
    ind_obs_store = []
    ind_suff_store = []
    ind_diss_store = []

    total_to_run = len(casecards)
    if args.first is not None:
        total_to_run = args.first

    pbar = tqdm(total=total_to_run, desc="Casecards", unit="cards")

    for card in casecards.values():

        if args.first is not None and count_all >= args.first:
            continue

        if args.reproduce is False:
            if card["card"]["network_name"] not in networks:
                continue

        if inference_output is None and networks is not None:
            counter_suff, counter_diss, obs, true_id = run_single_vignette(
                card=card,
                networks=networks,
                datapath=args.datapath,
            )
        else:
            output = inference_output[str(card["card"]["id"])]
            counter_suff = output["sufficiency"]
            counter_diss = output["disablement"]
            obs = output["posterior"]
            true_id = card["card"]["diseases"][0]["id"]

        pred_suff = np.array(
            [
                1
                if true_id
                in sorted(counter_suff, key=counter_suff.get, reverse=True)[:i]
                else 0
                for i in range(1, 21)
            ]
        )
        pred_diss = np.array(
            [
                1
                if true_id
                in sorted(counter_diss, key=counter_diss.get, reverse=True)[:i]
                else 0
                for i in range(1, 21)
            ]
        )
        pred_obs = np.array(
            [
                1 if true_id in sorted(obs, key=obs.get, reverse=True)[:i] else 0
                for i in range(1, 21)
            ]
        )
        topn_results_obs += [pred_obs]
        topn_results_counter_suff += [pred_suff]
        topn_results_counter_diss += [pred_diss]
        count_all += 1

        pbar.update(1)

        if args.verbose and (
            (count_all % 10 == 0) or (count_all == len(casecards) - 1)
        ):
            pbar.write(f"N_processed: {count_all}")
            pbar.write(f"TopN CFSuff: {sum(topn_results_counter_suff) / count_all}")
            pbar.write(f"TopN CFDiss: {sum(topn_results_counter_diss) / count_all}")
            pbar.write(f"TopN Obs:    {sum(topn_results_obs) / count_all}\n")

    write_to_pickle(topn_results_obs, args.results / RESULTS_OBS_FILE)
    write_to_pickle(
        topn_results_counter_diss, args.results / RESULTS_CF_DISSABLEMENT_FILE
    )
    write_to_pickle(
        topn_results_counter_suff, args.results / RESULTS_CF_SUFFICIENCY_FILE
    )
