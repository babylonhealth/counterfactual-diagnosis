from copy import deepcopy
from pprint import pprint

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from constants import (RESULTS_CF_DISSABLEMENT_FILE,
                       RESULTS_CF_SUFFICIENCY_FILE, RESULTS_OBS_FILE,
                       VIGNETTES_FILE)
from helpers import bintest, doctor_top_ns, mean_list
from utils import load_from_json, load_from_pickle

matplotlib.use("Agg")


def produce_results(*, args):
    print(f"> Producing results from {str(args.results.absolute())}")

    topn_results_obs = load_from_pickle(args.results / RESULTS_OBS_FILE)
    topn_results_counter_diss = load_from_pickle(
        args.results / RESULTS_CF_DISSABLEMENT_FILE
    )
    topn_results_counter_suff = load_from_pickle(
        args.results / RESULTS_CF_SUFFICIENCY_FILE
    )

    make_supplementary_table_one(
        args=args,
        topn_results_obs=topn_results_obs,
        topn_results_counter_diss=topn_results_counter_diss,
        topn_results_counter_suff=topn_results_counter_suff,
    )
    make_figure_three(
        args=args,
        topn_results_obs=topn_results_obs,
        topn_results_counter_diss=topn_results_counter_diss,
        topn_results_counter_suff=topn_results_counter_suff,
    )
    make_table_one_and_supplementary_table_two(
        args=args,
        topn_results_obs=topn_results_obs,
        topn_results_counter_diss=topn_results_counter_diss,
        topn_results_counter_suff=topn_results_counter_suff,
    )
    df_results, doc_topn = make_supplementary_table_three(
        args=args,
        topn_results_obs=topn_results_obs,
        topn_results_counter_diss=topn_results_counter_diss,
        topn_results_counter_suff=topn_results_counter_suff,
    )
    make_table_two(args=args, df_results=df_results, doc_topn=doc_topn)
    make_figure_four(args=args, df_results=df_results)


def make_supplementary_table_one(
    *, args, topn_results_obs, topn_results_counter_diss, topn_results_counter_suff
):

    p_obs = sum(topn_results_obs) / len(topn_results_obs)
    error_obs = np.sqrt(p_obs * (1 - p_obs) / len(topn_results_obs))

    p_counter = sum(topn_results_counter_suff) / len(topn_results_counter_suff)
    error_counter = np.sqrt(
        p_counter * (1 - p_counter) / len(topn_results_counter_suff)
    )

    _p_counter = sum(topn_results_counter_diss) / len(topn_results_counter_diss)
    _error_counter = np.sqrt(
        _p_counter * (1 - _p_counter) / len(topn_results_counter_diss)
    )

    dg2 = pd.DataFrame(
        {
            "N": list(np.arange(20) + 1),
            "Posterior Accuracy": list(p_obs),
            "Posterior Error": list(error_obs),
            "Expected sufficiency": list(p_counter),
            "ES error": list(error_counter),
            "Expected disablement": list(_p_counter),
            "ED error": list(_error_counter),
        }
    )

    dg2["Posterior"] = (
        dg2["Posterior Accuracy"].round(3).apply(str)
        + " $ \pm $ "
        + dg2["Posterior Error"].round(3).apply(str)
    )
    dg2["Disablement"] = (
        dg2["Expected disablement"].round(3).apply(str)
        + " $ \pm $ "
        + dg2["ED error"].round(3).apply(str)
    )
    dg2["Sufficiency"] = (
        dg2["Expected sufficiency"].round(3).apply(str)
        + " $ \pm $ "
        + dg2["ES error"].round(3).apply(str)
    )

    print(
        dg2[["N", "Posterior", "Disablement", "Sufficiency"]].to_latex(
            escape=False, index=False
        )
    )


def make_figure_three(
    *, args, topn_results_obs, topn_results_counter_diss, topn_results_counter_suff
):
    topn_results_obs = load_from_pickle(args.results / RESULTS_OBS_FILE)
    topn_results_counter_suff = load_from_pickle(
        args.results / RESULTS_CF_SUFFICIENCY_FILE
    )

    x = sum(topn_results_obs) / len(topn_results_obs)
    y = sum(topn_results_counter_suff) / len(topn_results_counter_suff)
    plt.figure(figsize=(5, 4))

    p_obs = sum(topn_results_obs) / len(topn_results_obs)
    error_obs = np.sqrt(p_obs * (1 - p_obs) / len(topn_results_obs))
    p_counter = sum(topn_results_counter_suff) / len(topn_results_counter_suff)
    error_counter = np.sqrt(
        p_counter * (1 - p_counter) / len(topn_results_counter_suff)
    )

    xmarks = [i + 1 for i in range(len(p_obs))]

    plt.plot(xmarks, 1 - p_obs, label="Associative", color="blue")
    plt.plot(xmarks, 1 - p_counter, label="Counterfactual", color="seagreen")
    plt.plot(xmarks, 1 - (1 - y) / (1 - x), linestyle="--", color="black")

    plt.fill_between(
        xmarks,
        1 - p_obs - 2 * error_obs,
        1 - p_obs + 2 * error_obs,
        alpha=0.2,
        edgecolor="#1B2ACC",
        facecolor="#089FFF",
        linewidth=0,
        linestyle="None",
        antialiased=True,
    )

    plt.fill_between(
        xmarks,
        1 - p_counter - 2 * error_counter,
        1 - p_counter + 2 * error_counter,
        alpha=0.2,
        edgecolor="#1B2ACC",
        facecolor="seagreen",
        linewidth=0,
        linestyle="None",
        antialiased=True,
    )

    plt.xticks([i + 1 for i in range(16)])
    plt.xlim(1, 15)
    plt.ylim(0, 0.5)
    plt.show()
    plt.savefig(args.results / "algo_vs_algo.pdf")


def make_table_one_and_supplementary_table_two(
    *, args, topn_results_obs, topn_results_counter_diss, topn_results_counter_suff
):
    casecards = load_from_json(args.datapath / VIGNETTES_FILE)

    results_obs = {
        "common": [],
        "rare": [],
        "very_rare": [],
        "almost_impossible": [],
        "uncommon": [],
        "very_common": [],
    }
    results_counter = {
        "common": [],
        "rare": [],
        "very_rare": [],
        "almost_impossible": [],
        "uncommon": [],
        "very_common": [],
    }
    wins_obs = {
        "common": 0,
        "rare": 0,
        "very_rare": 0,
        "almost_impossible": 0,
        "uncommon": 0,
        "very_common": 0,
    }
    wins_counter = {
        "common": 0,
        "rare": 0,
        "very_rare": 0,
        "almost_impossible": 0,
        "uncommon": 0,
        "very_common": 0,
    }
    draws = {
        "common": 0,
        "rare": 0,
        "very_rare": 0,
        "almost_impossible": 0,
        "uncommon": 0,
        "very_common": 0,
    }

    for num, card in enumerate(casecards.values()):
        if args.first is not None and num >= args.first:
            continue

        rareness = card["card"]["diseases"][0]["rareness"]
        r_obs = sum(topn_results_obs[num])
        r_suff = sum(topn_results_counter_suff[num])
        results_obs[rareness] += [min(21 - r_obs, 20)]
        results_counter[rareness] += [min(21 - r_suff, 20)]

        if r_obs > r_suff:
            wins_obs[rareness] += 1
        elif r_obs < r_suff:
            wins_counter[rareness] += 1
        else:
            draws[rareness] += 1

    results_obs_sum = []
    wins_obs_all, wins_counter_all, draws_all = 0, 0, 0
    for k, val in results_obs.items():
        results_obs_sum += val
        wins_obs_all += wins_obs[k]
        draws_all += draws[k]
        results_obs[k] = {"mean": np.mean(val), "std": np.std(val)}
    results_obs["all"] = {
        "mean": np.mean(results_obs_sum),
        "std": np.std(results_obs_sum),
    }

    results_counter_sum = []
    wins_counter_all = 0
    for k, val in results_counter.items():
        results_counter_sum += val
        wins_counter_all += wins_counter[k]
        results_counter[k] = {"mean": np.mean(val), "std": np.std(val)}
    results_counter["all"] = {
        "mean": np.mean(results_counter_sum),
        "std": np.std(results_counter_sum),
    }
    draws["all"] = draws_all
    wins_obs["all"] = wins_obs_all
    wins_counter["all"] = wins_counter_all

    print("> Observational Results")
    pprint(results_obs)
    print("")

    print("> Counterfactual Results")
    pprint(results_counter)
    print("")


def make_supplementary_table_three(
    *, args, topn_results_obs, topn_results_counter_diss, topn_results_counter_suff
):
    casecards = load_from_json(args.datapath / VIGNETTES_FILE)

    paired_results = {}
    doc_topn = {}
    doc_topn_caseav = {}
    doc_score, obs_score = [], []
    for num, card in enumerate(casecards.values()):
        if args.first is not None and num >= args.first:
            continue

        true_id = card["card"]["diseases"][0]["id"]
        pred_suff = topn_results_counter_suff[num]
        pred_diss = topn_results_counter_diss[num]
        pred_obs = topn_results_obs[num]
        doc_res_n = doctor_top_ns(card, true_id)
        for val in doc_res_n:
            if val[1] == 0:
                continue
            if val[0] not in paired_results.keys():
                paired_results[val[0]] = [
                    [
                        deepcopy(val[2]),
                        deepcopy(pred_suff[val[1] - 1]),
                        deepcopy(pred_diss[val[1] - 1]),
                        deepcopy(pred_obs[val[1] - 1]),
                    ]
                ]
            else:
                paired_results[val[0]] += [
                    [
                        deepcopy(val[2]),
                        deepcopy(pred_suff[val[1] - 1]),
                        deepcopy(pred_diss[val[1] - 1]),
                        deepcopy(pred_obs[val[1] - 1]),
                    ]
                ]
        for val in doc_res_n:
            if val[1] == 0:
                continue

            if val[0] not in doc_topn.keys():
                doc_topn[val[0]] = {
                    "count": 1,
                    "sufficiency": {
                        val[1]: np.array([1, deepcopy(pred_suff[val[1] - 1])])
                    },
                    "disablement": {
                        val[1]: np.array([1, deepcopy(pred_diss[val[1] - 1])])
                    },
                    "obs": {val[1]: np.array([1, deepcopy(pred_obs[val[1] - 1])])},
                    "doctor": {val[1]: np.array([1, deepcopy(val[2])])},
                }
            else:
                doc_topn[val[0]]["count"] += 1
                if (
                    val[1] not in doc_topn[val[0]]["sufficiency"].keys()
                ):  # this doctor has never had this score before

                    doc_topn[val[0]]["sufficiency"][val[1]] = np.array(
                        [1, deepcopy(pred_suff[val[1] - 1])]
                    )
                    doc_topn[val[0]]["disablement"][val[1]] = np.array(
                        [1, deepcopy(pred_diss[val[1] - 1])]
                    )
                    doc_topn[val[0]]["obs"][val[1]] = np.array(
                        [1, deepcopy(pred_obs[val[1] - 1])]
                    )
                    doc_topn[val[0]]["doctor"][val[1]] = np.array([1, deepcopy(val[2])])
                else:
                    doc_topn[val[0]]["sufficiency"][val[1]] += np.array(
                        [1, deepcopy(pred_suff[val[1] - 1])]
                    )
                    doc_topn[val[0]]["disablement"][val[1]] += np.array(
                        [1, deepcopy(pred_diss[val[1] - 1])]
                    )
                    doc_topn[val[0]]["obs"][val[1]] += np.array(
                        [1, deepcopy(pred_obs[val[1] - 1])]
                    )
                    doc_topn[val[0]]["doctor"][val[1]] += np.array(
                        [1, deepcopy(val[2])]
                    )

        this_card_res_doc = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
        }
        this_card_res_suff = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
        }
        this_card_res_diss = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
        }
        this_card_res_obs = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
        }
        for val in doc_res_n:
            if val[1] == 0:
                continue
            if val[1] > 9:
                continue
            this_card_res_doc[val[1]] += [val[2]]
            this_card_res_suff[val[1]] += [deepcopy(pred_suff[val[1] - 1])]
            this_card_res_diss[val[1]] += [deepcopy(pred_diss[val[1] - 1])]
            this_card_res_obs[val[1]] += [deepcopy(pred_obs[val[1] - 1])]
        this_card_res_doc = dict(
            [[k, mean_list(val)] for k, val in this_card_res_doc.items()]
        )
        this_card_res_suff = dict(
            [[k, mean_list(val)] for k, val in this_card_res_suff.items()]
        )
        this_card_res_diss = dict(
            [[k, mean_list(val)] for k, val in this_card_res_diss.items()]
        )
        this_card_res_obs = dict(
            [[k, mean_list(val)] for k, val in this_card_res_obs.items()]
        )
        for k, val in this_card_res_doc.items():
            if val == "none":  # no data collected on differentials of this size
                continue
            else:  # if a value was collected for this value for doctors, it was collected for the other two algorithms too
                if k not in doc_topn_caseav.keys():
                    doc_topn_caseav[k] = {
                        "count": 1,
                        "suff": deepcopy(this_card_res_suff[k]),
                        "diss": deepcopy(this_card_res_diss[k]),
                        "obs": deepcopy(this_card_res_obs[k]),
                        "doc": deepcopy(this_card_res_doc[k]),
                    }
                else:  # we have recorded a case of this length before
                    doc_topn_caseav[k]["count"] += 1
                    doc_topn_caseav[k]["suff"] += deepcopy(this_card_res_suff[k])
                    doc_topn_caseav[k]["diss"] += deepcopy(this_card_res_diss[k])
                    doc_topn_caseav[k]["obs"] += deepcopy(this_card_res_obs[k])
                    doc_topn_caseav[k]["doc"] += deepcopy(this_card_res_doc[k])

        # card mean score
        if [val[2] for val in [_val for _val in doc_res_n if _val[1] != 0]] == []:
            continue
        doc_mean_score = np.mean(
            [val[2] for val in [_val for _val in doc_res_n if _val[1] != 0]]
        )
        obs_mean_score = np.mean(
            [
                deepcopy(pred_obs[val[1] - 1])
                for val in [_val for _val in doc_res_n if _val[1] != 0]
            ]
        )
        doc_score += [doc_mean_score]
        obs_score += [obs_mean_score]

    doc_score = []
    doc_error = []
    obs_score = []
    obs_error = []
    suff_score = []
    suff_error = []
    diss_score = []
    diss_error = []

    for k, val in doc_topn.items():
        n = val["count"]
        if n < 50:
            continue

        docp = sum(val["doctor"].values())[1] / n
        obsp = sum(val["obs"].values())[1] / n
        suffp = sum(val["sufficiency"].values())[1] / n
        dissp = sum(val["disablement"].values())[1] / n
        doc_score += [docp]
        doc_error += [np.sqrt(docp * (1 - docp) / n)]
        obs_score += [obsp]
        obs_error += [np.sqrt(obsp * (1 - obsp) / n)]
        suff_score += [suffp]
        suff_error += [np.sqrt(suffp * (1 - suffp) / n)]
        diss_score += [suffp]
        diss_error += [np.sqrt(dissp * (1 - dissp) / n)]

    raw_data = {
        "doc_score": doc_score,
        "doc_error": doc_error,
        "obs_score": obs_score,
        "obs_error": obs_error,
        "sufficiency_score": suff_score,
        "sufficiency_error": suff_error,
        "disablement_score": diss_score,
        "disablement_error": diss_error,
    }
    df_results = pd.DataFrame(
        raw_data,
        columns=[
            "doc_score",
            "doc_error",
            "obs_score",
            "obs_error",
            "sufficiency_score",
            "sufficiency_error",
            "disablement_score",
            "disablement_error",
        ],
    )

    df_results.to_pickle(args.results / "supp_table_3_df.p")

    return df_results, doc_topn


def make_table_two(*, args, df_results, doc_topn):
    comf_thresh = 0.05
    res = np.zeros((4, 4))
    total_number = 0
    for k, val in doc_topn.items():
        if val["count"] < 50:
            continue
        d_doc = [
            [1 for i in range(_val[1])] + [0 for i in range(_val[0] - _val[1])]
            for _val in val["doctor"].values()
        ]
        fd_doc = [item for sublist in d_doc for item in sublist]
        d_obs = [
            [1 for i in range(_val[1])] + [0 for i in range(_val[0] - _val[1])]
            for _val in val["obs"].values()
        ]
        fd_obs = [item for sublist in d_obs for item in sublist]
        d_sufficiency = [
            [1 for i in range(_val[1])] + [0 for i in range(_val[0] - _val[1])]
            for _val in val["sufficiency"].values()
        ]
        fd_sufficiency = [item for sublist in d_sufficiency for item in sublist]
        d_disablement = [
            [1 for i in range(_val[1])] + [0 for i in range(_val[0] - _val[1])]
            for _val in val["disablement"].values()
        ]
        fd_disablement = [item for sublist in d_disablement for item in sublist]

        pvalue_matrix = np.array(
            [
                [
                    0,
                    bintest(fd_doc, fd_obs, comf_thresh),
                    bintest(fd_doc, fd_sufficiency, comf_thresh),
                    bintest(fd_doc, fd_disablement, comf_thresh),
                ],
                [
                    bintest(fd_obs, fd_doc, comf_thresh),
                    0,
                    bintest(fd_obs, fd_sufficiency, comf_thresh),
                    bintest(fd_obs, fd_disablement, comf_thresh),
                ],
                [
                    bintest(fd_sufficiency, fd_doc, comf_thresh),
                    bintest(fd_sufficiency, fd_obs, comf_thresh),
                    0,
                    bintest(fd_sufficiency, fd_disablement, comf_thresh),
                ],
                [
                    bintest(fd_disablement, fd_doc, comf_thresh),
                    bintest(fd_disablement, fd_obs, comf_thresh),
                    bintest(fd_disablement, fd_sufficiency, comf_thresh),
                    0,
                ],
            ]
        )

        res += pvalue_matrix

        total_number += 1
    print(np.around(res, decimals=3))

    print(
        f"Doctor Score:\t {round(df_results.doc_score.mean(), 4)}\n"
        f"Obs Score:\t{round(df_results.obs_score.mean(), 4)}\n"
        f"Suff Score: \t{round(df_results.sufficiency_score.mean(), 4)}\n"
        f"Disab Score:\t{round(df_results.disablement_score.mean(), 4)}\n"
    )


def make_figure_four(*, args, df_results):
    cmap, norm = matplotlib.colors.from_levels_and_colors(
        [0, 0.499, 0.501, 1], ["blue", "red", "green"]
    )

    x = df_results.doc_score
    y = df_results.obs_score
    col = [(x[i] - y[i]) + 0.5 for i in range(len(x))]
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, c=col, cmap=cmap, norm=norm)
    xticks = np.linspace(min(min(x), min(y)), max(max(x), max(y)), 100)
    yticks = xticks
    plt.xticks(np.arange(0.5, 0.95, 0.1))
    plt.yticks(np.arange(0.5, 0.95, 0.1))
    plt.plot(xticks, yticks, "-r", linestyle="--")

    plt.savefig(args.results / "obs_vs_doc.pdf")

    plt.clf()
    x = df_results.doc_score
    y = df_results.sufficiency_score
    col = [(x[i] - y[i]) + 0.5 for i in range(len(x))]
    plt.scatter(x, y, c=col, cmap=cmap, norm=norm)
    xticks = np.linspace(min(min(x), min(y)), max(max(x), max(y)), 100)
    yticks = xticks
    plt.xticks(np.arange(0.5, 0.95, 0.1))
    plt.yticks(np.arange(0.5, 0.95, 0.1))
    plt.plot(xticks, yticks, "-r", linestyle="--")

    plt.savefig(args.results / "suff_vs_doc.pdf")
