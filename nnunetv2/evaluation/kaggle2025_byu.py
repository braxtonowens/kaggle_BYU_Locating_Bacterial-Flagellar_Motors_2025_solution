from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from challenge2025_kaggle_byu_flagellarmotors.evaluation.compute_fbeta import compute_f_beta


def evaluate_folder(folder, gt_file, MIN_P=0.1):
    gt = load_json(gt_file)
    pred_files = [i for i in subfiles(folder, join=False, suffix='.json') if not i.startswith('scores')]
    pred_keys = [i[:-5] for i in pred_files]
    gt_list = [np.array(gt[k]) for k in pred_keys]
    preds = [load_json(join(folder, k)) for k in pred_files]

    probs = np.linspace(MIN_P, 1, num=200)  # np.unique([j for i in preds for j in i['probabilities']])
    fbeta = []
    for threshold in probs:
        pred_list_here = []
        for i in range(len(preds)):
            pred_list_here.append(np.array([preds[i]['coordinates'][j] for j in range(len(preds[i]['probabilities'])) if
                                            preds[i]['probabilities'][j] > threshold]))
        # we need to threshold the prediction
        fbeta.append(compute_f_beta(pred_list_here, gt_list, 2, 35))

    import seaborn as sns
    # Plot using seaborn
    sns.set(style="whitegrid", context="talk")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=probs, y=fbeta, marker='o')
    plt.xlabel("Probability Threshold", fontsize=14)
    plt.ylabel("F\u03B2 Score (β=2)", fontsize=14)
    plt.title("F\u03B2 Score vs Probability Threshold", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(join(folder, 'f2_vs_threshold.png'))

    fbeta_2 = []
    probs2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in probs2:
        pred_list_here = []
        for i in range(len(preds)):
            pred_list_here.append(np.array([preds[i]['coordinates'][j] for j in range(len(preds[i]['probabilities'])) if
                                            preds[i]['probabilities'][j] > threshold]))
        # we need to threshold the prediction
        fbeta_2.append(compute_f_beta(pred_list_here, gt_list, 2, 35))

    save_json(
        {
            'f2_max': np.max(fbeta),
            'f2_max_at_threshold': probs[fbeta == np.max(fbeta)][0],
            'f2_at_0.5': fbeta_2[4],
            'rough_sweep': {i: j for i, j in zip(probs2, fbeta_2)},
        }, join(folder, 'scores.json')
    )
    save_json(
        {
            'precise_sweep': {i: j for i, j in zip(probs, fbeta)},
        }, join(folder, 'scores_precise.json')
    )