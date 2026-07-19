import numpy as np
import os
import matplotlib.pyplot as plt 
from scipy.stats import norm
import pandas as pd 
import json
import pickle
import argparse
import matplotlib
matplotlib.use("Agg")
# plt.style.use(["ggplot"])
# %config InlineBackend.figure_format = 'svg'

def analysis(root_path, output_path, nde_path=None):

    result_data = []
    for file in os.listdir(root_path):
        if not file.endswith(".json"):
            for file2 in os.listdir(os.path.join(root_path, file)):
                if file2.endswith(".npz"):
                    file_path = os.path.join(root_path, file, file2)
                    result_data.extend((np.load(file_path)['is_weight'] * (1-np.load(file_path)['success'])).tolist())
    print("NADE crash rate:", np.sum(np.array(result_data))/len(result_data))

    nde_data = np.zeros((2000,))
    nde_data[:int(0.0206*2000)] += 1.0
    np.random.shuffle(nde_data)
    np.random.shuffle(result_data)
    nde_data = nde_data.tolist()
    print("NDE crash rate:", np.sum(np.array(nde_data))/len(nde_data))

    confidence_interval = 0.1
    z = norm.ppf(1-confidence_interval/2)

    original_result = pd.Series(result_data)
    crash_mean_result = original_result.rolling(len(original_result), min_periods=1).mean()
    unit_std_result = original_result.rolling(len(original_result), min_periods=1).std()
    half_CI = z*unit_std_result/(np.sqrt(np.array(range(1, len(original_result)+1)))*crash_mean_result)
    half_CI_numpy = half_CI.to_numpy().astype(float)
    crash_mean_result_numpy = crash_mean_result.to_numpy().astype(float)
    #print("crash rate:", crash_mean_result_numpy[-1])
    print("RHW converge to 0.3 episode:", np.where(half_CI_numpy > 0.3)[0][-1])
    print("Mean = {:.10f}, Lower bound = {:.10f}, Higher bound = {:.10f}".format(crash_mean_result_numpy[-1], crash_mean_result_numpy[-1] * (1 - half_CI_numpy[-1]), crash_mean_result_numpy[-1] * (1 + half_CI_numpy[-1])))
    print("Final RHW:", half_CI_numpy[-1])

    fig = plt.figure(figsize=(8,6), dpi=100)
    plt.ylim(0,10e-2)
    plt.plot(crash_mean_result_numpy, label=f"NADE: {crash_mean_result_numpy[-1]:.2e}", color="#8B0000", linewidth=2)
    plt.fill_between(range(len(crash_mean_result_numpy)), (1-half_CI_numpy)*crash_mean_result_numpy, (1+half_CI_numpy)*crash_mean_result_numpy, color=(1.0, 215/255, 215/255), alpha=0.75)
    # plt.plot([0,len(crash_mean_result_numpy)],[crash_mean_result_numpy[-1]]*2,"--",label=f"NADE: {crash_mean_result_numpy[-1]:.2e}", alpha=0.5)

    nde_result = pd.Series(nde_data)
    nde_crash_mean_result = nde_result.rolling(len(nde_result), min_periods=1).mean()
    nde_half_CI = z*nde_result.rolling(len(nde_result), min_periods=1).std()/(np.sqrt(np.array(range(1, len(nde_result)+1)))*nde_crash_mean_result)
    nde_half_CI_numpy = nde_half_CI.to_numpy().astype(float)
    nde_crash_mean_result_numpy = nde_crash_mean_result.to_numpy().astype(float)
    plt.plot(nde_crash_mean_result_numpy, label=f"NDE: {nde_crash_mean_result_numpy[-1]:.2e}", color="#012169", linewidth=2)
    plt.fill_between(range(len(nde_crash_mean_result_numpy)), (1-nde_half_CI_numpy)*nde_crash_mean_result_numpy, (1+nde_half_CI_numpy)*nde_crash_mean_result_numpy, color=(198/255, 222/255, 1.0), alpha=0.55)
    plt.plot([0,len(nde_crash_mean_result_numpy)],[nde_crash_mean_result_numpy[-1]]*2, "--", color="#000000", alpha=1, linewidth=1)

    plt.xlabel("Number of tests",fontsize=18)
    plt.ylabel("Failure rate",fontsize=18)
    ax = fig.gca()
    ax.ticklabel_format(style='sci', scilimits=(-10,-8), axis='y')
    ax.ticklabel_format(style='sci', scilimits=(-10,-8), axis='x')
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(15)
    tx = ax.yaxis.get_offset_text()
    tx.set_fontsize(15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(os.path.join(output_path,"crash_rate.png"),dpi=600)
    plt.close("all")

    RHF = half_CI_numpy.tolist()
    font = {'family': 'serif',
               'size': 15}
    plt.rc('font', **font)
     
    plt.figure(figsize=(12, 6))
    plt.plot(RHF, label='NADE RHF', color='#8B0000', linewidth=2)

    try:
        RHF_arr = np.array(RHF)
        idx_nade = np.where(RHF_arr <= 0.3)[0]
        if idx_nade.size > 0:
            x_nade = int(idx_nade[0])
            plt.axvline(x=x_nade, color='#8B0000', linestyle='--', linewidth=1.5, alpha=0.9)
            plt.plot([x_nade], [RHF_arr[x_nade]], marker='o', color='#8B0000')
            plt.text(x_nade + max(1, int(len(RHF_arr)*0.02)), RHF_arr[x_nade] + 0.02, f"{x_nade:.2e}", color='#8B0000', fontsize=16)
    except Exception:
        pass

    nde_RHF = nde_half_CI_numpy.tolist()
    plt.plot(nde_RHF, label='NDE RHF', color='#012169', linewidth=2)
    try:
        nde_RHF_arr = np.array(nde_RHF)
        idx_nde = np.where(nde_RHF_arr <= 0.3)[0]
        if idx_nde.size > 0:
            x_nde = int(idx_nde[0])
            plt.axvline(x=x_nde, color='#012169', linestyle='--', linewidth=1.5, alpha=0.9)
            plt.plot([x_nde], [nde_RHF_arr[x_nde]], marker='o', color='#012169')
            plt.text(x_nde + max(1, int(len(nde_RHF_arr)*0.02)), nde_RHF_arr[x_nde] + 0.02, f"{x_nde:.2e}", color='#012169', fontsize=16)
    except Exception:
        pass
    plt.plot([0, len(nde_RHF)], [0.3]*2, "--", color="#000000", alpha=1, linewidth=1, label='RHF = 0.3')
    
    plt.xlabel('Number of tests', fontsize=20)
    plt.ylabel('Relative Half-Width (RHF)', fontsize=20)
    plt.title('RHF Convergence Curve', fontsize=20)
    plt.legend()
    
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'RHF.png'), dpi=600)
    plt.close("all")

    boostrap_results = []
    boostrap_nde = []
    for _ in range(1000):
        sample = np.random.choice(result_data, size=len(result_data), replace=False)
        original_result = pd.Series(sample)
        crash_mean_result = original_result.rolling(len(original_result), min_periods=1).mean()
        unit_std_result = original_result.rolling(len(original_result), min_periods=1).std()
        half_CI = z*unit_std_result/(np.sqrt(np.array(range(1, len(original_result)+1)))*crash_mean_result)
        half_CI_numpy = half_CI.to_numpy()
        boostrap_results.append(np.where(half_CI_numpy > 0.3)[0][-1])

    for _ in range(1000):
        nde_sample = np.random.choice(nde_data, size=len(nde_data), replace=True)
        nde_result = pd.Series(nde_sample)
        nde_crash_mean_result = nde_result.rolling(len(nde_result), min_periods=1).mean()
        nde_unit_std_result = nde_result.rolling(len(nde_result), min_periods=1).std()
        nde_half_CI = z*nde_unit_std_result/(np.sqrt(np.array(range(1, len(nde_result)+1)))*nde_crash_mean_result)
        nde_half_CI_numpy = nde_half_CI.to_numpy()
        boostrap_nde.append(np.where(nde_half_CI_numpy > 0.3)[0][-1])

    plt.figure(figsize=(8, 6))
    nade_mean = np.mean(boostrap_results)
    nade_std = np.std(boostrap_results)
    nade_bars = plt.bar(['NADE'], [nade_mean], width=0.7, yerr=[nade_std], capsize=10, color='#8B0000')
    nde_bars = None

    nde_mean = np.mean(boostrap_nde)
    nde_std = np.std(boostrap_nde)
    nde_bars = plt.bar(['NDE'], [nde_mean], width=0.7, yerr=[nde_std], capsize=10, color='#012169')
    # annotate bars with mean ± std
    for bar in nade_bars:
        h = bar.get_height() + nade_std + nade_std*0.1
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{nade_mean:.1f} ± {nade_std:.1f}", ha='center', va='bottom', fontsize=15)
    if nde_bars is not None:
        for bar in nde_bars:
            h = bar.get_height() + nde_std + nade_std*0.1
            plt.text(bar.get_x() + bar.get_width() / 2, h, f"{nde_mean:.1f} ± {nde_std:.1f}", ha='center', va='bottom', fontsize=15)
    plt.xlim(-0.8, 1.8)
    plt.ylim(0, max(nade_mean + nade_std, nde_mean + nde_std) * 1.3)
    plt.ylabel('Number of tests to RHF <= 0.3', fontsize=15)
    plt.title('Bootstrap results (1000 samples)', fontsize=15)
    plt.savefig(os.path.join(output_path, 'bootstrap_results.png'), bbox_inches='tight', dpi=600)
    plt.close("all")

if __name__ == '__main__':

    root_path = "/mnt/hlx/SimpleVLA_libero_data/test_results/nade_original_policy"
    output_path = './'
    analysis(root_path, output_path)