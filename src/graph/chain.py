import pandas as pd
import numpy as np
from gerrychain import MarkovChain
from gerrychain.proposals import recom
from gerrychain.constraints import within_percent_of_ideal_population, Validator
from gerrychain.accept import always_accept
from functools import partial
from tqdm import tqdm
import multiprocessing as mp
import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from construction import build_precinct_graph, validate_precinct_graph
from metrics import MCalc

def gelman_rubin(chains, param="EfficiencyGap"):
    n = min(len(c) for c in chains)

    samples = [c[param].values[:n] for c in chains]
    
    W = np.mean([np.var(s, ddof=1) for s in samples])
    if not W:
        return 1.0
    
    chain_means = np.array([np.mean(s) for s in samples])
    B = n * np.var(chain_means, ddof=1)

    var_hat = (1 - 1/n) * W + B/n
    R_hat = np.sqrt(var_hat / W)
    return R_hat

def run_single_chain(args):
    chain_id, state, basepath, steps, pop_tol, thinning, burn_in, include_geometry = args
    
    graph, partition = build_precinct_graph(state, basepath)
    
    validation = validate_precinct_graph(graph, partition, tolerance=pop_tol)
    if not validation["overall"]:
        print(f"Chain {chain_id}: Invalid initial partition")
        return None

    ideal_pop = sum(partition["population"].values()) / len(partition)
    proposal = partial(
        recom,
        pop_col="P0010001",
        pop_target=ideal_pop,
        epsilon=pop_tol,
        node_repeats=2,
    )
    constraints = [within_percent_of_ideal_population(partition, pop_tol)]
    chain = MarkovChain(
        proposal=proposal,
        constraints=Validator(constraints),
        accept=always_accept,
        initial_state=partition,
        total_steps=steps
    )

    mc = MCalc()
    burn_in_steps = int(steps * burn_in)
    results = []
    sample_count = 0
    
    output_file = f"{basepath}/{state}/chain_{chain_id}_metrics.csv"
    Path(f"{basepath}/{state}").mkdir(parents=True, exist_ok=True)
    
    for i, part in enumerate(tqdm(chain, total=steps, desc=f"Chain {chain_id} progress")):
        if i % thinning != 0:
            continue

        if i < burn_in_steps:
            if i % 10000 == 0 and i > 0:
                print(f"Chain {chain_id}: Burn-in progress: {i}/{burn_in_steps} steps ({i/burn_in_steps*100:.1f}%)")
            continue

        metrics = mc.calculate_metrics(part, baseline=False, include_geometry=include_geometry).iloc[0]
        metrics["step"] = i
        metrics["chain"] = chain_id
        results.append(metrics)
        sample_count += 1

        if sample_count % 100 == 0:
            df = pd.DataFrame(results)
            df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            results = []
            print(f"Chain {chain_id}: Processed {sample_count} samples, saved to {output_file}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    
    print(f"Chain {chain_id}: Completed with {sample_count} total samples")
    return output_file


def run_chain(state: str, basepath: str, steps: int = 1000, pop_tol: float = 0.05,
              thinning: int = 10, n_chains: int = 3, conv_param: str = "EfficiencyGap", 
              burn_in: float = 0.0, include_geometry: bool = False, n_processes: int = None):
    
    if n_processes is None:
        n_processes = min(n_chains, mp.cpu_count())
    
    print(f"Running {n_chains} chains with {n_processes} parallel processes")
    print(f"Expected samples per chain: {int((steps - int(steps * burn_in)) / thinning)}")
    
    args_list = [
        (c, state, basepath, steps, pop_tol, thinning, burn_in, include_geometry)
        for c in range(n_chains)
    ]
    
    with mp.Pool(processes=n_processes) as pool:
        output_files = pool.map(run_single_chain, args_list)
    
    output_files = [f for f in output_files if f is not None]
    
    if not output_files:
        print("No chains completed successfully")
        return None
    
    print("Combining results from all chains...")
    all_results = []
    for file_path in output_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_results.append(df)
    
    if not all_results:
        print("No results found to combine")
        return None
    
    combined = pd.concat(all_results, ignore_index=True)
    
    R_hat = gelman_rubin(all_results, param=conv_param)
    print(f"Gelman-Rubin R_hat: {R_hat:.4f}")
    if R_hat < 1.1:
        print("Convergence reached")
    else:
        print("Chains may not have converged")
    
    out_file = f"{basepath}/{state}/ensemble_metrics.csv"
    combined.to_csv(out_file, index=False)
    print(f"Combined results saved to {out_file}")
    
    for file_path in output_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")
    
    return combined

if __name__ == "__main__":
    run_chain("az", "/home/arun/echo/thesis/data/processed", steps=300000, thinning=100, n_chains=4, burn_in=0.2, include_geometry=True)
