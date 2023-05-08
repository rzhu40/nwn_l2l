import numpy as np
import dask
from dask.distributed import Client, LocalCluster
from argparse import ArgumentParser, RawTextHelpFormatter
import os 
from run_MG import run_MG

def main():
    parser = ArgumentParser()
    
    parser.add_argument("--nworkers", type=int, 
                        required=False, default = 1,
                        help="Number of workers job in dask.")
    parser.add_argument("--tpw", type=int, 
                        required=False, default = 1,
                        help="Number of threads per worker. Local only!")
    parser.add_argument("--cpt", type=int, 
                        required=False, default = 1,
                        help="Number of cores per thread. Local only!")
    
    args = parser.parse_args()
    
    os.environ['OMP_NUM_THREADS'] = str(args.cpt)
    os.environ['MKL_NUM_THREADS'] = str(args.cpt)
    cluster = LocalCluster(
                processes = True,
                n_workers = args.nworkers,
                threads_per_worker = args.tpw,
                scheduler_port = 12121,
                dashboard_address = 'localhost:11113',
                )
    client = Client(cluster)
    
    amp_list = np.arange(0,5,2)
    b_list   = np.arange(0,2.1,0.5)

    job_pool = []
    for amp in amp_list:
        for b in b_list:
            eval = dask.delayed(run_MG)(amp, b)
            job_pool.append(eval)

    from_dask = client.compute(job_pool)
    collected = client.gather(from_dask)
    print(collected)
if __name__ == "__main__":
    main()
    