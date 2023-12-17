import numpy as np
from pathlib import Path

import loading
import generation

def wrapper(data_type, func, num_trials=5, improvement=False, **kwargs) -> None:
    """
    Generates data using func, saves to filepath, prints progress
    """
    n = kwargs["n"]
    d = kwargs["d"]

    filename = r"datasets/"

    if improvement:
        filename += r"improvement_datasets/"

    filename += r"{0}/{1}_n{2}_d{3}".format(data_type, data_type, n, d)

    if data_type == "uniform_ball_with_outliers":
        eta = kwargs["eta"]
        filename += "_eta{}".format(str(eta).replace(".","p"))
        
    for i in range(num_trials):

        print("Generating {} {}".format(filename, i))
        data = func(**kwargs)
        print("Saving {} {}".format(filename, i))
        loading.to_csv(data=data, filename=r"{0}_{1}.csv".format(filename, i))

    return None

if False:
    np.random.seed(seed=1234)
    n = 100000
    d = 1000

    wrapper(
        "normal",
        generation.normal,
        num_trials=1,
        mean=0,
        variance=1,
        n=n,
        dimension=d
    )
#===================================================

n_list = [1000 + 3000*i for i in range(10)]
d_list = [10 + 10*i for i in range(10)]

if False:
    np.random.seed(1234)
    for n in n_list:
        d = 30
        wrapper(
            "normal",
            generation.normal,
            n=n,
            d=d,
            mean=0,
            variance=1
        )
    
    for d in d_list:
        n = 10000
        wrapper(
            "normal",
            generation.normal,
            n=n,
            d=d,
            mean=0,
            variance=1
        )

if False:
    np.random.seed(1235)
    for n in n_list:
        d = 30
        wrapper(
            "uniform_ball",
            generation.uniform_ball,
            n=n,
            d=d,
            r=1,
            c=[0]*d
        )
    
    for d in d_list:
        n = 10000
        wrapper(
            "uniform_ball",
            generation.uniform_ball,
            n=n,
            d=d,
            r=1,
            c=[0]*d
        )

if False:
    np.random.seed(1236)
    for n in n_list:
        d = 30
        wrapper(
            "hyperspherical_shell",
            generation.hyperspherical_shell,
            n=n,
            d=d,
            r1=1,
            r2=2
        )
    
    for d in d_list:
        n = 10000
        wrapper(
            "hyperspherical_shell",
            generation.hyperspherical_shell,
            n=n,
            d=d,
            r1=1,
            r2=2
        )

eta_list = [0.5 + 0.1*i for i in range(5)]

if False:
    np.random.seed(1237)
    for n in n_list:
        d = 30
        eta = 0.9
        wrapper(
            "uniform_ball_with_outliers",
            generation.uniform_ball_with_ouliters,
            n=n,
            d=d,
            eta=eta,
            r=1,
            r1=2,
            r2=3
        )
    
    for d in d_list:
        n = 10000
        eta = 0.9
        wrapper(
            "uniform_ball_with_outliers",
            generation.uniform_ball_with_ouliters,
            n=n,
            d=d,
            eta=eta,
            r=1,
            r1=2,
            r2=3
        )

    for eta in eta_list:
        n = 10000
        d = 30
        wrapper(
            "uniform_ball_with_outliers",
            generation.uniform_ball_with_ouliters,
            n=n,
            d=d,
            eta=eta,
            r=1,
            r1=2,
            r2=3
        )

#===================================================
# Improvement data

n_list = [500+500*i for i in range(10)]
d_list = [10+10*i for i in range(15)]

if True:
    np.random.seed(5000)
    for n in n_list:
        d = 100
        wrapper(
            "normal",
            generation.normal,
            improvement=True,
            n=n,
            d=d,
            mean=0,
            variance=1
        )
    
    for d in d_list:
        n = 1000
        wrapper(
            "normal",
            generation.normal,
            improvement=True,
            n=n,
            d=d,
            mean=0,
            variance=1
        )

if True:
    np.random.seed(6000)
    for n in n_list:
        d = 100
        wrapper(
            "uniform_ball",
            generation.uniform_ball,
            improvement=True,
            n=n,
            d=d,
            r=1,
            c=[0]*d
        )
    
    for d in d_list:
        n = 1000
        wrapper(
            "uniform_ball",
            generation.uniform_ball,
            improvement=True,
            n=n,
            d=d,
            r=1,
            c=[0]*d
        )

if True:
    np.random.seed(7000)
    for n in n_list:
        d = 100
        wrapper(
            "hyperspherical_shell",
            generation.hyperspherical_shell,
            improvement=True,
            n=n,
            d=d,
            r1=1,
            r2=2
        )
    
    for d in d_list:
        n = 1000
        wrapper(
            "hyperspherical_shell",
            generation.hyperspherical_shell,
            improvement=True,
            n=n,
            d=d,
            r1=1,
            r2=2
        )

if True:
    np.random.seed(8000)
    eta = 0.9
    for n in n_list:
        d = 100
        wrapper(
            "uniform_ball_with_outliers",
            generation.uniform_ball_with_ouliters,
            improvement=True,
            n=n,
            d=d,
            eta=eta,
            r=1,
            r1=2,
            r2=3
        )
    
    for d in d_list:
        n = 1000
        wrapper(
            "uniform_ball_with_outliers",
            generation.uniform_ball_with_ouliters,
            improvement=True,
            n=n,
            d=d,
            eta=eta,
            r=1,
            r1=2,
            r2=3
        )