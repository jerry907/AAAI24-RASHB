from meb.ball import MEBwO
from benchmarking.utils import load_normal

n = 200
d = 8
eta = 0.9

data = load_normal(n,d)

exact = MEBwO().fit(data=data, method="exact", eta=eta)
shenmaier = MEBwO().fit(data=data, method="shenmaier", eta=eta)

print(shenmaier.radius/exact.radius)