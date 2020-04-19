mol = Mole([['I', 0, 0, -0.7], ['I', 0, 0, +0.7]], 'sto3g')
start = time()
#v1 = get_v2e(mol)
X = 0.52918  # 単位変換: angstrom -> a0
m = pyscf.gto.Mole()
m.build(atom='I 0 0 %f; I 0 0 %f' % (-0.7 * X, 0.7 * X),
                  basis="sto3g")
v1 = m.intor('int2e')

print("time1: ", time() - start)
print(v1.shape)
#print(v1)
#print(v1)
start = time()
v2 = c_get_v2e(mol)
print("time2: ",time() - start)
#print(v2)
ans = 0
l = len(v1)
rel_tol = 0
abs_tol = 0
rel_mean = 0
vol = 0

for i in range(l):
    for j in range(l):
        for k in range(l):
            for l in range(l):
                a = v1[i][j][k][l]
                b = v2[i][j][k][l]
                if abs(a) >= 1e-10 and abs(b) >= 1e-10:
                    vol += 1
                    rel1 = abs(a-b)/max(abs(a), abs(b))
                    rel_tol = max(rel_tol,rel1)
                    rel_mean += -math.log10(rel1)
                else:
                    abs1 = abs(a-b)
                    abs_tol = max(abs_tol, abs1)
                ans += abs(a-b)

"""
for i in range(l):
    for j in range(l):
        a = v1[i][j]
        b = v2[i][j]
        if abs(a) >= 1e-10 or abs(b) >= 1e-10:
            vol += 1
            rel1 = abs(a-b)/max(abs(a), abs(b))
            rel_tol = max(rel_tol,rel1)
            rel_mean += -math.log10(rel1)
        else:
            abs1 = abs(a-b)
            abs_tol = max(abs_tol, abs1)
        ans += abs(a-b)
"""
print("ans_sum: ",ans)
print("rel_max: ", rel_tol)
print("abs_max: ", abs_tol)
print("not 0 number: ", vol)
print("rel_mean: ",rel_mean/vol)
