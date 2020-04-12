def get_basis(basisname):
    bs_dir = "/home/masaya/PycharmProjects/mscf/mscf/basis/"
    path = bs_dir + basisname

    with open(path, 'r') as f:
        l = [s.strip() for s in f.readlines()]
    basis = dict()
    flag = False
    i = -1
    while i + 1 < len(l):
        i += 1
        if l[i] == 'BASIS "ao basis" PRINT':
            flag = True
            continue
        if not flag or l[i] == "END":
            continue
        if l[i].startswith("#"):
            i += 1
        symbol = l[i].split()
        if not symbol[0] in basis.keys():
            basis[symbol[0]] = []

        if symbol[1] == "S":
            basis[symbol[0]].append([symbol[1], [], []])
        elif symbol[1] == "SP":
            basis[symbol[0]].append([symbol[1], [], [], []])

        while i < len(l):
            i += 1
            l1 = l[i].split()
            if l1[0].startswith("#") or l1[0] == "END" or l1[0].isalpha():
                i -= 1
                break
            for j, s in enumerate(l[i].split()):
                basis[symbol[0]][-1][j + 1].append(float(s))
    return basis

