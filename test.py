if __name__ == '__main__':
    x =1
    y =1
    z =1
    n =2
    
    x_1 = [i for i in range(x+1)]
    y_1 = [i for i in range(y+1)]
    z_1 = [i for i in range(z+1)]
    all_comb = []
    for j in x_1:
        for i in y_1:
            for q in z_1:
                comb = [j, i, q]
                all_comb.append(comb)
    cp_comb = []
    print(all_comb)
    for i in all_comb:
        u = sum(i)
        if u != n:
            cp_comb.append(i)

    print(cp_comb)