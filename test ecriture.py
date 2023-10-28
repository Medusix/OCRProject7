list_cols = ['A', 'B', 'C']
with open('cols.txt', 'w') as fp:
    for col in list_cols:
        # write each item on a new line
        fp.write("%s\n" % col)
        