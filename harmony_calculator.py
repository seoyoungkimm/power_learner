def harmony (w1, w2):
    eh1 = math.exp(w1)
    eh2 = math.exp(w2)
    sum_h = eh1 + eh2
    return eh1/sum_h, eh2/sum_h
