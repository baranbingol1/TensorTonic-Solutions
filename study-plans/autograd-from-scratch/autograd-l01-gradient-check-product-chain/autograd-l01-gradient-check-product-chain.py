import numpy as np

def gradient_check_product_chain(a, b, c, f, h):
    """
    Returns: the loss, analytic gradients, numerical gradients, and maximum absolute disagreement
    """
    a, b, c, f, h = map(np.float64, (a, b, c, f, h))
    def calc_loss(a, b, c, f): return (a*b + c) * f
    dl_de = f
    dl_df = a*b + c
    de_da = b
    de_db = a
    de_dc = 1
    # analyticals
    dl_da = dl_de * de_da
    dl_db = dl_de * de_db
    dl_dc = dl_de * de_dc
    analytic_grads = [dl_da, dl_db, dl_dc, dl_df] 
    # numericals
    loss = calc_loss(a,b,c,f)
    dl_da_num = (calc_loss(a+h, b, c, f) - loss) / h
    dl_db_num = (calc_loss(a, b+h, c, f) - loss) / h
    dl_dc_num = (calc_loss(a, b, c+h, f) - loss) / h
    dl_df_num = (calc_loss(a, b, c, f+h) - loss) / h
    numerical_grads = [dl_da_num, dl_db_num, dl_dc_num, dl_df_num]

    max_disagreement = np.max(np.abs(np.asarray(numerical_grads) - np.asarray(analytic_grads)))
    return (float(loss), [float(x) for x in analytic_grads], [float(x) for x in numerical_grads], float(max_disagreement))