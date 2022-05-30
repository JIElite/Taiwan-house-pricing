from sklearn.metrics import (r2_score, 
                             mean_absolute_error,
                             mean_squared_error)
    

def simple_evaluate(model, X, y, verbose=False):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    if verbose:
        print(f'R2 score: {r2}')
        print(f'MAE score: {mae}')
        print(f'MSE score: {mse}')

    return r2, mae, mse