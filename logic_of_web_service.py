def logic_web(patient,dv,model): # this is our core logic for our program that we are serving using flask
    X = dv.transform([patient])  # note the dictvectorizer expects a list of dictionaries , that's why we put the customer in a list
    y_pred = model.predict_proba(X)[:, 1]
    stroke_risk = y_pred >= 0.5

    return y_pred , stroke_risk