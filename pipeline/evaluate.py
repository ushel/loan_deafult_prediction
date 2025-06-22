from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"\nROC-AUC Score: {auc:.4f}")
    return report, auc