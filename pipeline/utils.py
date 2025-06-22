def display_results(report_df, roc_auc):
    print("\n=== Classification Report ===")
    print(report_df)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")