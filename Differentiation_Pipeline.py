# %% GitHub Repo Code
# Checking Python Version
import sys
print(sys.version)

# LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn import metrics
from sklearn.base import clone
from sklearn.utils import resample
import shap
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer 
from matplotlib.colors import ListedColormap
from sklearn.calibration import CalibratedClassifierCV


print("Libraries Imported Successfully!")


# DATASET
pxpm_cohort = pd.read_csv("read_in_dataset.csv")
print(len(pxpm_cohort))

control = pxpm_cohort["types"] == 0
case = pxpm_cohort["types"] == 1

px_cohort = pxpm_cohort[control]
pm_cohort = pxpm_cohort[case]


#ML REQUIRES DIFFERENT MODELS
    #GDF15 - ASSESS THE PREDICTIVE ABILITY OF GDF15 ALONE
    #CLINICAL FEATURES ONLY - ESTABLISH A BASELINE MODEL (CLINICAL FEATURES)
    #PROTEIN ONLY MODEL - ASSESS PREDICTIVE ABILITY OF PROTEINS ALONE (PROTEINS)
    #FEATURE SELECTED - UNRESTRICTED ACCESS TO THE DATASET (CLINICAL AND PROTEINS), ARE COMBINATION OF FEATURES SELECTED AND HOW DOES THIS COMPARE TO PREVIOUS MODELS



# Unhash this block when running GDF15 analysis #
"""
###     GDF15 UNIVARIATE ANALYSIS     ###
X = pxpm_cohort[["GDF15"]]
y = pxpm_cohort["types"]
"""


# Unhash this block when running clinical model #
"""
###     CLINICAL BASELINE MODEL     ###
Contin_CV = pxpm_cohort.loc[:, ["age", "weight(kg)", "height(cm)", "BMI", "systolic", "diastolic"]]
Categor_CV = pxpm_cohort.loc[:, ["gender", "smoke", "alcohol", "com_hypertension_2", "diabetes", "com_heart_failure", "com_infarction", "com_vascular", "com_stroke", "covid"]].astype("category")

Clinical_Variables = pd.concat([Contin_CV, Categor_CV], axis = 1)
Clinical_Variables_Cols = Clinical_Variables.columns.tolist()

X = Clinical_Variables
y = pxpm_cohort["types"]
"""


# Unhash this block when running protein model #
"""
###     PROTEIN MODEL     ###
Proteins = pxpm_cohort.loc[:, "NPPB":"TIMP1"]

X = Proteins
y = pxpm_cohort["types"]
"""


# Unhash this block when running feature selected model #
"""
###     CLINICAL + PROTEIN MODEL     ###
Contin_CV = pxpm_cohort.loc[:, ["age", "weight(kg)", "height(cm)", "BMI", "systolic", "diastolic"]]
Categor_CV = pxpm_cohort.loc[:, ["gender", "smoke", "alcohol", "com_hypertension_2", "diabetes", "com_heart_failure", "com_infarction", "com_vascular", "com_stroke", "covid"]].astype("category")
Proteins = pxpm_cohort.loc[:, "NPPB":"TIMP1"]

Clinical_Variables = pd.concat([Contin_CV, Categor_CV], axis = 1)
Clinical_Variables_Cols = Clinical_Variables.columns.tolist()

X = pd.concat([Clinical_Variables, Proteins], axis = 1)
y = pxpm_cohort["types"]
"""


# Train-test split and cross validation definition is identical for every model ran (note: ensure each time the code is ran, X_test.index is identical for every run) #
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.25, stratify = y)

print(X_test.index)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#Defining Cross Validation
sKf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)


# Run this block with the above GDF15 block defining X and y #
"""
# RUN THIS SECTION WHEN GDF15 MODEL IS NEEDED #
results = []

models = {
    "LR": {
        "model":
        LogisticRegression(random_state = 42),
    "param_grid": {
        "LR__C": [0.001, 0.01, 0.1, 1.0, 10], 
        "LR__penalty": ["l1", "l2"],
        "LR__solver": ["liblinear"],
        "LR__max_iter": [1000, 3000, 5000]}
    },
    "RF": {
        "model":
        RandomForestClassifier(random_state = 42),
    "param_grid": {
        "RF__n_estimators": [100, 200, 300, 500],
        "RF__max_depth": [None, 5, 8], 
        "RF__min_samples_split": [2, 3, 4, 5]}
    },
    "SVM_L": {
        "model": 
         CalibratedClassifierCV(
            SVC(probability = False, random_state = 42),
            method = "sigmoid",
            cv = sKf,
            ensemble = False),
        "param_grid": {
            "SVM_L__estimator__C": [0.001, 0.01, 0.1, 1, 10],
            "SVM_L__estimator__kernel": ["linear"]} 
    }  
}         


for name, info in models.items():

    preprocess = ColumnTransformer([
        ("contin", Pipeline([
            ("scaler", StandardScaler())
        ]), make_column_selector(dtype_include = ["int64", "float64"]))
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocess),
        (name, info["model"])
    ])
    
    #Using Gridsearch
    grid = GridSearchCV(
        pipeline,
        info["param_grid"],
        cv = sKf,
        scoring = "roc_auc",
        refit = True,
        return_train_score = True
        )

    grid.fit(X_train, y_train)
    
    grid_results = pd.DataFrame(grid.cv_results_)

    # BOX PLOTS OF GDF15 EXPRESSION IN TRAIN AND TEST COHORT #
    train_boxplot = pxpm_cohort.loc[X_train.index]

    GDF15_train_boxplot = train_boxplot.loc[:, ["types", "GDF15"]]
    GDF15_train_boxplot["types"] = GDF15_train_boxplot["types"].map({0: "Paroxysmal AF", 1: "Permanent AF"})


    test_boxplot = pxpm_cohort.loc[X_test.index]

    GDF15_test_boxplot = test_boxplot.loc[:, ["types", "GDF15"]]
    GDF15_test_boxplot["types"] = GDF15_test_boxplot["types"].map({0: "Paroxysmal AF", 1: "Permanent AF"})

    sns.boxplot(data = GDF15_train_boxplot,
                x = "types",
                y = "GDF15",
                palette = {"Paroxysmal AF": "blue", "Permanent AF": "red"},
                width = 0.15)

    plt.ylim(-2.5, 3.5)
    plt.grid(alpha = 0.2)
    plt.title(f"GDF15 Expression Values ({name} - Train Data)")
    plt.savefig(f"GDF15_Expression_{name}_GDF15_Train_Data.jpg", dpi = 300)
    plt.show()


    sns.boxplot(data = GDF15_test_boxplot,
                x = "types",
                y = "GDF15",
                order = ["Paroxysmal AF", "Permanent AF"],
                palette = {"Paroxysmal AF": "blue", "Permanent AF": "red"},
                width = 0.15)

    plt.ylim(-2.5, 3.5)
    plt.grid(alpha = 0.2)
    plt.title(f"GDF15 Expression Values ({name} - Test Data)")
    plt.savefig(f"GDF15_Expression_{name}_GDF15_TestData.jpg", dpi = 300)
    plt.show() 



    # TRAINING DATA PREDICTIONS #
    train_prob = grid.predict_proba(X_train)[:, 1]
    train_pred_binary = (train_prob >= 0.5).astype(int)  
  
    train_fpr, train_tpr, _ = roc_curve(y_train, train_prob, pos_label = 1)
    train_auroc = auc(train_fpr, train_tpr)
    print("Training AUC Value: ", train_auroc)

    # TEST DATA PREDICTIONS #
    test_prob = grid.predict_proba(X_test)[:, 1]
    test_pred_binary = (test_prob >= 0.5).astype(int)
  
    test_fpr, test_tpr, _ = roc_curve(y_test, test_prob, pos_label = 1)
    test_auroc = auc(test_fpr, test_tpr)
    print("Test AUC Value: ", test_auroc)
    
    results.append({
        "Model": name,
        "Best Parameter Combination": grid.best_params_, 
        "CV Parameters": grid.cv_results_["params"],
        "Mean CV AUC": grid.best_score_,
        "Train AUC Value": train_auroc,
        "Test AUC": test_auroc
        })
    results_df = pd.DataFrame(results)

    # CONFUSION MATRIX OF TRAINING DATA #
    train_confmatrix = confusion_matrix(y_train, train_pred_binary)
    flip_traincm = train_confmatrix[:, [1,0]]
    flipped_traincm = flip_traincm[[1,0], :]
    flipped_traincm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = flipped_traincm,
        display_labels = [1, 0])
    flipped_traincm_display.plot(cmap = plt.cm.Blues, text_kw = {"fontsize": 18})
    plt.title(f"{name} Confusion Matrix of Train Data (GDF15)")
    plt.savefig(f"ConfusionMatrix_{name}_TrainData_GDF15.jpg", dpi = 300)
    plt.show()

    
    # CONFUSION MATRIX OF TEST DATA #
    test_confmatrix = confusion_matrix(y_test, test_pred_binary)
    flip_testcm = test_confmatrix[:, [1, 0]]
    flipped_testcm = flip_testcm[[1,0], :]
    flipped_testcm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = flipped_testcm,
        display_labels = [1, 0])
    flipped_testcm_display.plot(cmap = plt.cm.Blues, text_kw = {"fontsize": 18})
    plt.title(f"{name} Confusion Matrix of Test Data (GDF15)")
    plt.savefig(f"ConfusionMatrix_{name}_TestData_GDF15.jpg", dpi = 300)
    plt.show()

    
    # ROC CURVE OF TRAINING DATA #
    plt.figure()
    plt.plot(train_fpr, train_tpr, label = "ROC Curve (area = %0.2f)" % train_auroc, color = "royalblue")
    plt.plot([0.0, 1.0], [0.0, 1.0], label = "No Confidence", color = "r")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC Curve of {name} (Train Data, GDF15)")
    plt.legend()
    plt.savefig(f"ROCCurve_{name}_GDF15_TrainingData.jpg", dpi = 300)
    plt.show()


    # ROC CURVE OF TEST DATA #
    n_iterations = 1000
    boot_auc_scores = []
    boot_fpr = []
    boot_tpr = []
    interpolated_tpr_list = []
    test_samples = np.arange(len(y_test))
    
    for i in range(n_iterations):
        samples = resample(test_samples, replace = True)
        
        X_samples = X_test.iloc[samples]
        y_samples = y_test[samples]
        
        boot_prob = grid.predict_proba(X_samples)[:, 1]        
        boot_auroc = roc_auc_score(y_samples, boot_prob)
        boot_auc_scores.append(boot_auroc)
        
        fpr_iter, tpr_iter, _ = roc_curve(y_samples, boot_prob, pos_label = 1)
        boot_fpr.append(fpr_iter)
        boot_tpr.append(tpr_iter)
        
    fpr_common = np.linspace(0, 1, num = 500, endpoint = True)

    for i in range(n_iterations):
        interpolated_tpr = np.interp(fpr_common, boot_fpr[i], boot_tpr[i])
        interpolated_tpr_list.append(interpolated_tpr)
        
    interpolated_tpr_array = np.array(interpolated_tpr_list)
    
    lower_percentile = np.percentile(interpolated_tpr_array, 2.5, axis = 0)
    upper_percentile = np.percentile(interpolated_tpr_array, 97.5, axis = 0)
    
    plt.figure()
    plt.fill_between(fpr_common, lower_percentile, upper_percentile, color = "lightskyblue", alpha = 0.75, label = "Bootstrapped 95% CI")
    plt.plot(test_fpr, test_tpr, label = "ROC Curve (area = %0.2f)" % test_auroc, color = "royalblue")
    plt.plot([0.0, 1.0], [0.0, 1.0], label = "No Confidence", color = "r")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC Curve of {name} (Test Data, GDF15)")
    plt.legend(loc = "lower right")
    plt.savefig(f"ROCCurve_{name}_GDF15_TestData.jpg", dpi = 300)
    plt.show()


# END OF GDF15 MODEL #
"""




"""
# HISTOGRAM OF TRAIN AND TEST DATA (GDF15) # 
X = pxpm_cohort.loc[:, ["types", "GDF15"]]
y = pxpm_cohort["types"]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.25, stratify = y)

# Ensure identical patients as original train-test split #
print(X_test.index)


# HISTOGRAMS OF TRAIN AND TEST DATA #

#Defining Train test subtypes - Train data
px_train = X_train[X_train["types"] == 0]

pm_train = X_train[X_train["types"] == 1]

#Defining Train test subtypes - Test data
px_test = X_test[X_test["types"] == 0]

pm_test = X_test[X_test["types"] == 1]


# TRAIN DATA # 
plt.hist(px_train["GDF15"], label = "Paroxysmal", color = "blue", alpha = 0.65)
plt.hist(pm_train["GDF15"], label = "Permanent", color = "red", alpha = 0.5)
plt.xlabel("GDF15 Expression")
plt.ylabel("Number of Patients")
plt.legend(loc = "upper right")
plt.title("GDF15 Expression of Training Data")
plt.savefig(f"GDF15Expression_TrainingData.jpg", dpi = 300)
plt.show()



# TEST DATA #
plt.hist(px_test["GDF15"], label = "Paroxysmal", color = "blue", alpha = 0.65)
plt.hist(pm_test["GDF15"], label = "Permanent", color = "red", alpha = 0.5)
plt.xlabel("GDF15 Expression")
plt.ylabel("Number of Patients")
plt.legend(loc = "upper right")
plt.title("GDF15 Expression of Test Data")
plt.savefig(f"GDF15Expression_TestData.jpg", dpi = 300)
plt.show()
"""





# Run this block with the above GDF15 block defining X and y #
"""
# RUN THIS SECTION WHEN POOLED GDF15 MODEL IS NEEDED (LR AND SVM_L)
t = np.arange(len(X))

# Ensure Original_Test_Fold is identical to the test fold from train-test split #
Original_Test_Fold = np.where(X.index.isin(X_test.index))[0]   

Remaining_Patients = np.setdiff1d(np.arange(len(X)), Original_Test_Fold)

Fold_sKf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)

All_Patients_Folds = [Original_Test_Fold] + [Remaining_Patients[t] for _, t in Fold_sKf.split(Remaining_Patients, y.iloc[Remaining_Patients])]

sKf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

models = {
    "LR": {
        "model":
        LogisticRegression(random_state = 42),
    "param_grid": {
        "LR__C": [0.001, 0.01, 0.1, 1.0, 10], 
        "LR__penalty": ["l1", "l2"],
        "LR__solver": ["liblinear"],
        "LR__max_iter": [1000, 3000, 5000]}
    },
    "SVM_L": {
        "model": 
         CalibratedClassifierCV(
            SVC(probability = False, random_state = 42),
            method = "sigmoid",
            cv = sKf,
            ensemble = False),
        "param_grid": {
            "SVM_L__estimator__C": [0.001, 0.01, 0.1, 1, 10],
            "SVM_L__estimator__kernel": ["linear"]} 
    }  
}      

for name, info in models.items():
    results = []
    fold = []
    
    for l, test_val in enumerate(All_Patients_Folds):
        train_val = np.setdiff1d(t, test_val)    
    
        preprocess = ColumnTransformer([
            ("contin", Pipeline([
                ("scaler", StandardScaler())
            ]), make_column_selector(dtype_include = ["int64", "float64"]))
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocess),
            (name, info["model"])
        ])
        
        #Using Gridsearch
        grid = GridSearchCV(
            pipeline,
            info["param_grid"],
            cv = sKf,
            scoring = "roc_auc",
            refit = True,
            return_train_score = True
            )

        grid.fit(X.iloc[train_val], y.iloc[train_val])   
                

        # TRAINING DATA PREDICTIONS #
        train_prob = grid.predict_proba(X.iloc[train_val])[:, 1]
        train_pred_binary = (train_prob >= 0.5).astype(int)  
      
        train_fpr, train_tpr, _ = roc_curve(y.iloc[train_val], train_prob, pos_label = 1)
        train_auroc = auc(train_fpr, train_tpr)
        print("Training AUC Value: ", train_auroc)

        # TEST DATA PREDICTIONS #
        test_prob = grid.predict_proba(X.iloc[test_val])[:, 1]
        test_pred_binary = (test_prob >= 0.5).astype(int)
      
        test_fpr, test_tpr, _ = roc_curve(y.iloc[test_val], test_prob, pos_label = 1)
        test_auroc = auc(test_fpr, test_tpr)
        print("Test AUC Value: ", test_auroc)
        

        # CONFUSION MATRIX OF TRAINING DATA #
        train_confmatrix = confusion_matrix(y.iloc[train_val], train_pred_binary)
        flip_traincm = train_confmatrix[:, [1,0]]
        flipped_traincm = flip_traincm[[1,0], :]
        flipped_traincm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix = flipped_traincm,
            display_labels = [1, 0])
        flipped_traincm_display.plot(cmap = plt.cm.Blues, text_kw = {"fontsize": 18})
        plt.title(f"{name} Confusion Matrix of Training Folds (GDF15)")
        plt.savefig(f"ConfusionMatrix_{name}_TrainData_POOLED_GDF15_{l+1}.jpg", dpi = 300)
        plt.show()

        
        # CONFUSION MATRIX OF TEST DATA #
        test_confmatrix = confusion_matrix(y.iloc[test_val], test_pred_binary)
        flip_testcm = test_confmatrix[:, [1, 0]]
        flipped_testcm = flip_testcm[[1,0], :]
        flipped_testcm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix = flipped_testcm,
            display_labels = [1, 0])
        flipped_testcm_display.plot(cmap = plt.cm.Blues, text_kw = {"fontsize": 18})
        plt.title(f"{name} Confusion Matrix of Test Fold {l + 1} (GDF15)")
        plt.savefig(f"ConfusionMatrix_{name}_TestData_POOLED_GDF15_{l+1}.jpg", dpi = 300)
        plt.show()

        
        # ROC CURVE OF TRAINING DATA #
        plt.figure()
        plt.plot(train_fpr, train_tpr, label = "ROC Curve (area = %0.2f)" % train_auroc, color = "royalblue")
        plt.plot([0.0, 1.0], [0.0, 1.0], label = "No Confidence", color = "r")
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.title(f"ROC Curve of {name} (Training Folds, GDF15)")
        plt.legend(loc = "lower right")
        plt.savefig(f"ROCCurve_{name}_POOLEDGDF15_TrainingData_{l+1}.jpg", dpi = 300)
        plt.show()
        
        # ROC CURVE OF TEST DATA #
        n_iterations = 1000
        boot_auc_scores = []
        boot_fpr = []
        boot_tpr = []
        interpolated_tpr_list = []
        test_samples = np.arange(len(y.iloc[test_val]))
        
        for i in range(n_iterations):
            samples = resample(test_samples, replace = True)
            
            X_samples = X.iloc[test_val].iloc[samples]
            y_samples = y.iloc[test_val].iloc[samples]
            
            boot_prob = grid.predict_proba(X_samples)[:, 1]        
            boot_auroc = roc_auc_score(y_samples, boot_prob)
            boot_auc_scores.append(boot_auroc)
            
            fpr_iter, tpr_iter, _ = roc_curve(y_samples, boot_prob, pos_label = 1)
            boot_fpr.append(fpr_iter)
            boot_tpr.append(tpr_iter)
            
        fpr_common = np.linspace(0, 1, num = 500, endpoint = True)
    
        for i in range(n_iterations):
            interpolated_tpr = np.interp(fpr_common, boot_fpr[i], boot_tpr[i])
            interpolated_tpr_list.append(interpolated_tpr)
            
        interpolated_tpr_array = np.array(interpolated_tpr_list)
        
        lower_percentile = np.percentile(interpolated_tpr_array, 2.5, axis = 0)
        upper_percentile = np.percentile(interpolated_tpr_array, 97.5, axis = 0)
        
        plt.figure()
        plt.fill_between(fpr_common, lower_percentile, upper_percentile, color = "lightskyblue", alpha = 0.75, label = "Bootstrapped 95% CI")
        plt.plot(test_fpr, test_tpr, label = "ROC Curve (area = %0.2f)" % test_auroc, color = "royalblue")
        plt.plot([0.0, 1.0], [0.0, 1.0], label = "No Confidence", color = "r")
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.title(f"ROC Curve of {name} (Test Fold {l + 1}, GDF15)")
        plt.legend(loc = "lower right")
        plt.savefig(f"ROCCurve_{name}_POOLED_GDF15_TestData_{l+1}.jpg", dpi = 300)
        plt.show()
         
        
        fold_df = pd.DataFrame({
                    "fold": l + 1,
                    "Patient Index": X.index[test_val],
                    "True Label": y.iloc[test_val],
                    "Predicted Probability": test_prob,
                    "Binary Predictions": (test_prob >= 0.5).astype(int)
                })
                
        fold.append(fold_df)
                    
        all_folds = pd.concat(fold, ignore_index = True)        
    
        
    all_y_test = all_folds["True Label"].to_numpy()
    all_test_prob = all_folds["Predicted Probability"].to_numpy()
    
    all_test_pred_binary = (all_test_prob >= 0.5).astype(int)
  
    all_test_fpr, all_test_tpr, _ = roc_curve(all_y_test, all_test_prob, pos_label = 1)
    all_test_auroc = auc(all_test_fpr, all_test_tpr)
    
    
    pooled_confmatrix = confusion_matrix(all_y_test, all_test_pred_binary)
    flip_testcm = pooled_confmatrix[:, [1, 0]]
    flipped_testcm = flip_testcm[[1,0], :]
    flipped_testcm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = flipped_testcm,
        display_labels = [1, 0])
    flipped_testcm_display.plot(cmap = plt.cm.Blues, text_kw = {"fontsize": 18})
    plt.title(f"{name} Confusion Matrix of Pooled Test Data (GDF15)")
    plt.savefig(f"ConfusionMatrix_{name}_TestData_POOLED_GDF15.jpg", dpi = 300)
    plt.show()

    
    # ROC CURVE OF POOLED TEST DATA #
    pooled_n_iterations = 1000
    pooled_boot_auc_scores = []
    pooled_boot_fpr = []
    pooled_boot_tpr = []
    pooled_interpolated_tpr_list = []
    pooled_test_samples = np.arange(len(all_y_test))
    
    for i in range(pooled_n_iterations):
        pooled_samples = resample(pooled_test_samples, replace = True)
        
        pooled_y_samples = all_y_test[pooled_samples]
        
        pooled_boot_prob = all_test_prob[pooled_samples]        
        pooled_boot_auroc = roc_auc_score(pooled_y_samples, pooled_boot_prob)
        pooled_boot_auc_scores.append(pooled_boot_auroc)
        
        pooled_fpr_iter, pooled_tpr_iter, _ = roc_curve(pooled_y_samples, pooled_boot_prob, pos_label = 1)
        pooled_boot_fpr.append(pooled_fpr_iter)
        pooled_boot_tpr.append(pooled_tpr_iter)
        
    pooled_fpr_common = np.linspace(0, 1, num = 500, endpoint = True)

    for i in range(pooled_n_iterations):
        pooled_interpolated_tpr = np.interp(pooled_fpr_common, pooled_boot_fpr[i], pooled_boot_tpr[i])
        pooled_interpolated_tpr_list.append(pooled_interpolated_tpr)
        
    pooled_interpolated_tpr_array = np.array(pooled_interpolated_tpr_list)
    
    pooled_lower_percentile = np.percentile(pooled_interpolated_tpr_array, 2.5, axis = 0)
    pooled_upper_percentile = np.percentile(pooled_interpolated_tpr_array, 97.5, axis = 0)
    
    plt.figure()
    plt.fill_between(pooled_fpr_common, pooled_lower_percentile, pooled_upper_percentile, color = "lightskyblue", alpha = 0.75, label = "Bootstrapped 95% CI")
    plt.plot(all_test_fpr, all_test_tpr, label = "ROC Curve (area = %0.2f)" % all_test_auroc, color = "royalblue")
    plt.plot([0.0, 1.0], [0.0, 1.0], label = "No Confidence", color = "r")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC Curve of {name} (Pooled Test Data, GDF15)")
    plt.legend()
    plt.savefig(f"ROCCurve_{name}_POOLED_GDF15_TestData.jpg", dpi = 300)
    plt.show()


# END OF POOLED GDF15 MODEL #
"""









# Run this block with the above clinical block defining X and y #
"""
# RUN THIS SECTION WHEN CLINICAL MODEL IS NEEDED #
###     CLINICAL VALUES      ###

results = []

models = {
    "LR": {
        "model":
            LogisticRegression(random_state = 42),
        "param_grid": {
            "LR__C": [0.001, 0.01, 0.1, 1.0, 10],
            "LR__penalty": ["l1", "l2"],
            "LR__solver": ["liblinear"],
            "LR__max_iter": [1000, 3000, 5000]}
    },
    "RF": {
        "model": 
            RandomForestClassifier(random_state = 42),
        "param_grid": {
            # "RF__n_estimators": [100, 200, 300, 500],
            "RF__n_estimators": [80, 90, 100, 150, 200],     #TUNING1
            # "RF__max_depth": [None, 5, 8],
            "RF__max_depth": [None, 5, 6, 8, 10, 12, 14],    #TUNING1
            "RF__min_samples_split": [2, 3, 4, 5]}
    },
    "SVM_L": {
        "model":
        CalibratedClassifierCV(
            SVC(random_state = 42, probability = True),
            method = "sigmoid",
            cv = sKf,
            ensemble = False),
    "param_grid":{
        "SVM_L__estimator__C": [0.001, 0.01, 0.1, 1.0, 10], 
        "SVM_L__estimator__kernel": ["linear"]}
    },
    "SVM_RBF": {
        "model":
        CalibratedClassifierCV(
            SVC(random_state = 42, probability = True),
            method = "sigmoid",
            cv = sKf,
            ensemble = False),
    "param_grid":{
        "SVM_RBF__estimator__C": [0.01, 0.1, 1.0, 10],
        "SVM_RBF__estimator__gamma": [0.001, 0.01, 0.1, 1.0, 1.0],
        "SVM_RBF__estimator__kernel": ["rbf"]}
    }                                                        
}


print("Models Created Successfully!")

for name, info in models.items():
    selected_feats = []
    remaining_feats = list(X.columns)
    trace = []
    
    for step in range (1,8):
        step_best_feat = None
        step_best_AUC = -np.inf
        step_best_params = []
        step_best_acc = []
        
        for candidate_feat in remaining_feats:
            current_feats = selected_feats + [candidate_feat]
            
            step_X_train = X_train[current_feats]
            
            preprocess = ColumnTransformer([
                ("contin", Pipeline([
                    ("imputer", IterativeImputer(max_iter = 25, random_state = 42)),
                    ("scaler", StandardScaler())
                ]), make_column_selector(dtype_include = ["int64", "float64"])),
                ("categ", Pipeline([
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ]), make_column_selector(dtype_include = ["category"])
                 )
            ])

            pipeline = Pipeline([
                ("preprocess", preprocess),
                (name, info["model"])
        ])
            
            
            grid = GridSearchCV(
                pipeline,
                info["param_grid"],
                cv = sKf,
                scoring = "roc_auc",
                return_train_score = True,
                refit = True,
                error_score = "raise",
                n_jobs = -1
            )

            grid.fit(step_X_train, y_train)

            grid_CV_Res = pd.DataFrame(grid.cv_results_)

            mean_cv_AUC = grid.best_score_
            
            if mean_cv_AUC > step_best_AUC:
                step_best_feat = candidate_feat
                step_best_AUC = mean_cv_AUC
                step_best_params = grid.best_params_
        selected_feats.append(step_best_feat)
        remaining_feats.remove(step_best_feat)
        print(f"new feature {step_best_feat} added!")
        
        trace.append({
            "Step": step,
            "Selected Feature": step_best_feat,
            "Features Selected": selected_feats.copy(),
            "AUC": step_best_AUC,
            "Parameters": step_best_params
        })
    trace_results = pd.DataFrame(trace)
    
    step_X_train = X_train[selected_feats]
    step_X_test = X_test[selected_feats]
        
    cleaned_params = {
        parameter.removeprefix(f"{name}__"): value
        for parameter, value in step_best_params.items()
    }
    
    refit_pipeline = Pipeline([
        ("preprocessor", preprocess),
        (name, info["model"].set_params(**cleaned_params))
    ])
    
    training_all = refit_pipeline.fit(step_X_train, y_train)
    

    ### TRAINING DATA PREDICTIONS ###
    train_prob = training_all.predict_proba(step_X_train)[:, 1]
    train_pred_binary = (train_prob >= 0.5).astype(int)    
    
    train_fpr, train_tpr, _ = roc_curve(y_train, train_prob, pos_label = 1)
    train_auroc = auc(train_fpr, train_tpr)
    print("Training AUC Value: ", train_auroc)

    ### TEST DATA PREDICTIONS ###
    test_prob = training_all.predict_proba(step_X_test)[:, 1]
    test_pred_binary = (test_prob >= 0.5).astype(int)
    
    test_fpr, test_tpr, _ = roc_curve(y_test, test_prob, pos_label = 1)
    test_auroc = auc(test_fpr, test_tpr)
    print("Test AUC Value: ", test_auroc)
    
    results.append({
        "Model": name,
        "Feature(s)": selected_feats,
        "Best Parameter Combination": step_best_params,
        "Train AUC Value": train_auroc,
        "Test AUC": test_auroc,
        "Train - Test AUC": train_auroc - test_auroc})
    
    results_df = pd.DataFrame(results)
    
    # CONFUSION MATRIX OF TRAINING DATA #
    train_confmatrix = confusion_matrix(y_train, train_pred_binary)
    flip_traincm = train_confmatrix[:, [1,0]]
    flipped_traincm = flip_traincm[[1,0], :]
    flipped_traincm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = flipped_traincm,
        display_labels = [1, 0])
    flipped_traincm_display.plot(cmap = plt.cm.Blues, text_kw = {"fontsize": 18})
    plt.title(f"{name} Confusion Matrix of Train Data (Clinical Features)")
    plt.savefig(f"ConfusionMatrix_{name}_TrainData_clinfeats.jpg", dpi = 300)
    plt.show()
        
    # CONFUSION MATRIX OF TEST DATA #
    test_confmatrix = confusion_matrix(y_test, test_pred_binary)
    flip_testcm = test_confmatrix[:, [1, 0]]
    flipped_testcm = flip_testcm[[1,0], :]
    flipped_testcm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = flipped_testcm,
        display_labels = [1, 0])
    flipped_testcm_display.plot(cmap = plt.cm.Blues, text_kw = {"fontsize": 18})
    plt.title(f"{name} Confusion Matrix of Test Data (Clinical Features)")
    plt.savefig(f"ConfusionMatrix_{name}_TestData_clinfeats.jpg", dpi = 300)
    plt.show()

    # ROC CURVE OF TRAINING DATA #
    plt.figure()
    plt.plot(train_fpr, train_tpr, label = "ROC Curve (area = %0.2f)" % train_auroc, color = "royalblue")
    plt.plot([0.0, 1.0], [0.0, 1.0], label = "No Confidence", color = "r")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC Curve of {name} (Training Data, Clinical Features)")
    plt.legend()
    plt.savefig(f"ROCCurve_{name}_TrainingData_clinfeats.jpg", dpi = 300)
    plt.show()

    # ROC CURVE OF TEST DATA #
    # BOOTSTRAPPING DATA #
    n_iterations = 1000

    boot_auc_scores = []
    boot_fpr = []
    boot_tpr = []
    interpolated_tpr_list = []

    test_samples = np.arange(len(y_test))

    for i in range(n_iterations):
        samples = resample(test_samples, replace = True)
        X_samples = step_X_test.iloc[samples]
        y_samples = y_test[samples]

        boot_prob = training_all.predict_proba(X_samples)[:, 1]
        boot_auroc = roc_auc_score(y_samples, boot_prob)
        boot_auc_scores.append(boot_auroc)
        
        fpr_iter, tpr_iter, _ = roc_curve(y_samples, boot_prob, pos_label = 1)
        boot_fpr.append(fpr_iter)
        boot_tpr.append(tpr_iter)
        
    fpr_common = np.linspace(0, 1, num = 500, endpoint = True)
        
    for i in range(n_iterations):
        interpolated_tpr = np.interp(fpr_common, boot_fpr[i], boot_tpr[i])
        interpolated_tpr_list.append(interpolated_tpr)
    interpolated_tpr_array = np.array(interpolated_tpr_list)
        
    lower_percentile = np.percentile(interpolated_tpr_array, 2.5, axis = 0)
    upper_percentile = np.percentile(interpolated_tpr_array, 97.5, axis = 0)

    plt.figure()
    plt.fill_between(fpr_common, lower_percentile, upper_percentile, color = "lightskyblue", alpha = 0.75, label = "Bootstrapped 95% CI")
    plt.plot(test_fpr, test_tpr, label = "ROC Curve (area = %0.2f)" % test_auroc, color = "royalblue")
    plt.plot([0.0, 1.0], [0.0, 1.0], label = "No Confidence", color = "r")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC Curve of {name} (Test Data, Clinical Features)")
    plt.legend(loc = "lower right")
    plt.savefig(f"ROCCurve_{name}_TestData_clinfeats.jpg", dpi = 300)
    plt.show()


# END OF CLINICAL ONLY MODEL #
"""









# Run this block with the above protein block defining X and y #
"""
# RUN THIS SECTION WHEN PROTEOMIC MODEL IS NEEDED #
###     PROTEIN VALUES     ###
results = []

models = {
    "LR": {
        "model":
            LogisticRegression(random_state = 42),
        "param_grid": {
            # "LR__C": [0.001, 0.01, 0.1, 1.0, 10],
            "LR__C": [1.0, 3, 10, 30, 100, 300],                                        #TUNING1
            "LR__penalty": ["l1", "l2"],
            "LR__solver": ["liblinear"],
            "LR__max_iter": [1000, 3000, 5000]}
        },
    "RF": {
        "model": 
            RandomForestClassifier(random_state = 42),
        "param_grid": {
            # "RF__n_estimators": [100, 200, 300, 500],
            "RF__n_estimators": [100, 200, 300, 400, 500, 600, 700],                     #TUNING1
            "RF__max_depth": [None, 5, 8],
            # "RF__min_samples_split": [2, 3, 4, 5],                            
            "RF__min_samples_split": [2, 3, 4, 5, 6, 7, 8]}                              #TUNING1
        },
    "SVM_L": {
        "model":
        CalibratedClassifierCV(
            SVC(random_state = 42, probability = True),
            method = "sigmoid",
            cv = sKf,
            ensemble = False),
    "param_grid":{
        "SVM_L__estimator__C": [0.001, 0.01, 0.1, 1.0, 10],
        "SVM_L__estimator__kernel": ["linear"]}
    },
    "SVM_RBF": {
        "model":
        CalibratedClassifierCV(
            SVC(random_state = 42, probability = True),
            method = "sigmoid",
            cv = sKf,
            ensemble = False),
    "param_grid":{
        "SVM_RBF__estimator__C": [0.01, 0.1, 1.0, 10],
        "SVM_RBF__estimator__gamma": [0.001, 0.01, 0.1, 1.0, 1.0],
        "SVM_RBF__estimator__kernel": ["rbf"]}
    }
}

print("Models Successfully Created! :)")

for name, info in models.items():
    selected_feats = []
    remaining_feats = list(X.columns)
    trace = []

    # When Rebuilding 5 Feature Model, change (1,8) to (1,6) #
    for step in range (1,8):
        step_best_feat = None
        step_best_AUC = -np.inf
        step_best_params = []
        step_best_acc = []
        
        for candidate_feat in remaining_feats:
            current_feats = selected_feats + [candidate_feat]
            
            step_X_train = X_train[current_feats]
            
            preprocess = ColumnTransformer([
                ("contin", Pipeline([
                    ("imputer", IterativeImputer(random_state = 42)),
                    ("scaler", StandardScaler())
                ]), make_column_selector(dtype_include = ["int64", "float64"])),
                ("categ", Pipeline([
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ]), make_column_selector(dtype_include = ["category"])
                 )
            ])
            
            pipeline = Pipeline([
                ("preprocess", preprocess),
                (name, info["model"])
        ])

            grid = GridSearchCV(
                pipeline,
                info["param_grid"],
                cv = sKf,
                scoring = "roc_auc",
                return_train_score = True,
                refit = True,
                error_score = "raise",
                n_jobs = -1
            )

            grid.fit(step_X_train, y_train)

            Grid_CV_Res = pd.DataFrame(grid.cv_results_)
             
            mean_cv_AUC = grid.best_score_

            if mean_cv_AUC > step_best_AUC:
                step_best_feat = candidate_feat
                step_best_AUC = mean_cv_AUC
                step_best_params = grid.best_params_
        selected_feats.append(step_best_feat)
        remaining_feats.remove(step_best_feat)
        print(f"new feature {step_best_feat} added!")
        
        trace.append({
            "Step": step,
            "Selected Feature": step_best_feat,
            "Features Selected": selected_feats.copy(),
            "AUC": step_best_AUC,
            "Parameters": step_best_params
        })
    trace_results = pd.DataFrame(trace)
  
    step_X_train = X_train[selected_feats]
    step_X_test = X_test[selected_feats]
        
    cleaned_params = {
        parameter.removeprefix(f"{name}__"): value
        for parameter, value in step_best_params.items()
    }
    
    refit_pipeline = Pipeline([
        ("preprocessor", preprocess),
        (name, info["model"].set_params(**cleaned_params))
    ])
    
    training_all = refit_pipeline.fit(step_X_train, y_train)
    

    ### TRAINING DATA PREDICTIONS ###
    train_prob = training_all.predict_proba(step_X_train)[:, 1]
    train_pred_binary = (train_prob >= 0.5).astype(int)
    
    train_fpr, train_tpr, _ = roc_curve(y_train, train_prob, pos_label = 1)
    train_auroc = auc(train_fpr, train_tpr)
    print("Training AUC Value: ", train_auroc)

    ### TEST DATA PREDICTIONS ###
    test_prob = training_all.predict_proba(step_X_test)[:, 1]
    test_pred_binary = (test_prob >= 0.5).astype(int)
  
    test_fpr, test_tpr, _ = roc_curve(y_test, test_prob, pos_label = 1)
    test_auroc = auc(test_fpr, test_tpr)
    print("Test AUC Value: ", test_auroc)
    
    results.append({
        "Model": name,
        "Feature(s)": selected_feats,
        "Best Parameter Combination": step_best_params,
        "Train AUC Value": train_auroc,
        "Test AUC": test_auroc,
        "Train - Test AUC": train_auroc - test_auroc})
    
    results_df = pd.DataFrame(results)

    # CONFUSION MATRIX OF TRAINING DATA #
    train_confmatrix = confusion_matrix(y_train, train_pred_binary)
    flip_traincm = train_confmatrix[:, [1,0]]
    flipped_traincm = flip_traincm[[1,0], :]
    flipped_traincm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = flipped_traincm,
        display_labels = [1, 0])
    flipped_traincm_display.plot(cmap = plt.cm.Blues, text_kw = {"fontsize": 18})
    plt.title(f"{name} Confusion Matrix of Train Data (Protein Features)")
    plt.savefig(f"ConfusionMatrix_{name}_TrainData_protfeats.jpg", dpi = 300)
    plt.show()
        
    # CONFUSION MATRIX OF TEST DATA #
    test_confmatrix = confusion_matrix(y_test, test_pred_binary)
    flip_testcm = test_confmatrix[:, [1, 0]]
    flipped_testcm = flip_testcm[[1,0], :]
    flipped_testcm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = flipped_testcm,
        display_labels = [1, 0])
    flipped_testcm_display.plot(cmap = plt.cm.Blues, text_kw = {"fontsize": 18})
    plt.title(f"{name} Confusion Matrix of Test Data (Protein Features)")
    plt.savefig(f"ConfusionMatrix_{name}_TestData_protfeats.jpg", dpi = 300)
    plt.show()

    # ROC CURVE OF TRAINING DATA #
    plt.figure()
    plt.plot(train_fpr, train_tpr, label = "ROC Curve (area = %0.2f)" % train_auroc, color = "royalblue")
    plt.plot([0.0, 1.0], [0.0, 1.0], label = "No Confidence", color = "r")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC Curve of {name} (Training Data, Protein Features)")
    plt.legend()
    plt.savefig(f"ROCCurve_{name}_TrainingData_protfeats.jpg", dpi = 300)
    plt.show()

    # ROC CURVE OF TEST DATA #
    # BOOTSTRAPPING DATA #
    n_iterations = 1000

    boot_auc_scores = []
    boot_fpr = []
    boot_tpr = []
    interpolated_tpr_list = []

    test_samples = np.arange(len(y_test))

    for i in range(n_iterations):
        samples = resample(test_samples, replace = True)
        X_samples = step_X_test.iloc[samples]
        y_samples = y_test[samples]

        boot_prob = training_all.predict_proba(X_samples)[:, 1]
        boot_auroc = roc_auc_score(y_samples, boot_prob)
        boot_auc_scores.append(boot_auroc)
        
        fpr_iter, tpr_iter, _ = roc_curve(y_samples, boot_prob, pos_label = 1)
        boot_fpr.append(fpr_iter)
        boot_tpr.append(tpr_iter)
        
    fpr_common = np.linspace(0, 1, num = 500, endpoint = True)
        
    for i in range(n_iterations):
        interpolated_tpr = np.interp(fpr_common, boot_fpr[i], boot_tpr[i])
        interpolated_tpr_list.append(interpolated_tpr)
    interpolated_tpr_array = np.array(interpolated_tpr_list)
        
    lower_percentile = np.percentile(interpolated_tpr_array, 2.5, axis = 0)
    upper_percentile = np.percentile(interpolated_tpr_array, 97.5, axis = 0)

    plt.figure()
    plt.fill_between(fpr_common, lower_percentile, upper_percentile, color = "lightskyblue", alpha = 0.75, label = "Bootstrapped 95% CI")
    plt.plot(test_fpr, test_tpr, label = "ROC Curve (area = %0.2f)" % test_auroc, color = "royalblue")
    plt.plot([0.0, 1.0], [0.0, 1.0], label = "No Confidence", color = "r")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC Curve of {name} (Test Data, Protein Features)")
    plt.legend(loc = "lower right")
    plt.savefig(f"ROCCurve_{name}_TestData_protfeats.jpg", dpi = 300)
    plt.show()

  
    # SHAP
    explainer = shap.Explainer(training_all.predict_proba, step_X_train)
    shap_values = explainer(step_X_test)
    shap.plots.beeswarm(shap_values[:, :, 1])
    plt.savefig(f"SHAP_{name}_protfeats.jpg", dpi = 300)



    #   SVM Linear Feature vs AUC   #

    n_iterations_2 = 1000
    
    k = len(trace_results)
    boot_feat_auc = [[] for _ in range(k)]
    model_step = [None] * k
    lower_ci = [None] * k
    upper_ci = [None] * k 
    point_auc = [None] * k
    np.random.seed(42)
    
    #Defining test sample size
    train_samples_2 = np.arange(len(y_train))
    
    #Pulling best feats and params from ffs and subsetting X_train
    for f in range(len(trace_results)):
        features_for_step = trace_results["Features Selected"][f]
        params_for_step = trace_results["Parameters"][f]
        step_X_train_boot = X_train[features_for_step]
    
        cleaned_boot_params = {
            parameter.removeprefix(f"{name}__"):value
            for parameter, value in params_for_step.items()} 
        cloned_model = clone(info["model"])
        cloned_model.set_params(**cleaned_boot_params)


    #Building best model for each step as above code only uses final model of 7 best feats and params
        model_boot = Pipeline([
            ("preprocess", clone(preprocess)),
            (name, cloned_model)
        ])
        model_boot.fit(step_X_train_boot, y_train)
    
    #Saving best model for each step
        model_step[f] = model_boot
        
    #Bootstrapping - resampling test set 
    for i in range(n_iterations_2):
        samples_2 = resample(train_samples_2, replace = True)
        X_samp_2 = X_train.iloc[samples_2]
        y_samp_2 = y_train[samples_2]
        
        for f in range(len(trace_results)):
            features_for_step = trace_results["Features Selected"][f]
            X_boot = X_samp_2[features_for_step]
            boot_model_step = model_step[f]
            
            boot_probs = boot_model_step.predict_proba(X_boot)[:, 1]
            auc_boot = roc_auc_score(y_samp_2, boot_probs)
            boot_feat_auc[f].append(auc_boot)
        
    for f in range(len(trace_results)):    
        lower_ci[f] = np.percentile(boot_feat_auc[f], 2.5)
        upper_ci[f] = np.percentile(boot_feat_auc[f], 97.5)
        point_auc[f] = trace_results["AUC"][f]
            
    plt.figure(figsize = (10, 6))
    plt.ylim([0, 1.0])
    plt.plot(
        trace_results["Step"], 
        trace_results["AUC"], 
            marker = "o", 
            label = "Features", 
            color = "black")
    plt.plot(trace_results["Step"], lower_ci, color = "blue", linestyle = "--")
    plt.plot(trace_results["Step"], upper_ci, color = "blue", linestyle = "--")
    plt.fill_between(trace_results["Step"], lower_ci, upper_ci, color = "lightskyblue", alpha = 0.75, label = "Bootstrap 95% CI")
    for i, r in trace_results.iterrows():
        plt.text(
            r["Step"],
            r["AUC"] + 0.030,
            r["Selected Feature"],
            fontsize = 10,
            color = "black",
            ha = "center",
            fontweight = "bold")   
    plt.xlabel("Number of Features")
    plt.ylabel("AUROC Curve Score")
    plt.title(f"Feature vs AUC Plot (SVM_L)")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.savefig(f"FeatvsAUC_SVM_L_protfeats.jpg", dpi = 300)
    plt.show()


# END OF PROTEIN MODEL #
"""









# Run this block with the above feature selected block defining X and y #
"""
# RUN THIS SECTION WHEN FEATURE SELECTED MODEL IS NEEDED #
###     CLINICAL AND PROTEIN VALUES     ###
results = []

models = {
    "LR": {
        "model":
            LogisticRegression(random_state = 42),
        "param_grid": {
            "LR__C": [0.001, 0.01, 0.1, 1.0, 10],
            "LR__penalty": ["l1", "l2"],
            "LR__solver": ["liblinear"],
            "LR__max_iter": [1000, 3000, 5000]}
        },
    "RF": {
        "model": 
            RandomForestClassifier(random_state = 42),
        "param_grid": {
    #         "RF__n_estimators": [100, 200, 300, 500],                  
            "RF__n_estimators": [100, 200, 300, 400, 500, 600, 700],     #TUNING1
            "RF__max_depth": [None, 5, 8],                               
    #         "RF__min_samples_split": [2, 3, 4, 5],                     
            "RF__min_samples_split": [2, 3, 4, 5, 6, 7, 8]               #TUNING1
        }
    },
    "SVM_L": {
        "model":
        CalibratedClassifierCV(
            SVC(random_state = 42, probability = True),
            method = "sigmoid",
            cv = sKf,
            ensemble = False),
    "param_grid":{
        "SVM_L__estimator__C": [0.001, 0.01, 0.1, 1.0, 10],
        "SVM_L__estimator__kernel": ["linear"]}
    },
    "SVM_RBF": {
        "model":
        CalibratedClassifierCV(
            SVC(random_state = 42, probability = True),
            method = "sigmoid",
            cv = sKf,
            ensemble = False),
    "param_grid":{
        "SVM_RBF__estimator__C": [0.01, 0.1, 1.0, 10],
        "SVM_RBF__estimator__gamma": [0.001, 0.01, 0.1, 1.0, 1.0],
        "SVM_RBF__estimator__kernel": ["rbf"]}    
    }
}


print("Models Created Successfully!")

for name, info in models.items():
    selected_feats = []
    remaining_feats = list(X.columns)
    trace = []
    
    # Update range to (1,6) when rebuilding 5 feature model
    for step in range (1,8):
        step_best_feat = None
        step_best_AUC = -np.inf
        step_best_params = []
        step_best_acc = []
        
        for candidate_feat in remaining_feats:
            current_feats = selected_feats + [candidate_feat]
            
            step_X_train = X_train[current_feats]
            
            preprocess = ColumnTransformer([
                ("contin", Pipeline([
                    ("imputer", IterativeImputer(max_iter = 25, random_state = 42)),
                    ("scaler", StandardScaler())
                ]), make_column_selector(dtype_include = ["int64", "float64"])),
                ("categ", Pipeline([
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ]), make_column_selector(dtype_include = ["category"])
                 )
            ])

            pipeline = Pipeline([
                ("preprocess", preprocess),
                (name, info["model"])
        ])
            
            
            grid = GridSearchCV(
                pipeline,
                info["param_grid"],
                cv = sKf,
                scoring = "roc_auc",
                return_train_score = True,
                refit = True,
                error_score = "raise",
                n_jobs = -1
            )

            grid.fit(step_X_train, y_train)

            grid_CV_Res = pd.DataFrame(grid.cv_results_)
            
            mean_cv_AUC = grid.best_score_
            
            if mean_cv_AUC > step_best_AUC:
                step_best_feat = candidate_feat
                step_best_AUC = mean_cv_AUC
                step_best_params = grid.best_params_
        selected_feats.append(step_best_feat)
        remaining_feats.remove(step_best_feat)
        print(f"new feature {step_best_feat} added!")
        
        trace.append({
            "Step": step,
            "Selected Feature": step_best_feat,
            "Features Selected": selected_feats.copy(),
            "AUC": step_best_AUC,
            "Parameters": step_best_params
        })
    trace_results = pd.DataFrame(trace)
    
    step_X_train = X_train[selected_feats]
    step_X_test = X_test[selected_feats]
        
    cleaned_params = {
        parameter.removeprefix(f"{name}__"): value
        for parameter, value in step_best_params.items()
    }
    
    refit_pipeline = Pipeline([
        ("preprocessor", preprocess),
        (name, info["model"].set_params(**cleaned_params))
    ])
    
    training_all = refit_pipeline.fit(step_X_train, y_train)
    

    ### TRAINING DATA PREDICTIONS ###
    train_prob = training_all.predict_proba(step_X_train)[:, 1]
    train_pred_binary = (train_prob >= 0.5).astype(int)
    
    train_fpr, train_tpr, _ = roc_curve(y_train, train_prob, pos_label = 1)
    train_auroc = auc(train_fpr, train_tpr)
    print("Training AUC Value: ", train_auroc)

    ### TEST DATA PREDICTIONS ###
    test_prob = training_all.predict_proba(step_X_test)[:, 1]
    test_pred_binary = (test_prob >= 0.5).astype(int)
  
    test_fpr, test_tpr, _ = roc_curve(y_test, test_prob, pos_label = 1)
    test_auroc = auc(test_fpr, test_tpr)
    print("Test AUC Value: ", test_auroc)
    
    results.append({
        "Model": name,
        "Feature(s)": selected_feats,
        "Best Parameter Combination": step_best_params,
        "Train AUC Value": train_auroc,
        "Test AUC": test_auroc,
        "Train - Test AUC": train_auroc - test_auroc})
    
    results_df = pd.DataFrame(results)

    # CONFUSION MATRIX OF TRAINING DATA #
    train_confmatrix = confusion_matrix(y_train, train_pred_binary)
    flip_traincm = train_confmatrix[:, [1,0]]
    flipped_traincm = flip_traincm[[1,0], :]
    flipped_traincm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = flipped_traincm,
        display_labels = [1, 0])
    flipped_traincm_display.plot(cmap = plt.cm.Blues, text_kw = {"fontsize": 18})
    plt.title(f"{name} Confusion Matrix of Train Data (Feature Selection)")
    plt.savefig(f"ConfusionMatrix_{name}_TrainData_clin+proteins.jpg", dpi = 300)
    plt.show()
        
    # CONFUSION MATRIX OF TEST DATA #
    test_confmatrix = confusion_matrix(y_test, test_pred_binary)
    flip_testcm = test_confmatrix[:, [1, 0]]
    flipped_testcm = flip_testcm[[1,0], :]
    flipped_testcm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix = flipped_testcm,
        display_labels = [1, 0])
    flipped_testcm_display.plot(cmap = plt.cm.Blues, text_kw = {"fontsize": 18})
    plt.title(f"{name} Confusion Matrix of Test Data (Feature Selection)")
    plt.savefig(f"ConfusionMatrix_{name}_TestData_clin+proteins.jpg", dpi = 300)
    plt.show()

    # ROC CURVE OF TRAINING DATA #
    plt.figure()
    plt.plot(train_fpr, train_tpr, label = "ROC Curve (area = %0.2f)" % train_auroc, color = "royalblue")
    plt.plot([0.0, 1.0], [0.0, 1.0], label = "No Confidence", color = "r")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC Curve of {name} (Training Data, Feature Selection)")
    plt.legend()
    plt.savefig(f"ROCCurve_{name}_TrainingData_clin+proteins.jpg",  dpi = 300)
    plt.show()

    # ROC CURVE OF TEST DATA #
    # BOOTSTRAPPING DATA #
    n_iterations = 1000

    boot_auc_scores = []
    boot_fpr = []
    boot_tpr = []
    interpolated_tpr_list = []

    test_samples = np.arange(len(y_test))

    for i in range(n_iterations):
        samples = resample(test_samples, replace = True)
        X_samples = step_X_test.iloc[samples]
        y_samples = y_test[samples]

        boot_prob = training_all.predict_proba(X_samples)[:, 1]
        boot_auroc = roc_auc_score(y_samples, boot_prob)
        boot_auc_scores.append(boot_auroc)
        
        fpr_iter, tpr_iter, _ = roc_curve(y_samples, boot_prob, pos_label = 1)
        boot_fpr.append(fpr_iter)
        boot_tpr.append(tpr_iter)
        
    fpr_common = np.linspace(0, 1, num = 500, endpoint = True)
        
    for i in range(n_iterations):
        interpolated_tpr = np.interp(fpr_common, boot_fpr[i], boot_tpr[i])
        interpolated_tpr_list.append(interpolated_tpr)
    interpolated_tpr_array = np.array(interpolated_tpr_list)
        
    lower_percentile = np.percentile(interpolated_tpr_array, 2.5, axis = 0)
    upper_percentile = np.percentile(interpolated_tpr_array, 97.5, axis = 0)

    plt.figure()
    plt.fill_between(fpr_common, lower_percentile, upper_percentile, color = "lightskyblue", alpha = 0.75, label = "Bootstrapped 95% CI")
    plt.plot(test_fpr, test_tpr, label = "ROC Curve (area = %0.2f)" % test_auroc, color = "royalblue")
    plt.plot([0.0, 1.0], [0.0, 1.0], label = "No Confidence", color = "r")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC Curve of {name} (Test Data, Feature Selection)")
    plt.legend(loc = "lower right")
    plt.savefig(f"ROCCurve_{name}_TestData_clin+proteins.jpg", dpi = 300)
    plt.show()


    # SHAP
    explainer = shap.Explainer(training_all.predict_proba, step_X_train)
    shap_values = explainer(step_X_test)
    shap.plots.beeswarm(shap_values[:, :, 1])
    plt.savefig(f"SHAP{name}_clin+proteins.jpg", dpi = 300)
    


    # LR Feature vs AUC #
    
    n_iterations_2 = 1000
    
    k = len(trace_results)
    boot_feat_auc = [[] for _ in range(k)]
    model_step = [None] * k
    lower_ci = [None] * k
    upper_ci = [None] * k 
    point_auc = [None] * k
    np.random.seed(42)
    
    #Defining test sample size
    train_samples_2 = np.arange(len(y_train))
    
    #Pulling best feats and params from ffs and subsetting X_train
    for f in range(len(trace_results)):
        features_for_step = trace_results["Features Selected"][f]
        params_for_step = trace_results["Parameters"][f]
        step_X_train_boot = X_train[features_for_step]
    
        cleaned_boot_params = {
            parameter.removeprefix(f"{name}__"):value
            for parameter, value in params_for_step.items()} 
        cloned_model = clone(info["model"])
    
        cloned_model.set_params(**cleaned_boot_params)
    
    #Building best model for each step as above code only uses final model of 7 best feats and params
        model_boot = Pipeline([
            ("preprocess", clone(preprocess)),
            (name, cloned_model)
        ])
        model_boot.fit(step_X_train_boot, y_train)
    
    #Saving best model for each step
        model_step[f] = model_boot
        
    #Bootstrapping - resampling test set 
    for i in range(n_iterations_2):
        samples_2 = resample(train_samples_2, replace = True)
        X_samp_2 = X_train.iloc[samples_2]
        y_samp_2 = y_train[samples_2]
        
        for f in range(len(trace_results)):
            features_for_step = trace_results["Features Selected"][f]
            X_boot = X_samp_2[features_for_step]
            boot_model_step = model_step[f]
            
            boot_probs = boot_model_step.predict_proba(X_boot)[:, 1]
            auc_boot = roc_auc_score(y_samp_2, boot_probs)
            boot_feat_auc[f].append(auc_boot)
        
    for f in range(len(trace_results)):    
        lower_ci[f] = np.percentile(boot_feat_auc[f], 2.5)
        upper_ci[f] = np.percentile(boot_feat_auc[f], 97.5)
        point_auc[f] = trace_results["AUC"][f]
            
            
    plt.figure(figsize = (10, 6))
    plt.ylim([0, 1.0])
    plt.plot(
        trace_results["Step"], 
        trace_results["AUC"], 
            marker = "o", 
            label = "Features", 
            color = "black")
    plt.plot(trace_results["Step"], lower_ci, color = "blue", linestyle = "--")
    plt.plot(trace_results["Step"], upper_ci, color = "blue", linestyle = "--")
    plt.fill_between(trace_results["Step"], lower_ci, upper_ci, color = "lightskyblue", alpha = 0.75, label = "Bootstrap 95% CI")
    for i, r in trace_results.iterrows():
        plt.text(
            r["Step"],
            r["AUC"] + 0.030,
            r["Selected Feature"],
            fontsize = 10,
            color = "black",
            ha = "center",
            fontweight = "bold")   
    plt.xlabel("Number of Features")
    plt.ylabel("AUROC Curve Score")
    plt.title(f"Feature vs AUC Plot (LR)")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.savefig(f"FINAL_FeatvsAUC_LR_clin+prot.jpg",  dpi = 300)
    plt.show()


# END OF FEATURE SELECTED MODEL # 
"""
